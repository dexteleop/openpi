"""把一段已取回的 FFMPEGPackets(IDR 关键帧 → 当前帧) 解码成当前帧图像。

只解码不读取: packets 由调用方用 MCAP_Message_Fetcher 取回, 调用方因此可以把一个 sample 的
全部 location 一起并行读取。

CPU解码器：PyAV(libavcodec)
GPU解码器：NVDEC(hevc_cuvid 等), 解码、裁出单目、缩放全在 GPU 上完成, 只把缩放后的小图拷回内存

GPU 解码集中在每块 GPU 一个的解码进程里(start_gpu_decoders), DataLoader 的 worker 只把 packets
发过去、收回小图: 一个进程的 CUDA 上下文约 420MB, 256 个 worker 各建一个会把显存占满。
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from multiprocessing.connection import Client, Listener

import av
from av.codec.context import Flags
from av.codec.hwaccel import HWAccel
import numpy as np
import torch

# FFMPEGPacket.encoding 可能带像素格式后缀(如 'hevc;nv12')或用别名, 只取编码名并归一化
_CODEC_ALIASES = {"h265": "hevc", "avc": "h264"}
# 每个 GPU 解码进程的解码线程数, 每个线程各持一套常驻解码器; A800 实测 8 线程约 1750 帧/s(4K HEVC), 再多不涨
GPU_DECODE_THREADS = 8
# cuvid 输出比输入晚 1 帧: 取 GOP 最后一帧要再补喂关键帧把它推出来, 补喂超过这么多次仍没出来即判为解码异常
MAX_PUSH_PACKETS = 8
# 等解码进程启动(spawn 要重新 import 主模块)的最长秒数
GPU_DECODER_START_TIMEOUT = 300
# 同时连入一个解码进程的 worker 可达数百个, 默认 backlog=1 不够
GPU_DECODER_BACKLOG = 1024
# pts 只当输入 packet 的序号用, 时间基任取
_PTS_TIME_BASE = Fraction(1, 1000)

logger = logging.getLogger(__name__)

# 主进程: 已启动的解码进程与其监听地址, 重复创建 dataset 时直接复用
_gpu_decoder_processes = []
_gpu_decoder_addresses = []
# worker 进程: 到各解码进程的连接, 首次请求时建立
_gpu_decoder_connections = {}
# 解码进程: 每个解码线程各自的 {(topic, eye, rotate, resolution): GpuStreamDecoder}
_thread_decoders = threading.local()


def _codec_name(encoding: str) -> str:
    """FFMPEGPacket.encoding -> libavcodec 解码器名。"""
    codec = (encoding or "").split(";")[0].strip().lower()
    return _CODEC_ALIASES.get(codec, codec) or "hevc"


def cpu_decode_current_frame(packets: list, pixel_format: str = "rgb24") -> np.ndarray:
    """解码 [I,P,P,...] 这段 FFMPEGPacket, 返回其中最后一帧(即当前帧)的像素数组。

    packets 按 locations 顺序排列, 首元素肯定是 IDR (MCAP_Player 已保证)

    Returns:
        ndarray, shape (height, width, 3) for rgb24
    """
    assert packets, "没有任何 FFMPEGPacket 可解码"
    codec_ctx = av.CodecContext.create(_codec_name(packets[0].encoding), "r")
    frames = []
    for packet in packets:
        frames.extend(codec_ctx.decode(av.Packet(bytes(packet.data))))
    frames.extend(codec_ctx.decode(None))  # flush: 取出解码器内缓存的帧
    if not frames:
        raise ValueError(f"{len(packets)} 个 FFMPEGPacket 未解出任何帧")
    return frames[-1].to_ndarray(format=pixel_format)


def eye_crop_and_size(width: int, height: int, eye: str, rotate: bool,
                      resolution: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    """求 cuvid 的 crop 参数与缩放后的 (宽, 高), 几何与 openpi 的图像变换完全一致:

    TeleavatarInputs._extract_stereo_view: 宽 >= 2 倍高才算左右拼接的双目帧, rotate 时先整帧转 180°,
    再左目取 [:half], 右目取 [half:]; 先转再裁等价于在原帧上裁对侧对应的列、裁完再转, 旋转由调用方对缩放后的小图做。
    ResizeImages -> resize_with_pad: 按同一公式不变形缩放, 补黑边留给 ResizeImages 自己做。
    resolution 为 (高, 宽), 与 ResizeImages(height, width) 同序。
    """
    crop_left = crop_right = 0
    if width >= 2 * height:
        half = width // 2
        # 原帧上"左块"的宽: 转 180° 后的 [:half] 对应原帧的 [width-half:], 故旋转时左右两块的宽互换
        left_block_width = width - half if rotate else half
        if (eye == "left") != rotate:
            crop_right, width = width - left_block_width, left_block_width
        else:
            crop_left, width = left_block_width, width - left_block_width
    ratio = max(width / resolution[1], height / resolution[0])
    # cuvid crop 的格式为 (top)x(bottom)x(left)x(right)
    return f"0x0x{crop_left}x{crop_right}", (int(width / ratio), int(height / ratio))


class GpuStreamDecoder:
    """一路视频流的常驻 NVDEC 解码器, 跨 sample 复用: flush 会销毁重建 cuvid 解码器(约 40ms), 故从不 flush。

    每个 GOP 都以 IDR 开头, 直接接着上一段喂即可。输入 packet 以递增序号作 pts, 输出帧带回同一 pts,
    按 pts 认出 GOP 的最后一帧; 上一次为推帧补喂的关键帧解出的残帧 pts 更小, 自然被跳过。
    """

    def __init__(self, encoding: str, gpu_idx: int, first_packet: bytes, eye: str, rotate: bool,
                 resolution: tuple[int, int]):
        # CPU 解一次首个 IDR, 只为拿到分辨率与色彩范围(cuvid 输出的帧不带色彩范围)
        probe_ctx = av.CodecContext.create(_codec_name(encoding), "r")
        probe_frame = (probe_ctx.decode(av.Packet(first_packet)) + probe_ctx.decode(None))[0]
        self.color_range = probe_frame.color_range
        crop, (resized_width, resized_height) = eye_crop_and_size(
            probe_frame.width, probe_frame.height, eye, rotate, resolution)
        self.rotate = rotate  # 裁剪已取了对侧的列, 解出的小图还要再转 180°
        # primary_ctx: 同进程内全部解码器共用 GPU 的主上下文, 不再各建一个约 420MB 的 CUDA 上下文
        hwaccel = HWAccel("cuda", device=str(gpu_idx), allow_software_fallback=False, options={"primary_ctx": "1"})
        self.codec_ctx = av.CodecContext.create(f"{_codec_name(encoding)}_cuvid", "r", hwaccel=hwaccel)
        self.codec_ctx.options = {"crop": crop, "resize": f"{resized_width}x{resized_height}"}
        self.codec_ctx.flags |= Flags.low_delay  # 不攒显示队列, 输出只晚 1 帧
        self.next_pts = 0

    def decode_current_frame(self, packets: list[bytes]) -> np.ndarray:
        """解码一整段 GOP, 返回其最后一帧(已裁单目、已缩放)的 (H, W, 3) uint8 RGB 数组。"""
        target_pts = self.next_pts + len(packets) - 1
        current_frame = None
        for packet_bytes in packets:
            decoded_frame = self._decode_packet(packet_bytes, target_pts)
            current_frame = current_frame if decoded_frame is None else decoded_frame
        push_count = 0
        while current_frame is None:
            assert push_count < MAX_PUSH_PACKETS, f"补喂 {push_count} 个关键帧仍未输出 pts={target_pts} 的帧"
            current_frame = self._decode_packet(packets[0], target_pts)
            push_count += 1
        rgb_frame = current_frame.to_ndarray(format="rgb24", src_color_range=self.color_range)
        # 与 _extract_stereo_view 的 np.rot90(k=2) 相同; 在缩放后的小图上转, 拷贝量只有一张 224 图
        return np.ascontiguousarray(rgb_frame[::-1, ::-1]) if self.rotate else rgb_frame

    def _decode_packet(self, packet_bytes: bytes, target_pts: int):
        """喂一个 packet, 返回本次输出中 pts == target_pts 的帧, 没有则返回 None。"""
        packet = av.Packet(packet_bytes)
        packet.pts, packet.time_base = self.next_pts, _PTS_TIME_BASE
        self.next_pts += 1
        return next((frame for frame in self.codec_ctx.decode(packet) if frame.pts == target_pts), None)


def _decode_in_thread(gpu_idx: int, topic: str, encoding: str, eye: str, rotate: bool,
                      resolution: tuple[int, int], packets: list[bytes]) -> np.ndarray:
    """在解码线程里用本线程该路视频的常驻解码器解一段 GOP; 解码出错就丢掉该解码器, 下次重建。"""
    stream_decoders = _thread_decoders.__dict__.setdefault("by_stream", {})
    # 裁剪与缩放在建解码器时就定下了, 故同一 topic 的裁剪参数不同也要分开建
    stream_key = (topic, eye, rotate, resolution)
    if stream_key not in stream_decoders:
        stream_decoders[stream_key] = GpuStreamDecoder(encoding, gpu_idx, packets[0], eye, rotate, resolution)
    try:
        return stream_decoders[stream_key].decode_current_frame(packets)
    except Exception:
        del stream_decoders[stream_key]  # 输入输出的 pts 已对不上, 解码器状态不可信
        raise


def _serve_connection(connection, gpu_idx: int, thread_pool: ThreadPoolExecutor) -> None:
    """服务一个 worker 的连接: 收一个 sample 的全部视频请求, 各路交给解码线程并行解, 按请求顺序回传当前帧。

    解码出错时回传异常(附解码进程内的调用栈), 由 worker 端重新抛出; worker 退出时连接断开, 本线程随之结束。
    """
    while True:
        try:
            video_requests = connection.recv()
        except (EOFError, OSError):
            return  # worker 退出; 请求发到一半被 terminate 时是 OSError("got end of file during message")
        frame_futures = [thread_pool.submit(_decode_in_thread, gpu_idx, *video_request)
                         for video_request in video_requests]
        try:
            connection.send([frame_future.result() for frame_future in frame_futures])
        except (BrokenPipeError, ConnectionResetError):
            return  # worker 已被终止(DataLoader 关闭时会 terminate 还在取样的 worker), 结果无人接收
        except Exception:
            connection.send(RuntimeError(f"GPU {gpu_idx} 解码失败:\n{traceback.format_exc()}"))


def serve_gpu_decoder(address: str, gpu_idx: int, ready_event) -> None:
    """解码进程入口: 监听 address, 每个连接一个收发线程, 解码统一交给 GPU_DECODE_THREADS 个解码线程。"""
    # authkey 用主进程的: spawn/forkserver 起的子进程都继承它, 其他进程连不上
    listener = Listener(address, backlog=GPU_DECODER_BACKLOG, authkey=multiprocessing.current_process().authkey)
    thread_pool = ThreadPoolExecutor(max_workers=GPU_DECODE_THREADS)
    ready_event.set()
    while True:
        connection = listener.accept()
        threading.Thread(target=_serve_connection, args=(connection, gpu_idx, thread_pool), daemon=True).start()


def start_gpu_decoders() -> list[str]:
    """在主进程里给每块可见 GPU 起一个解码进程, 返回各自的监听地址; 已启动过则直接复用。

    解码进程是 daemon, 主进程退出时随之结束。
    """
    if _gpu_decoder_addresses:
        return _gpu_decoder_addresses
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "没有可见的 GPU, 无法进行 GPU 解码"
    # spawn: 主进程此时多半已初始化 CUDA(JAX), fork 出的子进程用不了 CUDA;
    # 也不用 forkserver: 会抢在 DataLoader 设置 forkserver preload 之前把 server 起起来, 使 preload 失效
    mp_context = multiprocessing.get_context("spawn")
    ready_events = []
    for gpu_idx in range(num_gpus):
        # Linux 抽象命名空间的 unix socket, 不落文件, 无需清理
        address = f"\0lingyu_gpu_decoder_{os.getpid()}_{gpu_idx}"
        ready_event = mp_context.Event()
        decoder_process = mp_context.Process(target=serve_gpu_decoder, args=(address, gpu_idx, ready_event),
                                             name=f"gpu_decoder_{gpu_idx}", daemon=True)
        decoder_process.start()
        _gpu_decoder_processes.append(decoder_process)
        _gpu_decoder_addresses.append(address)
        ready_events.append(ready_event)
    for decoder_process, ready_event in zip(_gpu_decoder_processes, ready_events):
        assert ready_event.wait(GPU_DECODER_START_TIMEOUT) and decoder_process.is_alive(), \
            f"{decoder_process.name} 启动失败, exitcode={decoder_process.exitcode}"
    logger.info(f"已在 {num_gpus} 块 GPU 上各起一个解码进程, 每个 {GPU_DECODE_THREADS} 个解码线程")
    return _gpu_decoder_addresses


def gpu_decode_current_frames(address: str, video_requests: list[tuple]) -> list[np.ndarray]:
    """把一个 sample 的各路视频请求发给 address 处的解码进程, 收回各路当前帧(已裁单目、已缩放的 uint8 RGB)。

    video_requests: [(topic, encoding, eye, rotate, resolution, [packet bytes, IDR 在前]), ...], 返回顺序与之一致。
    """
    if address not in _gpu_decoder_connections:
        _gpu_decoder_connections[address] = Client(address, authkey=multiprocessing.current_process().authkey)
    connection = _gpu_decoder_connections[address]
    connection.send(video_requests)
    current_frames = connection.recv()
    if isinstance(current_frames, Exception):
        raise current_frames
    return current_frames
