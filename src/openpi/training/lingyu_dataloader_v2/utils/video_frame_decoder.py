"""Decode a fetched span of FFMPEGPackets (IDR keyframe -> current frame) into the current frame image.

Decoding only, no fetching: packets are fetched by the caller with MCAP_Message_Fetcher, so the caller can read
all locations of one sample in parallel.

CPU decoder: PyAV (libavcodec)
GPU decoder: NVDEC (hevc_cuvid etc.); decoding, mono-eye cropping and scaling all happen on the GPU, and only the small scaled image is copied back to memory

GPU decoding is concentrated in one decode process per GPU (start_gpu_decoders); the DataLoader's workers only send packets
over and receive small images: one process's CUDA context is about 420MB, and 256 workers each building one would exhaust GPU memory.
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

# FFMPEGPacket.encoding may carry a pixel-format suffix (e.g. 'hevc;nv12') or use an alias; take only the codec name and normalize it
_CODEC_ALIASES = {"h265": "hevc", "avc": "h264"}
# Decode threads per GPU decode process, each thread holding its own resident decoders; measured on A800, 8 threads reach about 1750 fps (4K HEVC), and more does not help
GPU_DECODE_THREADS = 8
# cuvid output lags input by 1 frame: producing the GOP's last frame requires feeding the keyframe extra times to push it out; if it still has not appeared after this many extra feeds, decoding is deemed faulty
MAX_PUSH_PACKETS = 8
# Maximum seconds to wait for a decode process to start (spawn must re-import the main module)
GPU_DECODER_START_TIMEOUT = 300
# Up to hundreds of workers may connect to one decode process; the default backlog=1 is not enough
GPU_DECODER_BACKLOG = 1024
# pts is only used as the input packet's sequence number; any time base works
_PTS_TIME_BASE = Fraction(1, 1000)

logger = logging.getLogger(__name__)

# Main process: started decode processes and their listening addresses; reused directly when the dataset is created again
_gpu_decoder_processes = []
_gpu_decoder_addresses = []
# Worker processes: connections to each decode process, established on first request
_gpu_decoder_connections = {}
# Decode process: each decode thread's own {(topic, eye, rotate, resolution): GpuStreamDecoder}
_thread_decoders = threading.local()


def _codec_name(encoding: str) -> str:
    """FFMPEGPacket.encoding -> libavcodec decoder name."""
    codec = (encoding or "").split(";")[0].strip().lower()
    return _CODEC_ALIASES.get(codec, codec) or "hevc"


def cpu_decode_current_frame(packets: list, pixel_format: str = "rgb24") -> np.ndarray:
    """Decode the [I,P,P,...] span of FFMPEGPackets, returning the pixel array of its last frame (the current frame).

    packets are ordered by locations; the first element is guaranteed to be an IDR (MCAP_Player guarantees this)

    Returns:
        ndarray, shape (height, width, 3) for rgb24
    """
    assert packets, "没有任何 FFMPEGPacket 可解码"
    codec_ctx = av.CodecContext.create(_codec_name(packets[0].encoding), "r")
    frames = []
    for packet in packets:
        frames.extend(codec_ctx.decode(av.Packet(bytes(packet.data))))
    frames.extend(codec_ctx.decode(None))  # flush: retrieve frames buffered inside the decoder
    if not frames:
        raise ValueError(f"{len(packets)} 个 FFMPEGPacket 未解出任何帧")
    return frames[-1].to_ndarray(format=pixel_format)


def eye_crop_and_size(width: int, height: int, eye: str, rotate: bool,
                      resolution: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    """Compute cuvid's crop parameters and the scaled (width, height), with geometry exactly matching openpi's image transforms:

    TeleavatarInputs._extract_stereo_view: width >= 2x height marks a side-by-side stereo frame; when rotating, the whole frame is rotated 180° first,
    then the left eye takes [:half] and the right eye takes [half:]; rotate-then-crop equals cropping the opposite side's columns on the original frame and rotating after the crop, with the rotation applied by the caller to the small scaled image.
    ResizeImages -> resize_with_pad: scales without distortion by the same formula; black-border padding is left to ResizeImages itself.
    resolution is (height, width), in the same order as ResizeImages(height, width).
    """
    crop_left = crop_right = 0
    if width >= 2 * height:
        half = width // 2
        # Width of the "left block" on the original frame: after a 180° rotation [:half] corresponds to the original's [width-half:], so rotation swaps the two blocks' widths
        left_block_width = width - half if rotate else half
        if (eye == "left") != rotate:
            crop_right, width = width - left_block_width, left_block_width
        else:
            crop_left, width = left_block_width, width - left_block_width
    ratio = max(width / resolution[1], height / resolution[0])
    # cuvid crop format is (top)x(bottom)x(left)x(right)
    return f"0x0x{crop_left}x{crop_right}", (int(width / ratio), int(height / ratio))


class GpuStreamDecoder:
    """One video stream's resident NVDEC decoder, reused across samples: flush destroys and rebuilds the cuvid decoder (about 40ms), so it never flushes.

    Every GOP starts with an IDR, so feeding can simply continue from the previous span. Input packets use increasing sequence numbers as pts, and output frames carry the same pts,
    recognizing the GOP's last frame by pts; residual frames decoded from keyframes fed to push frames out carry smaller pts and are naturally skipped.
    """

    def __init__(self, encoding: str, gpu_idx: int, first_packet: bytes, eye: str, rotate: bool,
                 resolution: tuple[int, int]):
        # Decode the first IDR on the CPU, solely to get resolution and color range (cuvid's output frames carry no color range)
        probe_ctx = av.CodecContext.create(_codec_name(encoding), "r")
        probe_frame = (probe_ctx.decode(av.Packet(first_packet)) + probe_ctx.decode(None))[0]
        self.color_range = probe_frame.color_range
        crop, (resized_width, resized_height) = eye_crop_and_size(
            probe_frame.width, probe_frame.height, eye, rotate, resolution)
        self.rotate = rotate  # Cropping already took the opposite side's columns, so the decoded small image still needs a 180° rotation
        # primary_ctx: all decoders in this process share the GPU's primary context instead of each building an about-420MB CUDA context
        hwaccel = HWAccel("cuda", device=str(gpu_idx), allow_software_fallback=False, options={"primary_ctx": "1"})
        self.codec_ctx = av.CodecContext.create(f"{_codec_name(encoding)}_cuvid", "r", hwaccel=hwaccel)
        self.codec_ctx.options = {"crop": crop, "resize": f"{resized_width}x{resized_height}"}
        self.codec_ctx.flags |= Flags.low_delay  # Do not accumulate a display queue; output lags by only 1 frame
        self.next_pts = 0

    def decode_current_frame(self, packets: list[bytes]) -> np.ndarray:
        """Decode a whole GOP, returning the last frame's (H, W, 3) uint8 RGB array (mono-cropped and scaled)."""
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
        # Same as _extract_stereo_view's np.rot90(k=2); rotating the small scaled image copies only one 224-sized image
        return np.ascontiguousarray(rgb_frame[::-1, ::-1]) if self.rotate else rgb_frame

    def _decode_packet(self, packet_bytes: bytes, target_pts: int):
        """Feed one packet, returning the frame with pts == target_pts among this call's outputs, or None if absent."""
        packet = av.Packet(packet_bytes)
        packet.pts, packet.time_base = self.next_pts, _PTS_TIME_BASE
        self.next_pts += 1
        return next((frame for frame in self.codec_ctx.decode(packet) if frame.pts == target_pts), None)


def _decode_in_thread(gpu_idx: int, topic: str, encoding: str, eye: str, rotate: bool,
                      resolution: tuple[int, int], packets: list[bytes]) -> np.ndarray:
    """In a decode thread, decode one GOP with this thread's resident decoder for that video stream; on decode error the decoder is discarded and rebuilt next time."""
    stream_decoders = _thread_decoders.__dict__.setdefault("by_stream", {})
    # Crop and scale are fixed when the decoder is built, so even the same topic with different crop parameters builds separate decoders
    stream_key = (topic, eye, rotate, resolution)
    if stream_key not in stream_decoders:
        stream_decoders[stream_key] = GpuStreamDecoder(encoding, gpu_idx, packets[0], eye, rotate, resolution)
    try:
        return stream_decoders[stream_key].decode_current_frame(packets)
    except Exception:
        del stream_decoders[stream_key]  # Input and output pts no longer line up; the decoder's state cannot be trusted
        raise


def _serve_connection(connection, gpu_idx: int, thread_pool: ThreadPoolExecutor) -> None:
    """Serve one worker's connection: receive all video requests of one sample, hand each stream to decode threads in parallel, and return current frames in request order.

    On decode error an exception (with the in-process traceback) is returned and re-raised on the worker side; when the worker exits the connection closes and this thread ends.
    """
    while True:
        try:
            video_requests = connection.recv()
        except (EOFError, OSError):
            return  # Worker exited; OSError("got end of file during message") when terminate hits mid-request
        frame_futures = [thread_pool.submit(_decode_in_thread, gpu_idx, *video_request)
                         for video_request in video_requests]
        try:
            connection.send([frame_future.result() for frame_future in frame_futures])
        except (BrokenPipeError, ConnectionResetError):
            return  # Worker already terminated (the DataLoader terminates still-sampling workers on shutdown), so nobody will receive the result
        except Exception:
            connection.send(RuntimeError(f"GPU {gpu_idx} 解码失败:\n{traceback.format_exc()}"))


def serve_gpu_decoder(address: str, gpu_idx: int, ready_event) -> None:
    """Decode process entry: listen on address, one send/receive thread per connection, decoding delegated to GPU_DECODE_THREADS decode threads."""
    # authkey comes from the main process: children started via spawn/forkserver all inherit it; other processes cannot connect
    listener = Listener(address, backlog=GPU_DECODER_BACKLOG, authkey=multiprocessing.current_process().authkey)
    thread_pool = ThreadPoolExecutor(max_workers=GPU_DECODE_THREADS)
    ready_event.set()
    while True:
        connection = listener.accept()
        threading.Thread(target=_serve_connection, args=(connection, gpu_idx, thread_pool), daemon=True).start()


def start_gpu_decoders() -> list[str]:
    """Start one decode process per visible GPU in the main process and return their listening addresses; reuse directly if already started.

    Decode processes are daemons and end when the main process exits.
    """
    if _gpu_decoder_addresses:
        return _gpu_decoder_addresses
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "没有可见的 GPU, 无法进行 GPU 解码"
    # spawn: the main process has usually already initialized CUDA (JAX), and forked children cannot use CUDA;
    # nor forkserver: it would start the server before the DataLoader sets the forkserver preload, defeating the preload
    mp_context = multiprocessing.get_context("spawn")
    ready_events = []
    for gpu_idx in range(num_gpus):
        # Linux abstract-namespace unix socket, no file on disk, nothing to clean up
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
    """Send one sample's video requests to the decode process at address and receive each stream's current frame (mono-cropped, scaled uint8 RGB).

    video_requests: [(topic, encoding, eye, rotate, resolution, [packet bytes, IDR first]), ...]; the return order matches it.
    """
    if address not in _gpu_decoder_connections:
        _gpu_decoder_connections[address] = Client(address, authkey=multiprocessing.current_process().authkey)
    connection = _gpu_decoder_connections[address]
    connection.send(video_requests)
    current_frames = connection.recv()
    if isinstance(current_frames, Exception):
        raise current_frames
    return current_frames
