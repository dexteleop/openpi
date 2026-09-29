"""把一段已取回的 FFMPEGPackets(IDR 关键帧 → 当前帧) 解码成当前帧图像。

只解码不读取: packets 由调用方用 MCAP_Message_Fetcher 取回, 调用方因此可以把一个 sample 的
全部 location 一起并行读取。

CPU解码器：PyAV(libavcodec)
GPU解码器：NVDEC
"""
from __future__ import annotations

import av
import numpy as np

# FFMPEGPacket.encoding 可能带像素格式后缀(如 'hevc;nv12')或用别名, 只取编码名并归一化
_CODEC_ALIASES = {"h265": "hevc", "avc": "h264"}


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


# def gpu_decode_current_frame