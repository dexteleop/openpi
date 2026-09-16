"""把 locations 指向的一段 FFMPEGPackets(IDR 关键帧 → 当前帧) 解码成当前帧图像。

CPU解码器：PyAV(libavcodec)
GPU解码器：NVDEC
"""
from __future__ import annotations

import av
import numpy as np

from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher

# FFMPEGPacket.encoding 可能带像素格式后缀(如 'hevc;nv12')或用别名, 只取编码名并归一化
_CODEC_ALIASES = {"h265": "hevc", "avc": "h264"}

_fetcher = MCAP_Message_Fetcher()


def _codec_name(encoding: str) -> str:
    """FFMPEGPacket.encoding -> libavcodec 解码器名。"""
    codec = (encoding or "").split(";")[0].strip().lower()
    return _CODEC_ALIASES.get(codec, codec) or "hevc"


def cpu_decode_current_frame(msg_type: str, msg_def: str, locations: list[tuple],
                             pixel_format: str = "rgb24") -> np.ndarray:
    """解码 locations 覆盖的 [I,P,P,...], 返回其中最后一帧(即当前帧)的像素数组。

    locations 首元素肯定是 IDR (MCAP_Player 已保证)

    Returns:
        ndarray, shape (height, width, 3) for rgb24
    """
    packets = _fetcher.fetch_message(msg_type, msg_def, locations)
    codec_ctx = av.CodecContext.create(_codec_name(packets[0].encoding), "r")
    frames = []
    for packet in packets:
        frames.extend(codec_ctx.decode(av.Packet(bytes(packet.data))))
    frames.extend(codec_ctx.decode(None))  # flush: 取出解码器内缓存的帧
    if not frames:
        raise ValueError(f"{len(packets)} 个 FFMPEGPacket 未解出任何帧")
    return frames[-1].to_ndarray(format=pixel_format)


# def gpu_decode_current_frame