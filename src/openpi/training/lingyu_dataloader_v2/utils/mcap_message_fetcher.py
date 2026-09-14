"""
利用 ("mcap_url", "chunk_file_offset", "uncompressed_byte_offset", "record_length") 来找到一个message
"""
from __future__ import annotations

import os
import struct

from rosbags.typesys import Stores, get_typestore, get_types_from_msg

_MAGIC_SIZE = 8
_RECORD_PREFIX = 9          # opcode(1) + record_length(8)
_MESSAGE_HEADER = 22        # channel_id(2) + sequence(4) + log_time(8) + publish_time(8)
_CHUNK_HEADER_FIXED = 8 + 8 + 8 + 4 + 4 + 8
_OP_SCHEMA = 0x03
_OP_MESSAGE = 0x05
_OP_CHUNK = 0x06

_u16 = struct.Struct("<H").unpack_from
_u32 = struct.Struct("<I").unpack_from
_u64 = struct.Struct("<Q").unpack_from


def _fetch_data_bytes(
    mcap_url: str,
    chunk_file_offset: int,
    uncompressed_byte_offset: int,
    record_length: int,
) -> bytes:
    """按 (chunk_file_offset, uncompressed_byte_offset) O(1) 定位并读取单条 message。
    无需任何扫描，两次 pread 完成定位与读取。仅支持未压缩 chunk。
    chunk_file_offset 是 chunk 记录在文件中的绝对字节位置.

    Returns:
        cdr_bytes
    """
    fd = os.open(mcap_url, os.O_RDONLY)
    try:
        # 解析 chunk header，定位 records 数据区起始
        head = os.pread(fd, _CHUNK_HEADER_FIXED + 4, chunk_file_offset + _RECORD_PREFIX)
        (name_len,) = _u32(head, 28)
        compression = head[32 : 32 + name_len].decode()
        if compression:
            raise ValueError(f"不支持的 chunk 压缩方式: {compression!r}")
        data_start = chunk_file_offset + _RECORD_PREFIX + _CHUNK_HEADER_FIXED + name_len

        # 一次 pread 读取目标 message 记录
        buf = os.pread(fd, _RECORD_PREFIX + record_length,
                       data_start + uncompressed_byte_offset)
        if buf[0] != _OP_MESSAGE:
            raise ValueError(
                f"(chunk_file_offset={chunk_file_offset}, offset={uncompressed_byte_offset}) "
                f"处不是 Message 记录 (opcode=0x{buf[0]:02x})"
            )
        body = buf[_RECORD_PREFIX:]
        return body[_MESSAGE_HEADER:]
    finally:
        os.close(fd)


class MCAP_Message_Fetcher:
    """
    按 locations 随机读取并反序列化 mcap 中的 message。

    所有 topic 均将 CDR 字节反序列化为 rosbags message 对象后返回，包括 FFMPEGPacket。
    FFMPEGPacket 的视频解码（→ 图像像素）由调用方负责。
    """

    def __init__(self):
        self._typestore = get_typestore(Stores.ROS2_HUMBLE)
        self._registered_types: set[str] = set()  # 已注册的自定义类型名, 避免重复 register

    def _ensure_type_registered(self, msg_type: str, msg_def: str) -> None:
        """若 msg_type 不在标准库且尚未注册, 用 msg_def 动态注册进 typestore。"""
        if msg_type in self._registered_types:
            return
        try:
            self._typestore.deserialize_cdr(b"\x00\x01\x00\x00", msg_type)
        except Exception:
            self._typestore.register(get_types_from_msg(msg_def, msg_type))
        self._registered_types.add(msg_type)

    def fetch_message(self, msg_type: str, msg_def: str, locations: list[tuple]):
        """
        输入 mcap_player.play_messages 产出的 msg_type/msg_def/locations, 返回反序列化后的 message 列表。

        对 locations 中每个定位元组调用一次 _fetch_data_bytes + deserialize_cdr,
        返回 list[ros2 message]。调用方根据 msg_type 决定后续处理（如 FFMPEGPacket 的视频解码）。
        locations 长度为 1 时返回单元素列表; 长度 > 1 (ffmpeg GOP) 时返回整段帧的列表。
        """
        self._ensure_type_registered(msg_type, msg_def)
        return [self._typestore.deserialize_cdr(_fetch_data_bytes(*loc), msg_type)
                for loc in locations]

