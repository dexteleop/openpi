"""
利用 ("mcap_url", "chunk_file_offset", "uncompressed_byte_offset", "record_length") 来找到一个message
"""
from __future__ import annotations
from rosbags.typesys import (
    Stores, get_typestore, get_types_from_msg,
)

from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    # MCAP Record 类型
    OP_MESSAGE,
    # MCAP常见结构长度
    RECORD_PREFIX, MESSAGE_HEADER, CHUNK_HEADER_FIXED,
    # 字节流转整数
    u32, u64,
    # MCAP功能函数
    fetch_mcap_bytes, decompress_chunk_record
)


def fetch_data_bytes(
    mcap_url: str,
    chunk_file_offset: int,
    uncompressed_byte_offset: int,
    msg_record_length: int,
) -> bytes:
    """按 (chunk_file_offset, uncompressed_byte_offset) O(1) 定位并读取单条 message。
    未压缩 chunk 无需任何扫描，两次 pread 完成定位与读取；
    压缩 chunk(zstd/lz4) 先整块解压，再用同一偏移在解压流内切片。
    chunk_file_offset 是 chunk 记录在文件中的绝对字节位置.

    Returns:
        cdr_bytes
    """
    # --- 解析 chunk header, 定位 records 起始位置 ---
    chunk_head = fetch_mcap_bytes(
        mcap_url,
        CHUNK_HEADER_FIXED + 4, # max(name_len)=4
        chunk_file_offset + RECORD_PREFIX # 跳过 Chunk Record Prefix
    )
    # 读取 Chunk Header compression 中的前缀值。
    # name_len=0,不压缩; name_len=3,lz4; name_len=4,zstd
    (name_len,) = u32(chunk_head, 28)
    # 读取压缩算法，空字符串,'lz4','zstd'
    compression = chunk_head[32 : 32 + name_len].decode() # '' or 'lz4' or 'zstd'
    # 读取 Chunk Header records_length
    (records_length,) = u64(chunk_head, 32 + name_len)
    # 从第一个record开始
    records_start = chunk_file_offset + RECORD_PREFIX + CHUNK_HEADER_FIXED + name_len

    if compression:
        # 压缩 chunk: uncompressed_byte_offset 是解压流内的偏移，须先整块解压
        records_bytes = decompress_chunk_record(
            mcap_url, records_start, records_length, compression,
            u64(chunk_head, 16)[0],   # uncompressed_size, zstd 解压上界
        )
        msg_record = records_bytes[uncompressed_byte_offset:
                            uncompressed_byte_offset + RECORD_PREFIX + msg_record_length]
    else:
        # 未压缩 chunk: uncompressed_byte_offset 即文件内偏移，一次 range 读取目标 message 记录
        msg_record = fetch_mcap_bytes(mcap_url, RECORD_PREFIX + msg_record_length,
                               records_start + uncompressed_byte_offset)

    if msg_record[0] != OP_MESSAGE:
        raise ValueError(
            f"(chunk_file_offset={chunk_file_offset}, "
            f"uncompressed_byte_offset={uncompressed_byte_offset}) "
            f"处不是 Message 记录 (opcode=0x{msg_record[0]:02x})"
        )
    return msg_record[RECORD_PREFIX+MESSAGE_HEADER:]


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
        """若 msg_type 不在标准库且尚未注册, 用 msg_def 动态注册进 typestore"""
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

        对 locations 中每个定位元组调用一次 fetch_data_bytes + deserialize_cdr,
        返回 list[ros2 message]。调用方根据 msg_type 决定后续处理（如 FFMPEGPacket 的视频解码）。
        locations 长度为 1 时返回单元素列表; 长度 > 1 (ffmpeg GOP) 时返回整段帧的列表。
        """
        self._ensure_type_registered(msg_type, msg_def)
        message = []
        for loc in locations:
            data_bytes = fetch_data_bytes(loc[0], int(loc[1]), int(loc[2]), int(loc[3]))
            message.append(
                self._typestore.deserialize_cdr(data_bytes, msg_type)
            )
        return message
