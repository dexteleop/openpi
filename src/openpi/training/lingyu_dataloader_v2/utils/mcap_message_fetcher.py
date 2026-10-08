"""
利用 ("mcap_url", "chunk_file_offset", "uncompressed_byte_offset", "record_length") 来找到一个message
"""
from __future__ import annotations
import threading

from rosbags.typesys import (
    Stores, get_typestore, get_types_from_msg,
)

from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    # MCAP Record 类型
    OP_MESSAGE,
    # MCAP常见结构长度
    OPCODE_PREFIX, RECORD_PREFIX, MESSAGE_HEADER, CHUNK_HEADER_FIXED,
    # 字节流转整数
    u32, u64,
    # MCAP功能函数
    fetch_mcap_bytes, decompress_chunk_record
)

# 最近一个被整块读入内存的 chunk: (mcap_url, chunk_file_offset, records_bytes)。
# 顺序播放时 player 刚读完一个 chunk, 紧接着就会按条取其中的 message(关键帧判断、按键解析),
# 命中即直接切片, 免去每条 message 两次 S3 往返; 只存一块, 内存恒定
_last_chunk_records: tuple[str, int, bytes] | None = None


def remember_chunk_records(mcap_url: str, chunk_file_offset: int, records_bytes: bytes) -> None:
    """Keep the chunk just read whole so fetch_data_bytes() can slice it instead of hitting S3."""
    global _last_chunk_records
    # 整个元组一次赋值: 读方拿到的要么是旧块要么是新块, 不会读到拼了一半的状态
    _last_chunk_records = (mcap_url, chunk_file_offset, records_bytes)


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
    last_chunk = _last_chunk_records
    if last_chunk is not None and last_chunk[:2] == (mcap_url, chunk_file_offset):
        # 命中刚播完的 chunk: records_bytes 与下方两条路径读到的是同一段字节流, 直接切片
        msg_record = last_chunk[2][uncompressed_byte_offset:
                                   uncompressed_byte_offset + RECORD_PREFIX + msg_record_length]
        return _message_cdr(msg_record, chunk_file_offset, uncompressed_byte_offset)

    # --- 解析 chunk header, 定位 records 起始位置 ---
    chunk_head = fetch_mcap_bytes(
        mcap_url,
        CHUNK_HEADER_FIXED + 4, # max(name_len)=4
        chunk_file_offset + RECORD_PREFIX # 跳过 Chunk Record Prefix
    )
    # 读取 Chunk Header compression 中的前缀值。
    # name_len=0,不压缩; name_len=3,lz4; name_len=4,zstd
    name_len, = u32(chunk_head, 28)
    # 读取压缩算法，空字符串,'lz4','zstd'
    compression = chunk_head[32 : 32 + name_len].decode() # '' or 'lz4' or 'zstd'
    # 读取 Chunk Header records_length
    records_length, = u64(chunk_head, 32 + name_len)
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
    return _message_cdr(msg_record, chunk_file_offset, uncompressed_byte_offset)


def _message_cdr(msg_record: bytes, chunk_file_offset: int, uncompressed_byte_offset: int) -> bytes:
    """Check msg_record really is a Message record and strip its headers, leaving the CDR bytes."""
    if msg_record[0] != OP_MESSAGE:
        raise ValueError(
            f"(chunk_file_offset={chunk_file_offset}, "
            f"uncompressed_byte_offset={uncompressed_byte_offset}) "
            f"处不是 Message 记录 (opcode=0x{msg_record[0]:02x})"
        )
    return msg_record[RECORD_PREFIX+MESSAGE_HEADER:]


def slice_span_cdr(span_bytes: bytes, span_start: int,
                   uncompressed_byte_offset: int, msg_record_length: int) -> bytes | None:
    """从按未压缩 chunk 读出的 records 区字节段里切出一条 message 的 CDR。

    span_bytes 是 records 区内从 span_start 起的一段; 切出的记录 opcode 与长度字段都须与索引一致,
    否则返回 None(如 chunk 实际被压缩, 文件内偏移与解压流内偏移不同), 由调用方退回 fetch_data_bytes。
    """
    record_pos = uncompressed_byte_offset - span_start
    assert record_pos >= 0, f"message 偏移 {uncompressed_byte_offset} 在读取段起点 {span_start} 之前"
    msg_record = span_bytes[record_pos: record_pos + RECORD_PREFIX + msg_record_length]
    if (len(msg_record) != RECORD_PREFIX + msg_record_length
            or msg_record[0] != OP_MESSAGE or u64(msg_record, OPCODE_PREFIX)[0] != msg_record_length):
        return None
    return msg_record[RECORD_PREFIX + MESSAGE_HEADER:]


class MCAP_Message_Fetcher:
    """
    按 locations 随机读取并反序列化 mcap 中的 message。

    所有 topic 均将 CDR 字节反序列化为 rosbags message 对象后返回，包括 FFMPEGPacket。
    FFMPEGPacket 的视频解码（→ 图像像素）由调用方负责。
    fetch_message 可被多线程并发调用: 唯一的共享可变状态(typestore 注册)由锁保护。
    """

    def __init__(self):
        self._typestore = get_typestore(Stores.ROS2_HUMBLE)
        self._registered_types: set[str] = set()  # 已注册的自定义类型名, 避免重复 register
        self._register_lock = threading.Lock()    # 多线程首次遇到同一类型时, 只让一个线程去 register

    def _ensure_type_registered(self, msg_type: str, msg_def: str) -> None:
        """若 msg_type 不在标准库且尚未注册, 用 msg_def 动态注册进 typestore"""
        if msg_type in self._registered_types:
            return
        with self._register_lock:
            if msg_type in self._registered_types:  # 等锁期间已被别的线程注册
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

    def fetch_span_messages(self, mcap_url: str, chunk_file_offset: int, span_messages: list[tuple]) -> list:
        """一次 range 读取同一 chunk 内一段相邻的 message, 再逐条切片、反序列化。

        span_messages 为 [(msg_type, msg_def, uncompressed_byte_offset, record_length), ...], 可跨 topic;
        按未压缩 chunk 直接算出 records 区起点, 省去读 chunk header 的那次请求。
        切片校验不通过的 message(压缩 chunk)退回 fetch_data_bytes 逐条读取。返回顺序与 span_messages 一致。
        """
        assert span_messages, "span_messages 不能为空"
        span_start = min(span_message[2] for span_message in span_messages)
        span_end = max(span_message[2] + RECORD_PREFIX + span_message[3] for span_message in span_messages)
        # 未压缩 chunk 的 header 中 compression 为空串(name_len=0), records 区紧跟其后
        records_start = chunk_file_offset + RECORD_PREFIX + CHUNK_HEADER_FIXED
        span_bytes = fetch_mcap_bytes(mcap_url, span_end - span_start, records_start + span_start)

        messages = []
        for msg_type, msg_def, uncompressed_byte_offset, msg_record_length in span_messages:
            cdr_bytes = slice_span_cdr(span_bytes, span_start, uncompressed_byte_offset, msg_record_length)
            if cdr_bytes is None:
                cdr_bytes = fetch_data_bytes(mcap_url, chunk_file_offset, uncompressed_byte_offset, msg_record_length)
            self._ensure_type_registered(msg_type, msg_def)
            messages.append(self._typestore.deserialize_cdr(cdr_bytes, msg_type))
        return messages
