"""逐条播放 mcap 里的每个 message, 只靠记录自身的长度往下走。

完全不看文件元数据(Footer / Summary / Statistics / MessageIndex), 所以电脑掉电、
mcap 只录了一半、统计信息没写完的文件, 也能把已经落盘的消息全部播出来, 走到写坏
的那条记录处自然结束。

mcap 中每条 Message 记录必然位于且仅位于一个 Chunk 内, 且在该 Chunk 的
records 字节流中有唯一的起始偏移, 因此
(chunk_file_offset, uncompressed_byte_offset) 就是这条消息的定位主键。
chunk_file_offset 是该 Chunk 记录在文件中的绝对字节位置。

用法::
    for message in play_messages("/path/to/rec.mcap"):
        print(message["chunk_file_offset"],
              message["uncompressed_byte_offset"],
              message["record_length"],
              message["topic_name"], message["msg_type"], message["msg_defs"]
              message["log_time"])
"""
from __future__ import annotations
import os
import struct
from typing import Iterator
from openpi.training.lingyu_dataloader_v2.utils.topics_filter import filter_topics
from openpi.training.lingyu_dataloader_v2.robots_config.config import load_video_topics_gop

# 每个message需要得到的信息
_PER_MESSAGE_KEYS = (
    "chunk_file_offset", "uncompressed_byte_offset", "record_length",  # message_location
    "topic_name",  # resolved from channel_id via Channel records
    "msg_type",    # ROS2 type string, e.g. "sensor_msgs/msg/JointState"; passed to get_message()
    "msg_def",     # ROS2 msg definition string, used to register custom types in typestore
    "log_time",    # timestamp_ns
)

# mcap 二进制常量
_MAGIC_SIZE = 8  # b"\x89MCAP0\r\n"
_RECORD_PREFIX = 9  # opcode(1) + record_length(8)
_OP_SCHEMA  = 0x03
_OP_CHANNEL = 0x04
_OP_MESSAGE = 0x05
_OP_CHUNK   = 0x06

# Chunk 记录头: message_start_time(8) + message_end_time(8) + uncompressed_size(8)
#            + uncompressed_crc(4) + compression(4+n) + records_length(8)
_CHUNK_HEADER_FIXED = 8 + 8 + 8 + 4 + 4 + 8

_u16 = struct.Struct("<H").unpack_from
_u32 = struct.Struct("<I").unpack_from
_u64 = struct.Struct("<Q").unpack_from


def _play_messages(path: str) -> Iterator[dict[str, int]]:
    """
    按MCAP中消息保存顺序逐条产出 message 的定位信息、解析格式、筛选信息
    输入： rosbag 挂载的文件路径
    输出： 一个 message 对应 _PER_MESSAGE_KEYS 中的全部数据
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        file_size = os.fstat(fd).st_size
        channel_topics: dict[int, str] = {}   # channel_id -> topic_name, 遇到 Channel 记录时累积更新
        channel_schemas: dict[int, int] = {}   # channel_id -> schema_id
        schema_names: dict[int, str] = {}      # schema_id  -> msg_type string
        schema_msgdefs: dict[int, str] = {}    # schema_id  -> msg definition string
        pos = _MAGIC_SIZE
        while pos + _RECORD_PREFIX <= file_size:
            # --- 获取顶层记录头 (opcode + length) ---
            header = os.pread(fd, _RECORD_PREFIX, pos)
            if len(header) < _RECORD_PREFIX:
                break
            (record_length,) = _u64(header, 1)
            if header[0] == _OP_CHUNK:  # 只有 Chunk 内部才有 Message 记录
                chunk_file_offset = pos  # chunk 在文件中的绝对偏移，作为定位主键

                # --- 获取 chunk: 解析 chunk 头, 读取 chunk 内部的全部字节 ---
                chunk_head = os.pread(fd, _CHUNK_HEADER_FIXED + 64, pos + _RECORD_PREFIX)
                (name_len,) = _u32(chunk_head, 28)
                compression = chunk_head[32 : 32 + name_len].decode()
                if compression:  # 只支持未压缩 chunk
                    raise ValueError(f"不支持的 chunk 压缩方式: {compression!r}")
                (records_length,) = _u64(chunk_head, 32 + name_len)
                blob = os.pread(fd, records_length,
                                pos + _RECORD_PREFIX + _CHUNK_HEADER_FIXED + name_len)

                offset = 0
                while offset + _RECORD_PREFIX <= len(blob):
                    (inner_length,) = _u64(blob, offset + 1)
                    body_at = offset + _RECORD_PREFIX
                    if body_at + inner_length > len(blob):  # 掉电写坏的半条记录, 到此为止
                        break
                    inner_op = blob[offset]

                    # --- 读取 channel_id -> topic_name, schema_id 映射 ---
                    if inner_op == _OP_CHANNEL:
                        b = blob[body_at : body_at + inner_length]
                        cid, = _u16(b, 0); sid, = _u16(b, 2); tl, = _u32(b, 4)
                        channel_topics[int(cid)] = b[8 : 8 + tl].decode()
                        channel_schemas[int(cid)] = int(sid)
                    # --- 读取 schema_id -> msg_type 映射 ---
                    elif inner_op == _OP_SCHEMA:
                        b = blob[body_at : body_at + inner_length]
                        sid, = _u16(b, 0); nl, = _u32(b, 2)
                        name = b[6 : 6 + nl].decode()
                        enc_len, = _u32(b, 6 + nl)
                        dat_len, = _u32(b, 6 + nl + 4 + enc_len)
                        dat = b[6 + nl + 4 + enc_len + 4 : 6 + nl + 4 + enc_len + 4 + dat_len]
                        schema_names[int(sid)] = name
                        schema_msgdefs[int(sid)] = dat.decode("utf-8", errors="replace")

                    # --- 读取 message, 产出定位信息 ---
                    elif inner_op == _OP_MESSAGE:
                        cid = _u16(blob, body_at)[0]
                        sid = channel_schemas.get(cid, 0)
                        yield dict(zip(_PER_MESSAGE_KEYS,
                                       (chunk_file_offset, offset, inner_length,
                                        channel_topics.get(cid, str(cid)),   # topic_name
                                        schema_names.get(sid, ""),           # msg_type
                                        schema_msgdefs.get(sid, ""),         # msg_def
                                        _u64(blob, body_at + 6)[0])))        # log_time
                    offset = body_at + inner_length
            pos += _RECORD_PREFIX + record_length
    finally:
        os.close(fd)


_NAL_START3 = b'\x00\x00\x01'


def _iter_nal_types(data: bytes):
    """Yield NAL unit types in annex-B HEVC data (handles 3- and 4-byte start codes)."""
    pos = data.find(_NAL_START3)
    n = len(data)
    while pos != -1 and pos + 3 < n:
        yield (data[pos + 3] >> 1) & 0x3F
        pos = data.find(_NAL_START3, pos + 3)


def _has_idr(data: bytes) -> bool:
    """Check if HEVC data contains an IDR NAL unit (type 19 or 20).

    Stops at the first VCL NAL (type < 32): in a single access unit the first
    VCL NAL determines the picture type, so P-frame packets bail immediately.
    """
    for nal_type in _iter_nal_types(data):
        if nal_type < 32:
            return nal_type in (19, 20)
    return False


class MCAP_Player:
    """
    Wraps play_messages() with topic filtering and video keyframe buffering.

    Yield format (from play_messages):
        (topic_name, msg_type, msg_def, log_time, locations)

    For topics whose GOP length > 1 (inter-frame coded, see load_video_topics_gop()):
        locations = [(mcap_url, chunk_file_offset, uncompressed_byte_offset, record_length), ...]
        — all frames from the most recent IDR keyframe up to and including the current frame,
          so the decoder always has a complete GOP available.
        — frames arriving before the topic's first IDR are undecodable and are not yielded.

    For all other topics (GOP length 1, i.e. every message decodes on its own):
        locations = [(mcap_url, chunk_file_offset, uncompressed_byte_offset, record_length)]
        — single-element list.

    Only topics resolved by topics_filter.filter_topics() (USER_SELECTED_TOPICS mapped
    through TELEAVATAV2_MCAP_TOPICS_MAPPING), plus any extra_topics passed to
    play_messages(), are yielded.
    """

    # channel_id(2) + sequence(4) + log_time(8) + publish_time(8)
    _MSG_HDR_SIZE = 22

    def __init__(self, mcap_url: str):
        self.mcap_url = mcap_url

    def _read_msg_payload(self, fd: int, chunk_file_offset: int,
                          uncompressed_byte_offset: int, record_length: int) -> bytes:
        """Read the raw payload bytes of one message (strips the fixed 22-byte message header)."""
        chunk_head = os.pread(fd, _CHUNK_HEADER_FIXED + 64, chunk_file_offset + _RECORD_PREFIX)
        (name_len,) = _u32(chunk_head, 28)
        blob_start = chunk_file_offset + _RECORD_PREFIX + _CHUNK_HEADER_FIXED + name_len
        payload_offset = blob_start + uncompressed_byte_offset + _RECORD_PREFIX + self._MSG_HDR_SIZE
        payload_len = record_length - self._MSG_HDR_SIZE
        return os.pread(fd, payload_len, payload_offset)

    def _accumulate_frames_from_keyframe(
        self,
        ffmpeg_buffers: dict,
        topic_name: str,
        loc: tuple,
        fd: int,
    ) -> list:
        """
        记录下能够解码该帧图像的所有帧数据索引，方便后续操作

        Returns [I] when this frame is an IDR.
        Returns [I P ...] when an IDR (keyframe) is detected.
        Returns [] if no IDR has been seen yet for this message.
        """
        payload = self._read_msg_payload(fd, loc[1], loc[2], loc[3])
        if _has_idr(payload):
            ffmpeg_buffers[topic_name] = [loc]
        elif topic_name not in ffmpeg_buffers:
            return []  # 尚未遇到关键帧, 该帧无法解码, 丢弃
        else:
            ffmpeg_buffers[topic_name].append(loc)
        return list(ffmpeg_buffers[topic_name])

    def play_messages(self, extra_topics: tuple[str, ...] = ()):
        """
        输出： topic_name, msg_type, msg_def, log_time, locations

        extra_topics: 除用户选定数据外还需产出的 mcap topic (如标记 episode 起止的按键信号),
                      这些 topic 不属于训练数据, 故不写进 USER_SELECTED_TOPICS。
        """
        selected_mcap_topics = filter_topics().keys() | set(extra_topics)
        mcap_video_topics_gop = load_video_topics_gop()  # 键已是 mcap topic 名, 无需映射
        ffmpeg_buffers: dict[str, list] = {}
        fd = os.open(self.mcap_url, os.O_RDONLY)
        try:
            for msg in _play_messages(self.mcap_url):
                topic = msg["topic_name"]
                if topic not in selected_mcap_topics:
                    continue
                loc = (self.mcap_url,
                       msg["chunk_file_offset"],
                       msg["uncompressed_byte_offset"],
                       msg["record_length"])
                if mcap_video_topics_gop.get(topic, 1) > 1:  # GOP>1: 帧间编码, 需回溯到关键帧
                    locations = self._accumulate_frames_from_keyframe(
                        ffmpeg_buffers, topic, loc, fd)
                    if not locations:  # 无关键帧可依托, 跳过该帧
                        continue
                else:
                    locations = [loc]
                yield topic, msg["msg_type"], msg["msg_def"], msg["log_time"], locations
        finally:
            os.close(fd)
