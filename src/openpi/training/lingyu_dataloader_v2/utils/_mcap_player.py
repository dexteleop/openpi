"""逐条播放 mcap 里的每个 message, 只靠记录自身的长度往下走。

完全不看文件元数据(Footer / Summary / Statistics / MessageIndex), 所以电脑掉电、
mcap 只录了一半、统计信息没写完的文件, 也能把已经落盘的消息全部播出来, 走到写坏
的那条记录处自然结束。

mcap 中每条 Message 记录必然位于且仅位于一个 Chunk 内, 且在该 Chunk 的
records 字节流中有唯一的起始偏移, 因此
(chunk_file_offset, uncompressed_byte_offset) 就是这条消息的定位主键。
chunk_file_offset 是该 Chunk 记录在文件中的绝对字节位置。

用法::
    for message in play_messages("raw/.../rec.mcap"):
        print(message["chunk_file_offset"],
              message["uncompressed_byte_offset"],
              message["record_length"],
              message["topic_name"], message["msg_type"], message["msg_defs"]
              message["log_time"])
"""
from __future__ import annotations
from typing import Generator
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import (
    filter_topics
)
from openpi.training.lingyu_dataloader_v2.mcap_config.config import (
    load_mcap_video_topics_gop
)
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import (
    MCAP_Message_Fetcher, remember_chunk_records
)
from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    # MCAP Record 类型
    OP_CHANNEL, OP_MESSAGE, OP_CHUNK, OP_SCHEMA,
    # MCAP常见结构长度
    MAGIC_SIZE, OPCODE_PREFIX, RECORD_PREFIX, CHUNK_HEADER_FIXED,
    # 字节流转整数
    u16, u32, u64,
    # MCAP功能函数
    fetch_mcap_bytes, fetch_mcap_length, decompress_chunk_record
)

# 每个message需要得到的信息
_PER_MESSAGE_KEYS = (
    "chunk_file_offset", "uncompressed_byte_offset", "record_length",  # message_location
    "topic_name",  # resolved from channel_id via Channel records
    "msg_type",    # ROS2 type string, e.g. "sensor_msgs/msg/JointState"; passed to get_message()
    "msg_def",     # ROS2 msg definition string, used to register custom types in typestore
    "log_time",    # timestamp_ns
)

# 顺序播放的预读字节数: _fetch_buffered 缓冲未命中时, 在请求长度之外额外多读这么多字节。
# 为什么需要: 文件布局为 | Chunk | MessageIndex × C | Chunk | ..., 每个 channel 一条 MessageIndex,
#   它们是与 Chunk 同级的顶层记录。主循环不用其内容, 但仍须逐条读 9 字节 prefix 拿到长度才能跳过;
#   无缓冲时每个 chunk 后会有约 50 次只取 9 字节的 S3 往返。
# 作用: 把本 chunk 后的全部 MessageIndex 与下一个 chunk 的 prefix + head 一并取回, 之后只需对
#   下一个 chunk 的 records 再发一次请求。不压缩的 chunk 读 records 时顺带预读; 压缩的 chunk 由
#   decompress_chunk_record 单独读取, 预读由其后第一条 MessageIndex 的 prefix 未命中触发。
# 取值依据: MessageIndex 总大小 ≈ 15×channel 数 + 16×消息数, 实测 20~30KB, 64KB 约留 2 倍余量。
#   超出时只会每 64KB 多一次请求, 解析结果不变。
_SEQUENTIAL_READAHEAD = 64 * 1024


def _fetch_buffered(mcap_path: str, buffer: tuple[int, bytes], length: int,
                   offset: int) -> tuple[bytes, tuple[int, bytes]]:
    """Read [offset, offset+length) from buffer=(start, bytes), refilling it with readahead on a miss.

    Returns (data, buffer). Like fetch_mcap_bytes, data is shorter than length at end of file.
    """
    buffer_start, buffer_bytes = buffer
    if not buffer_start <= offset <= offset + length <= buffer_start + len(buffer_bytes):
        buffer_start = offset
        buffer_bytes = fetch_mcap_bytes(mcap_path, length + _SEQUENTIAL_READAHEAD, offset)
    relative_offset = offset - buffer_start
    return (buffer_bytes[relative_offset:relative_offset + length],
            (buffer_start, buffer_bytes))


def _play_messages(mcap_path: str,
                   extra_topics: tuple[str, ...] = ()) -> Generator[dict, None, None]:
    """
    按MCAP中消息保存顺序逐条产出 message 的定位信息、解析格式、筛选信息
    输入： rosbag 挂载的文件路径
    输出： 一个 message 对应 _PER_MESSAGE_KEYS 中的全部数据,
          只产出 filter_topics() 选中的 topic 与 extra_topics
    """
    selected_mcap_topics = filter_topics().keys() | set(extra_topics)
    file_size = fetch_mcap_length(mcap_path)
    channel_topics: dict[int, str] = {}    # channel_id -> topic_name, 遇到 Channel 记录时累积更新
    channel_schemas: dict[int, int] = {}   # channel_id -> schema_id
    schema_names: dict[int, str] = {}      # schema_id  -> msg_type string
    schema_msgdefs: dict[int, str] = {}    # schema_id  -> msg definition string
    pos = MAGIC_SIZE    # pos指向第一个Record
    read_buffer: tuple[int, bytes] = (0, b"")   # (缓冲区起始偏移, 字节), 顺序读取的预读缓冲
    while pos + RECORD_PREFIX <= file_size:
        # --- 获取 Record Prefix ---
        header, read_buffer = _fetch_buffered(mcap_path, read_buffer, RECORD_PREFIX, pos)
        if len(header) < RECORD_PREFIX:
            break
        top_record_length, = u64(header, OPCODE_PREFIX) # 获取当前Record长度，每个Record可能还包括Record

        # --- 只有 Chunk Record 内部才有 Message Record，因此只读取 Chunk Record ---
        if header[0] == OP_CHUNK:
            chunk_file_offset = pos  # chunk 在文件中的绝对偏移，作为定位主键

            # --- 解析 chunk header, 定位 records 起始位置 ---
            chunk_head, read_buffer = _fetch_buffered(
                mcap_path, read_buffer,
                CHUNK_HEADER_FIXED + 4, # max(name_len)=4
                pos + RECORD_PREFIX
            )
            # name_len=0,不压缩; name_len=3,lz4; name_len=4,zstd
            name_len, = u32(chunk_head, 28)
            # 读取压缩算法，空字符串,'lz4','zstd'
            compression = chunk_head[32 : 32 + name_len].decode() # '' or 'lz4' or 'zstd'
            # 读取 Chunk Header records_length
            records_length, = u64(chunk_head, 32 + name_len)
            # 从第一个record开始
            records_start = pos + RECORD_PREFIX + CHUNK_HEADER_FIXED + name_len
            # 读取整个 Chunk Record
            if compression:
                records_bytes = decompress_chunk_record(
                    mcap_path, records_start, records_length, compression,
                    u64(chunk_head, 16)[0],  # uncompressed_size, zstd 解压上界
                )
            else:
                # 顺带预读其后的 MessageIndex 与下一个 chunk 头, 本 chunk 通常只需这一次请求
                records_bytes, read_buffer = _fetch_buffered(
                    mcap_path, read_buffer, records_length, records_start)
            # 交给 fetch_data_bytes 复用: 下游按条取本 chunk 的 message 时直接切片, 不再回头请求 S3
            remember_chunk_records(mcap_path, chunk_file_offset, records_bytes)

            uncompressed_byte_offset, records_length = 0, len(records_bytes)
            while uncompressed_byte_offset + RECORD_PREFIX <= records_length:
                inner_record_length, = u64(records_bytes, uncompressed_byte_offset + OPCODE_PREFIX)
                body_at = uncompressed_byte_offset + RECORD_PREFIX
                # 掉电写坏的半条记录, 到此为止
                if body_at + inner_record_length > records_length:
                    break
                inner_op = records_bytes[uncompressed_byte_offset]

                # --- 读取 channel_id -> topic_name, schema_id 映射 ---
                # 未选中的 topic 不建映射, 其 message 在下方据此跳过
                if inner_op == OP_CHANNEL:
                    b = records_bytes[body_at : body_at + inner_record_length]
                    cid, = u16(b, 0); sid, = u16(b, 2); tl, = u32(b, 4)
                    topic_name = b[8 : 8 + tl].decode()
                    if topic_name in selected_mcap_topics:
                        channel_topics[int(cid)] = topic_name
                        channel_schemas[int(cid)] = int(sid)

                # --- 读取 schema_id -> msg_type 映射 ---
                elif inner_op == OP_SCHEMA:
                    b = records_bytes[body_at : body_at + inner_record_length]
                    sid, = u16(b, 0); nl, = u32(b, 2)
                    name = b[6 : 6 + nl].decode()
                    enc_len, = u32(b, 6 + nl)
                    dat_len, = u32(b, 6 + nl + 4 + enc_len)
                    dat = b[6 + nl + 4 + enc_len + 4 : 6 + nl + 4 + enc_len + 4 + dat_len]
                    schema_names[int(sid)] = name
                    schema_msgdefs[int(sid)] = dat.decode("utf-8", errors="replace")

                # --- 读取 message, 产出定位信息 ---
                elif inner_op == OP_MESSAGE:
                    cid, = u16(records_bytes, body_at)
                    # cid 不在映射中即该 topic 未被选中, 跳过读取下一条 message
                    if cid in channel_topics:
                        sid = channel_schemas[cid]
                        yield dict(zip(_PER_MESSAGE_KEYS,
                                       (chunk_file_offset,
                                        uncompressed_byte_offset,
                                        inner_record_length,  # record_length
                                        channel_topics[cid],  # topic_name
                                        schema_names.get(sid, ""),  # msg_type
                                        schema_msgdefs.get(sid, ""),  # msg_def
                                        u64(records_bytes, body_at + 6)[0])))  # log_time
                uncompressed_byte_offset = body_at + inner_record_length

        pos += RECORD_PREFIX + top_record_length  # pos 指向下一个 Record


_NAL_START3 = b'\x00\x00\x01'


def _detect_codec(data: bytes) -> str:
    """Detect H.264 or H.265 by scanning for codec-exclusive NAL types.

    H.265 VPS/SPS/PPS have types 32/33/34 (6-bit formula: (b >> 1) & 0x3F).
    H.264 SPS/PPS have types 7/8 (5-bit formula: b & 0x1F).
    Falls back to 'h265' when no discriminating NAL is found.
    """
    pos = data.find(_NAL_START3)
    n = len(data)
    while pos != -1 and pos + 3 < n:
        b = data[pos + 3]
        if (b >> 1) & 0x3F in (32, 33, 34):  # H.265 VPS / SPS / PPS
            return 'h265'
        if b & 0x1F in (7, 8):               # H.264 SPS / PPS
            return 'h264'
        pos = data.find(_NAL_START3, pos + 3)
    return 'h265'  # 默认与原有行为保持一致


def _iter_nal_types(data: bytes, codec: str = 'h265'):
    """Yield NAL unit types from Annex-B bitstream (handles 3- and 4-byte start codes).

    H.265: nal_unit_type = (b >> 1) & 0x3F (6-bit, 2-byte header).
    H.264: nal_unit_type = b & 0x1F        (5-bit, 1-byte header).
    """
    pos = data.find(_NAL_START3)
    n = len(data)
    while pos != -1 and pos + 3 < n:
        b = data[pos + 3]
        yield ((b >> 1) & 0x3F) if codec == 'h265' else (b & 0x1F)
        pos = data.find(_NAL_START3, pos + 3)


def _has_idr(data: bytes) -> bool:
    """Check if Annex-B data contains an IDR keyframe. Auto-detects H.264/H.265.

    H.265: IDR types 19 (IDR_W_RADL) / 20 (IDR_N_LP); bails at first VCL (type < 32).
    H.264: IDR type 5; bails at first VCL (type 1–5).
    """
    codec = _detect_codec(data)
    for nal_type in _iter_nal_types(data, codec):
        if codec == 'h265':
            if nal_type < 32:
                return nal_type in (19, 20)
        else:  # h264
            if 1 <= nal_type <= 5:
                return nal_type == 5
    return False


class MCAP_Player:
    """
    Wraps play_messages() with video keyframe buffering.

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

    Topic filtering happens inside _play_messages(): it only produces topics resolved by
    topics_filter.filter_topics() (USER_SELECTED_TOPICS mapped through TELEAVATAV2_MCAP_TOPICS_MAPPING), plus the
    extra_topics forwarded from play_messages(); this class does not filter again.
    """

    def __init__(self, mcap_url: str):
        self.mcap_url = mcap_url
        # 关键帧判断需反序列化出 FFMPEGPacket.data, fetcher 自带 typestore 与自定义类型注册
        self._fetcher = MCAP_Message_Fetcher()

    def _accumulate_frames_from_keyframe(
        self,
        ffmpeg_buffers: dict,
        topic_name: str,
        msg_type: str,
        msg_def: str,
        loc: tuple,
    ) -> list:
        """
        记录下能够解码该帧图像的所有帧数据索引，方便后续操作

        Returns [I] when this frame is an IDR.
        Returns [I P ...] when an IDR (keyframe) is detected.
        Returns [] if no IDR has been seen yet for this message.
        """
        # 只检测 data 字段的码流: header/encoding 等字段的字节可能凑出 00 00 01 伪起始码,
        # 使 IDR 漏检, GOP 列表无限增长(OOM); data 在消息内偏移随字符串长度变化, 故须反序列化
        payload = bytes(self._fetcher.fetch_message(msg_type, msg_def, [loc])[0].data)
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
        mcap_video_topics_gop = load_mcap_video_topics_gop()  # 键已是 mcap topic 名, 无需映射
        ffmpeg_buffers: dict[str, list] = {}

        # 顺序播放一个 mcap 中的 message
        msg_iter = _play_messages(self.mcap_url, extra_topics)

        for msg in msg_iter:
            topic = msg["topic_name"]
            loc = (self.mcap_url,
                   msg["chunk_file_offset"],
                   msg["uncompressed_byte_offset"],
                   msg["record_length"])
            if mcap_video_topics_gop.get(topic,1) > 1:  # GOP>1: 帧间编码, 需回溯到关键帧
                locations = self._accumulate_frames_from_keyframe(
                    ffmpeg_buffers, topic, msg["msg_type"], msg["msg_def"], loc)
                if not locations:  # 无关键帧可依托, 跳过该帧
                    continue
            else:
                locations = [loc]
            # 将定位整数与时间戳统一转为字符串后对外产出
            locations_str = [(loc[0], str(loc[1]), str(loc[2]), str(loc[3]))
                             for loc in locations]
            yield topic, msg["msg_type"], msg["msg_def"], str(msg["log_time"]), locations_str
