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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import (
    filter_topics
)
from openpi.training.lingyu_dataloader_v2.mcap_config.config import (
    load_mcap_video_topics_gop
)
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import (
    fetch_data_bytes
)
from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    # MCAP Record 类型
    OP_CHANNEL, OP_MESSAGE, OP_CHUNK, OP_SCHEMA, OP_MESSAGE_IDX, OP_CHUNK_IDX,
    # MCAP常见结构长度
    MAGIC_SIZE, OPCODE_PREFIX, RECORD_PREFIX, CHUNK_HEADER_FIXED,
    # 字节流转整数
    u16, u32, u64,
    # MCAP功能函数
    fetch_mcap_bytes, fetch_mcap_length, decompress_chunk_record,
    fetch_SummaryRecords,
    parse_SchemaRecord_in_SummarySection,
    parse_ChannelRecord_in_SummarySection,
    parse_ChunkIndexRecord_in_SummarySection,
    parse_MessageIdx_from_ChunkIndexRecord,
)

# 每个message需要得到的信息
_PER_MESSAGE_KEYS = (
    "chunk_file_offset", "uncompressed_byte_offset", "record_length",  # message_location
    "topic_name",  # resolved from channel_id via Channel records
    "msg_type",    # ROS2 type string, e.g. "sensor_msgs/msg/JointState"; passed to get_message()
    "msg_def",     # ROS2 msg definition string, used to register custom types in typestore
    "log_time",    # timestamp_ns
)

# chunk 级预读并发度: 每个 chunk 读一次 MessageIdx
_CHUNK_PREFETCH = 4
# message 级预读并发度: 每条 message 读一次 record_length(8 字节), 靠并行掩盖往返延迟
_MESSAGE_PREFETCH = 256


def _read_chunk_msg_idx(
    mcap_path: str,
    chunk: dict[str, Any],
    channel_info: dict[int, tuple[str, str, str]],
) -> list[tuple[int, int, int, int]]:
    """读一个 chunk 的 MessageIdx, 滤掉未选中 channel 后按 log_time 升序返回。

    返回 [(chunk_start_offset, log_time, uncompressed_byte_offset, channel_id), ...]
    """
    msg_idxes = parse_MessageIdx_from_ChunkIndexRecord(mcap_path, chunk)

    # 汇总选中 channel 的 MessageIdx, 再按 log_time 升序排成播放顺序
    msg_idx_all = []   # [(log_time, uncompressed_byte_offset, channel_id), ...]
    for channel_id, msg_idx_ls in msg_idxes.items():
        if channel_id not in channel_info:   # 该 channel 的 topic 未被选中
            continue
        for log_time, ub_offset in msg_idx_ls:
            msg_idx_all.append((log_time, ub_offset, channel_id))
    msg_idx_all.sort()

    return [(chunk["chunk_start_offset"], log_time, ub_offset, channel_id)
            for log_time, ub_offset, channel_id in msg_idx_all]


def _iter_chunks_msg_idx(
    mcap_path: str,
    chunks: list[dict],
    channel_info: dict[int, tuple[str, str, str]],
) -> Iterator[tuple[int, int, int, int]]:
    """按 chunk 时间顺序产出 MessageIdx; 各 chunk 并行预读, 产出顺序仍与 chunks 一致"""
    read_msg_idx = lambda chunk: _read_chunk_msg_idx(mcap_path, chunk, channel_info)
    with ThreadPoolExecutor(max_workers=_CHUNK_PREFETCH) as pool:
        pending = deque()
        for chunk in chunks:
            pending.append(pool.submit(read_msg_idx, chunk))
            if len(pending) >= _CHUNK_PREFETCH:
                yield from pending.popleft().result()
        while pending:
            yield from pending.popleft().result()


def _iter_chunks_messages(
    mcap_path: str,
    chunks: list[dict],
    channel_info: dict[int, tuple[str, str, str]],
) -> Iterator[dict]:
    """按 chunk 先后 + chunk 内 log_time 升序逐条产出 message。

    record_length 须在各消息偏移处读它自身 record header 里的 8 字节, 这些单点读
    以 _MESSAGE_PREFETCH 条为窗口并行预读; 按提交顺序取结果, 故产出顺序不变。
    调用方已确保 chunk 未压缩, 故 uncompressed_byte_offset 可直接换算成文件偏移。
    """
    msg_idx_iter = _iter_chunks_msg_idx(mcap_path, chunks, channel_info)
    with ThreadPoolExecutor(max_workers=_MESSAGE_PREFETCH) as pool:
        pending = deque()   # [(msg_idx, record_length 所在 8 字节的 future), ...]
        while True:
            # 先把预读窗口填满, MessageIdx 取尽时 next() 返回 None
            while len(pending) < _MESSAGE_PREFETCH:
                msg_idx = next(msg_idx_iter, None)
                if msg_idx is None:
                    break
                chunk_start_offset, log_time, ub_offset, channel_id = msg_idx
                # record_length 紧随 message record 的 opcode, 只取这 8 字节
                record_length_offset = (chunk_start_offset + RECORD_PREFIX
                                        + CHUNK_HEADER_FIXED + ub_offset + OPCODE_PREFIX)
                pending.append((msg_idx, pool.submit(fetch_mcap_bytes, mcap_path,
                                                     RECORD_PREFIX - OPCODE_PREFIX,
                                                     record_length_offset)))
            if not pending:
                break
            # 按提交顺序取结果, 故产出顺序与 MessageIdx 顺序一致
            msg_idx, record_length_bytes = pending.popleft()
            chunk_start_offset, log_time, ub_offset, channel_id = msg_idx
            record_length, = u64(record_length_bytes.result(), 0)
            yield dict(zip(_PER_MESSAGE_KEYS,
                           (chunk_start_offset,
                            ub_offset,
                            record_length,
                            *channel_info[channel_id],  # topic_name, msg_type, msg_def
                            log_time)))


def _play_messages_via_metadata(mcap_path: str,
                                extra_topics: tuple[str, ...] = ()) -> Iterator[dict]:
    """
    借 Summary 中的索引产出 message, 避免逐块扫描未选中 topic 的 chunk 数据。
    输出： 一个 message 对应 _PER_MESSAGE_KEYS 中的全部数据,
          只产出 filter_topics() 选中的 topic 与 extra_topics,
          按 chunk 先后 + chunk 内 log_time 升序排列

    MessageIndex 与 ChunkIndex 都不在 Summary 中时抛 ValueError, 交由 _play_messages 接手。

    ！！ 由于获取每个 message 的 record_length 需要频繁的网络访问，导致时间消耗多于顺序读取，放弃
    """
    selected_mcap_topics = filter_topics().keys() | set(extra_topics)
    summary_records_loca = fetch_SummaryRecords(mcap_path)

    # 索引来源为ChunkIndex, 没有则交回调用方
    if bytes([OP_CHUNK_IDX]) not in summary_records_loca:
        raise ValueError("Summary 中无 ChunkIndex")

    # 使用 Chunk Index Record 找到所有的 MessageIdx
    # --- 1. 全部 chunk 信息, chunk_start_offset 已具有时间先后顺序 ---
    chunks = parse_ChunkIndexRecord_in_SummarySection(
        mcap_path, *summary_records_loca[bytes([OP_CHUNK_IDX])])
    if not chunks:
        raise ValueError("Summary 中 Chunk Index Record 为空")
    # 同一文件允许逐 chunk 采用不同压缩方式, 故须全量校验
    if any(chunk["compression"] for chunk in chunks):
        raise ValueError("该 MCAP 中包含压缩 Chunk")

    # --- 2. Schema / Channel: Summary 缺失时从首个 chunk 的 records 区解析 ---
    schema_loca  = summary_records_loca.get(bytes([OP_SCHEMA]))
    channel_loca = summary_records_loca.get(bytes([OP_CHANNEL]))
    if schema_loca is None or channel_loca is None:
        records_loca = (
            chunks[0]["chunk_start_offset"] + RECORD_PREFIX + CHUNK_HEADER_FIXED,
            chunks[0]["compressed_size"]
        )
        schema_loca  = schema_loca  or records_loca
        channel_loca = channel_loca or records_loca
    schemas  = parse_SchemaRecord_in_SummarySection(mcap_path, *schema_loca)
    channels = parse_ChannelRecord_in_SummarySection(mcap_path, *channel_loca)

    # --- 3. 只为选中 topic 建 channel_id -> (topic_name, msg_type, msg_def) ---
    schema_by_id = {schema["schema_id"]: schema for schema in schemas}
    channel_info = {
        channel["channel_id"]: (channel["topic"],
                                schema_by_id.get(channel["schema_id"], {}).get("name", ""),
                                schema_by_id.get(channel["schema_id"], {}).get("msgdef", ""))
        for channel in channels if channel["topic"] in selected_mcap_topics
    }
    return _iter_chunks_messages(mcap_path, chunks, channel_info)


def _play_messages(mcap_path: str,
                   extra_topics: tuple[str, ...] = ()) -> Iterator[dict]:
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
    while pos + RECORD_PREFIX <= file_size:
        # --- 获取 Record Prefix ---
        header = fetch_mcap_bytes(mcap_path, RECORD_PREFIX, pos)
        if len(header) < RECORD_PREFIX:
            break
        (top_record_length,) = u64(header, OPCODE_PREFIX) # 获取当前Record长度，每个Record可能还包括Record

        # --- 只有 Chunk Record 内部才有 Message Record，因此只读取 Chunk Record ---
        if header[0] == OP_CHUNK:
            chunk_file_offset = pos  # chunk 在文件中的绝对偏移，作为定位主键

            # --- 解析 chunk header, 定位 records 起始位置 ---
            chunk_head = fetch_mcap_bytes(
                mcap_path,
                CHUNK_HEADER_FIXED + 4, # max(name_len)=4
                pos + RECORD_PREFIX
            )
            # name_len=0,不压缩; name_len=3,lz4; name_len=4,zstd
            (name_len,) = u32(chunk_head, 28)
            # 读取压缩算法，空字符串,'lz4','zstd'
            compression = chunk_head[32 : 32 + name_len].decode() # '' or 'lz4' or 'zstd'
            # 读取 Chunk Header records_length
            (records_length,) = u64(chunk_head, 32 + name_len)
            # 从第一个record开始
            records_start = pos + RECORD_PREFIX + CHUNK_HEADER_FIXED + name_len
            # 读取整个 Chunk Record
            if compression:
                records_bytes = decompress_chunk_record(
                    mcap_path, records_start, records_length, compression,
                    u64(chunk_head, 16)[0],  # uncompressed_size, zstd 解压上界
                )
            else:
                records_bytes = fetch_mcap_bytes(mcap_path, records_length, records_start)

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

    Topic filtering happens inside _play_messages() / _play_messages_via_metadata():
    they only produce topics resolved by topics_filter.filter_topics()
    (USER_SELECTED_TOPICS mapped through TELEAVATAV2_MCAP_TOPICS_MAPPING), plus the
    extra_topics forwarded from play_messages(); this class does not filter again.
    """

    def __init__(self, mcap_url: str):
        self.mcap_url = mcap_url

    @staticmethod
    def _accumulate_frames_from_keyframe(
        ffmpeg_buffers: dict,
        topic_name: str,
        loc: tuple,
    ) -> list:
        """
        记录下能够解码该帧图像的所有帧数据索引，方便后续操作

        Returns [I] when this frame is an IDR.
        Returns [I P ...] when an IDR (keyframe) is detected.
        Returns [] if no IDR has been seen yet for this message.
        """
        payload = fetch_data_bytes(*loc)
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

        # 暂时不使用元数据读取方式，
        # 使用ChunkIndexRecord会在寻找 record_length 上花费更长时间
        # try:
        #     msg_iter = _play_messages_via_metadata(self.mcap_url, extra_topics)
        # except ValueError:
        #     msg_iter = _play_messages(self.mcap_url, extra_topics)

        # 直接使用顺序播放形式
        msg_iter = _play_messages(self.mcap_url, extra_topics)
        for msg in msg_iter:
            topic = msg["topic_name"]
            loc = (self.mcap_url,
                   msg["chunk_file_offset"],
                   msg["uncompressed_byte_offset"],
                   msg["record_length"])
            if mcap_video_topics_gop.get(topic,1) > 1:  # GOP>1: 帧间编码, 需回溯到关键帧
                locations = self._accumulate_frames_from_keyframe(
                    ffmpeg_buffers, topic, loc)
                if not locations:  # 无关键帧可依托, 跳过该帧
                    continue
            else:
                locations = [loc]
            # 将定位整数与时间戳统一转为字符串后对外产出
            locations_str = [(loc[0], str(loc[1]), str(loc[2]), str(loc[3]))
                             for loc in locations]
            yield topic, msg["msg_type"], msg["msg_def"], str(msg["log_time"]), locations_str
