from __future__ import annotations
from functools import lru_cache
import logging
import struct
import threading
import time
from typing import Any

from botocore.awsrequest import AWSHTTPConnection, AWSHTTPSConnection
from botocore.config import Config

from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
    BUCKET_NAME, make_client
)

logger = logging.getLogger(__name__)


##########
# MCAP Record 类型
##########
# 存储信息
OP_HEADER         = 0x01
OP_FOOTER         = 0x02
OP_SUMMARY_OFFSET = 0x0E
OP_MESSAGE        = 0x05
OP_CHUNK          = 0x06
# 存储索引
OP_SCHEMA         = 0x03
OP_CHANNEL        = 0x04
OP_MESSAGE_IDX    = 0x07
OP_CHUNK_IDX      = 0x08
# 存储统计数据
OP_STATISTICS     = 0x0B
OP_METADATA       = 0x0C
OP_METADATA_IDX   = 0x0D
OP_ATTACHMENT     = 0x09
OP_ATTACHMENT_IDX = 0x0A
OP_DATAEND        = 0x0F


##########
# MCAP常见结构长度
##########
MAGIC_SIZE = 8  # b"\x89MCAP0\r\n"
OPCODE_PREFIX = 1
RECORD_PREFIX = OPCODE_PREFIX + 8  # opcode(1) + length(8)

# Footer Record: opcode(1) + len(8) + Data(20)
# Footer Data(20): summary_start(8) + summary_offset_start(8) + summary_crc(4)
FOOTER_SIZE = RECORD_PREFIX + 8 + 8 + 4

# Summary Offset Record 中可能出现的 opcode 集合
SUMMARY_OFFSET_OPCODES = {
    OP_SCHEMA, OP_CHANNEL, OP_CHUNK_IDX,
}
# 单条 Summary Offset Record： group_opcode(1)
#                             + group_start(8)
#                             + group_length(8)
SUMMARY_OFFSET_REC_LEN = 17

# Chunk Record: RECORD_PREFIX + Chunk Header
# Chunk Header: msessage_start_time(8)
#               + msessage_end_time(8)
#               + uncompressed_size(8)
#               + uncompressed_crc(4)
#               + compression(4+name_len)
#               + records_length(8)
CHUNK_HEADER_FIXED = 8 + 8 + 8 + 4 + 4 + 8
# chunk header 长度并不是固定的：不压缩,name_len=0; lz4,name_len=3; zstd,name_len=4

# Message Record: RECORD_PREFIX + Message Header
# Message Header: channel_id(2)
#                 + sequence(4)
#                 + log_time(8)
#                 + publish_time(8)
MESSAGE_HEADER = 2 + 4 + 8 + 8


##########
# 字节流转整数
##########
u16 = struct.Struct("<H").unpack_from # 2 bytes -> int
u32 = struct.Struct("<I").unpack_from # 4 bytes -> int
u64 = struct.Struct("<Q").unpack_from # 8 bytes -> int


##########
# 对象存储读取函数
##########
# 单进程连接池上限: 超过该数的并发请求仍会新建连接, 但用完即关闭, 无法复用
S3_MAX_POOL_CONNECTIONS = 200

# 当前线程最近一次新建 TCP 连接的耗时(ms); None 表示本次请求复用了池内空闲连接
_tcp_connect_local = threading.local()
# http / https 连接都继承 urllib3 的同一个 _new_conn(DNS 解析 + TCP 三次握手)
_new_conn_original = AWSHTTPConnection._new_conn
assert AWSHTTPSConnection._new_conn is _new_conn_original, "botocore 连接类结构已变化"


def _new_conn_timed(self):
    """包装 urllib3 建连, 把真实建连耗时写入当前线程; get_object 在调用线程内同步建连, 多线程互不干扰"""
    connect_start = time.perf_counter()
    sock = _new_conn_original(self)
    _tcp_connect_local.connect_ms = (time.perf_counter() - connect_start) * 1000
    return sock


# 只替换 botocore 自己的连接子类, 不影响进程内其他 urllib3 用户
AWSHTTPConnection._new_conn = _new_conn_timed
AWSHTTPSConnection._new_conn = _new_conn_timed


@lru_cache(maxsize=1)
def s3_client():
    """boto3 client 内部带连接池, 全进程复用一个即可"""
    return make_client(Config(max_pool_connections=S3_MAX_POOL_CONNECTIONS))


def pool_stats() -> list[dict[str, Any]]:
    """读取 s3_client() 各连接池(每个 host:port 一个)的实时状态

    idle: 池内空闲连接数; in_use: 被取走的池位数(上限 maxsize, 溢出的临时连接不计入)
    created / requests: 累计新建连接数 / 累计请求数
    依赖 botocore / urllib3 私有属性, 升级后结构变化时由 assert 直接报错
    """
    http_session = getattr(getattr(s3_client(), "_endpoint", None), "http_session", None)
    manager = getattr(http_session, "_manager", None)
    assert manager is not None and hasattr(manager, "pools"), "botocore 内部结构已变化, 无法读取连接池"

    stats = []
    for pool_key in manager.pools.keys():
        conn_pool = manager.pools.get(pool_key)
        if conn_pool is None or conn_pool.pool is None:  # 池已被淘汰或已关闭
            continue
        # 持队列锁取快照, 避免与其他线程的取/还连接交错; 锁内不能再调 qsize()(同一把非重入锁)
        with conn_pool.pool.mutex:
            idle_count = sum(conn is not None for conn in conn_pool.pool.queue)
            in_use_count = conn_pool.pool.maxsize - len(conn_pool.pool.queue)
        stats.append({
            "host":     f"{conn_pool.host}:{conn_pool.port}",
            "maxsize":  conn_pool.pool.maxsize,
            "idle":     idle_count,
            "in_use":   in_use_count,
            "created":  conn_pool.num_connections,
            "requests": conn_pool.num_requests,
        })
    return stats


def _log_fetch_timing(mcap_url: str, offset: int, data_size: int,
                      request_start: float, header_received: float, read_end: float) -> None:
    """DEBUG 输出单次 range 读取的建连 / 首字节 / 传输耗时、吞吐与连接池状态"""
    connect_ms = getattr(_tcp_connect_local, "connect_ms", None)
    transfer_sec = read_end - header_received
    throughput_mbps = data_size / (1024 * 1024) / transfer_sec if transfer_sec > 0 else 0.0
    logger.debug(
        "fetch key=%s offset=%d bytes=%d tcp=%s ttfb=%.3fms transfer=%.3fms "
        "total=%.3fms throughput=%.2fMB/s pool=%s",
        mcap_url, offset, data_size,
        f"{connect_ms:.3f}ms(新建)" if connect_ms is not None else "复用",
        (header_received - request_start) * 1000, transfer_sec * 1000,
        (read_end - request_start) * 1000, throughput_mbps, pool_stats())


def fetch_mcap_bytes(mcap_url: str, length: int, offset: int) -> bytes:
    """按 byte range 读取桶内字节流; logger 开启 DEBUG 时记录本次读取的计时与连接池状态

    ttfb 从发请求到响应头返回(含可能的建连、签名与服务端处理), transfer 为读取响应体耗时。
    读到文件尾时 S3 会截断 range, 返回字节数可能小于 length(_play_messages 依赖此行为)
    """
    # length<=0 会拼出非法 Range, S3 可能直接返回整个对象
    assert length > 0 and offset >= 0, f"非法读取范围: length={length}, offset={offset}"
    byte_range = f"bytes={offset}-{offset + length - 1}"
    _tcp_connect_local.connect_ms = None  # 清空本线程上一次请求的建连记录
    request_start = time.perf_counter()
    response = s3_client().get_object(Bucket=BUCKET_NAME, Key=mcap_url, Range=byte_range)
    header_received = time.perf_counter()
    data_bytes = response["Body"].read()
    read_end = time.perf_counter()
    if logger.isEnabledFor(logging.DEBUG):
        _log_fetch_timing(mcap_url, offset, len(data_bytes),
                          request_start, header_received, read_end)
    return data_bytes


##########
# MCAP功能函数
##########
# 读取 MCAP 字节长度
def fetch_mcap_length(mcap_url: str) -> int:
    """对象字节数"""
    return s3_client().head_object(
        Bucket=BUCKET_NAME,
        Key=mcap_url
    )["ContentLength"]


# 对压缩chunk进行解压缩
def decompress_chunk_record(
    mcap_url: str,
    records_start: int,
    records_length: int,
    compression: str,
    uncompressed_size: int,
) -> bytes:
    """整块解压 chunk 的 records 区; 只缓存最近一块, 供同 chunk 的连续读取(如 ffmpeg GOP)复用。"""
    import lz4.frame
    import zstandard

    records_bytes = fetch_mcap_bytes(
        mcap_url, records_length, records_start)
    if compression == "zstd":
        return zstandard.ZstdDecompressor().decompress(
            records_bytes, max_output_size=uncompressed_size)
    if compression == "lz4":
        return lz4.frame.decompress(records_bytes)
    raise ValueError(f"不支持的 chunk 压缩方式: {compression!r}")


# 找到 Summary Records 的位置偏移
def fetch_SummaryRecords(mcap_url: str) -> dict[bytes, tuple[int, int]]:
    """ 定位 Summary Setion 中多个 Record 位置偏移
    MCAP尾部的基本结构：
    [Schema Record] (可选重复)  <-- 3. 摘要与索引区 (Summary Section)
    [Channel Record] (可选重复)
    [Chunk Index Record]  (所有 Chunk 的文件物理偏移量)
    [Message Index Record]  (每个 Chunk 中 Message 的时间戳与偏移量，暂时不存在)
    ...
    [Summary Offset Record] (指向上述各个摘要 Record 类型的起始位置)
    [Footer Record]
    [Magic Record]
    """
    file_size = fetch_mcap_length(mcap_url)

    # 从 MCAP尾部 找到 Footer
    footer_offset_start = file_size - MAGIC_SIZE - FOOTER_SIZE
    footer_bytes = fetch_mcap_bytes(
        mcap_url, FOOTER_SIZE, footer_offset_start
    )

    if footer_bytes[0] != OP_FOOTER:
        raise ValueError(f"Invalid MCAP file: "
                         f"expected Footer Record opcode 0x02, "
                         f"got 0x{footer_bytes[0]:02x}. "
                         f"The file might be incomplete or streaming.")

    summary_start, = u64(footer_bytes, RECORD_PREFIX)
    if summary_start == 0:
        raise ValueError("MCAP don't include Summary"
                         "（summary_start == 0）")
    summary_offset_start, = u64(footer_bytes, RECORD_PREFIX + 8)

    # 利用 Footer 找到 Summary Offset Records
    summary_offset_length = footer_offset_start - summary_offset_start
    if summary_offset_length <= 0:
        return {}

    summary_offset_bytes = fetch_mcap_bytes(mcap_url,
                                            summary_offset_length,
                                            summary_offset_start)

    # 利用 Summary Offset Records 来找到 Summary Records 的位置
    summary_records_loca: dict[bytes, tuple[int, int]] = {}
    pos = 0
    while pos + RECORD_PREFIX <= summary_offset_length:
        op = summary_offset_bytes[pos]
        if op != OP_SUMMARY_OFFSET:
            # 遇到非 Summary Offset Record，停止解析
            break
        payload_len, = u64(summary_offset_bytes, pos + OPCODE_PREFIX)

        # 每条 payload 固定 17 字节：
        # group_opcode(1) + group_start(8) + group_length(8)
        payload_pos = pos + RECORD_PREFIX
        group_opcode = summary_offset_bytes[payload_pos: payload_pos + OPCODE_PREFIX]

        if group_opcode[0] in SUMMARY_OFFSET_OPCODES:
            group_start,  = u64(summary_offset_bytes, payload_pos + 1)
            group_length, = u64(summary_offset_bytes, payload_pos + 9)
            summary_records_loca[group_opcode] = (group_start, group_length)

        pos += RECORD_PREFIX + payload_len

    return summary_records_loca


# 解析 Summary 中的 Schema
def parse_SchemaRecord_in_SummarySection(mcap_url: str, record_start: int, record_length: int) -> list[dict]:
    """解析 Summary 中所有 Schema Record
    返回字段: schema_id, name, encoding, msgdef
    """
    section = fetch_mcap_bytes(mcap_url, record_length, record_start)
    result = []
    pos = 0
    while pos + RECORD_PREFIX <= len(section):
        payload_len, = u64(section, pos + OPCODE_PREFIX)
        payload_end = pos + RECORD_PREFIX + payload_len
        if section[pos] == OP_SCHEMA:
            payload = section[pos + RECORD_PREFIX:payload_end]
            schema_id, = u16(payload, 0)
            # name / encoding / data 均为 prefixed 字段: uint32 长度 + 内容
            name_len, = u32(payload, 2)
            name_end = 2 + 4 + name_len
            encoding_len, = u32(payload, name_end)
            encoding_end = name_end + 4 + encoding_len
            data_len, = u32(payload, encoding_end)
            data_end = encoding_end + 4 + data_len
            result.append({
                "schema_id": schema_id,
                "name":      payload[2 + 4:name_end].decode("utf-8", errors="replace"),
                "encoding":  payload[name_end + 4:encoding_end].decode("utf-8", errors="replace"),
                "msgdef":    payload[encoding_end + 4:data_end].decode("utf-8", errors="replace"),
            })
        pos = payload_end
    return result


# 解析 Summary 中的 Channel
def parse_ChannelRecord_in_SummarySection(mcap_url: str, record_start: int, record_length: int) -> list[dict]:
    """解析 Summary 中所有 Channel Record
    返回字段: channel_id, schema_id, topic, encoding
    """
    section = fetch_mcap_bytes(mcap_url, record_length, record_start)
    result = []
    pos = 0
    while pos + RECORD_PREFIX <= len(section):
        payload_len, = u64(section, pos + OPCODE_PREFIX)
        payload_end  = pos + RECORD_PREFIX + payload_len
        if section[pos] == OP_CHANNEL:
            payload = section[pos + RECORD_PREFIX:payload_end]
            channel_id, = u16(payload, 0)
            schema_id,  = u16(payload, 2)
            # topic / message_encoding 均为 prefixed 字符串: uint32 长度 + UTF-8 字节
            topic_len, = u32(payload, 4)
            topic_end = 4 + 4 + topic_len
            encoding_len, = u32(payload, topic_end)
            encoding_end = topic_end + 4 + encoding_len
            result.append({
                "channel_id": channel_id,
                "schema_id":  schema_id,
                "topic":      payload[4 + 4:topic_end].decode("utf-8", errors="replace"),
                "encoding":   payload[topic_end + 4:encoding_end].decode("utf-8", errors="replace"),
            })
        pos = payload_end
    return result


# 解析 Summary 中的 Chunk
def parse_ChunkIndexRecord_in_SummarySection(mcap_url: str, record_start: int, record_length: int) -> list[dict]:
    """解析 Summary 中所有 Chunk Index Record，其中包含的9个字段：
    msg_start_time, msg_end_time, chunk_start_offset, chunk_length,
    msg_idx_offsets, msg_idx_length, compression, compressed_size, uncompressed_size

    返回字段：
    chunk_start_offset, chunk_length,
    msg_idx_offsets, msg_idx_length,
    compression, compressed_size, uncompressed_size
    """
    section = fetch_mcap_bytes(mcap_url, record_length, record_start)
    result = []
    record_pos = 0
    while record_pos + RECORD_PREFIX <= len(section):
        payload_len, = u64(section, record_pos + OPCODE_PREFIX)
        payload_end = record_pos + RECORD_PREFIX + payload_len
        if section[record_pos] == OP_CHUNK_IDX:
            payload = section[record_pos + RECORD_PREFIX:payload_end]
            pos = 8 + 8  # 跳过 msg_start_time / msg_end_time
            chunk_start_offset, = u64(payload, pos); pos += 8
            chunk_length,       = u64(payload, pos); pos += 8
            # message_index_offsets: map<uint16 channel_id, uint64 file_offset>
            # u32 字段是 map 的字节总长，不是条目数；每条 10 字节（2+8）
            map_byte_len, = u32(payload, pos); pos += 4
            map_end = pos + map_byte_len
            msg_idx_offsets: dict[int, int] = {}
            while pos + 10 <= map_end:
                ch_id,  = u16(payload, pos); pos += 2
                offset, = u64(payload, pos); pos += 8
                msg_idx_offsets[ch_id] = offset
            pos = map_end
            msg_idx_length, = u64(payload, pos); pos += 8
            # compression 为 prefixed 字符串: uint32 长度 + UTF-8 字节
            compression_len, = u32(payload, pos); pos += 4
            compression = payload[pos:pos + compression_len].decode(
                "utf-8", errors="replace")
            pos += compression_len
            # compressed_size 即 records 区在文件中的字节数; uncompressed_size 供解压上界使用
            compressed_size,   = u64(payload, pos); pos += 8
            uncompressed_size, = u64(payload, pos)
            result.append({
                "chunk_start_offset": chunk_start_offset,
                "chunk_length":       chunk_length,
                "msg_idx_offsets":    msg_idx_offsets,
                "msg_idx_length":     msg_idx_length,
                "compression":        compression,
                "compressed_size":    compressed_size,
                "uncompressed_size":  uncompressed_size,
            })
        record_pos = payload_end
    return result


# 利用 Chunk Index Record 解析某个 Chunk 内的 Message Index Record
def parse_MessageIdx_from_ChunkIndexRecord(
    mcap_url: str,
    chunk: dict[str, Any],
) -> dict[int, list[tuple[int, int]]]:
    """读取一个 Chunk 的整组 Message Index Record

    参数:
        chunk: parse_ChunkIndexRecord_in_SummarySection() 返回列表中的一项

    返回:
        channel_id -> [(log_time, uncompressed_byte_offset), ...]，按写入顺序保持原样
    """
    # 同一 chunk 的 MessageIndex 连续排列, msg_idx_offsets 中的最小偏移即整组起点
    msg_idx_bytes = fetch_mcap_bytes(mcap_url,
                                     chunk["msg_idx_length"],
                                     min(chunk["msg_idx_offsets"].values()))

    msg_idxes: dict[int, list[tuple[int, int]]] = {}
    record_pos = 0
    while record_pos + RECORD_PREFIX <= chunk["msg_idx_length"]:
        if msg_idx_bytes[record_pos] != OP_MESSAGE_IDX:
            break
        payload_len, = u64(msg_idx_bytes, record_pos + OPCODE_PREFIX)
        payload_pos = record_pos + RECORD_PREFIX
        # MessageIndex payload: channel_id(2) + records_byte_len(4)
        #                     + (log_time(8) + uncompressed_byte_offset(8)) * n
        channel_id, = u16(msg_idx_bytes, payload_pos)
        records_byte_len, = u32(msg_idx_bytes, payload_pos + 2)
        records = msg_idxes.setdefault(channel_id, [])
        for entry_idx in range(records_byte_len // 16):
            entry_pos = payload_pos + 2 + 4 + entry_idx * 16
            log_time,  = u64(msg_idx_bytes, entry_pos)
            ub_offset, = u64(msg_idx_bytes, entry_pos + 8)
            records.append((log_time, ub_offset))
        record_pos = payload_pos + payload_len

    return msg_idxes
