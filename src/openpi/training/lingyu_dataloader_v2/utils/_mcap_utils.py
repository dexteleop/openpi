from __future__ import annotations
from functools import lru_cache
import logging
import random
import struct
import threading
import time
from typing import Any

from botocore.awsrequest import AWSHTTPConnection, AWSHTTPSConnection
from botocore.config import Config
from botocore.exceptions import ClientError, IncompleteReadError, ReadTimeoutError, ResponseStreamingError

from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
    BUCKET_NAME, KEY_PREFIX, make_client
)

logger = logging.getLogger(__name__)


##########
# MCAP Record types
##########
# Storage information
OP_HEADER         = 0x01
OP_FOOTER         = 0x02
OP_SUMMARY_OFFSET = 0x0E
OP_MESSAGE        = 0x05
OP_CHUNK          = 0x06
# Storage index
OP_SCHEMA         = 0x03
OP_CHANNEL        = 0x04
OP_MESSAGE_IDX    = 0x07
OP_CHUNK_IDX      = 0x08
# Storage statistics
OP_STATISTICS     = 0x0B
OP_METADATA       = 0x0C
OP_METADATA_IDX   = 0x0D
OP_ATTACHMENT     = 0x09
OP_ATTACHMENT_IDX = 0x0A
OP_DATAEND        = 0x0F


##########
# Common MCAP structure lengths
##########
MAGIC_SIZE = 8  # b"\x89MCAP0\r\n"
OPCODE_PREFIX = 1
RECORD_PREFIX = OPCODE_PREFIX + 8  # opcode(1) + length(8)

# Footer Record: opcode(1) + len(8) + Data(20)
# Footer Data(20): summary_start(8) + summary_offset_start(8) + summary_crc(4)
FOOTER_SIZE = RECORD_PREFIX + 8 + 8 + 4

# Set of opcodes that may appear in Summary Offset Records
SUMMARY_OFFSET_OPCODES = {
    OP_SCHEMA, OP_CHANNEL, OP_CHUNK_IDX,
}
# A single Summary Offset Record: group_opcode(1)
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
# The chunk header length is not fixed: no compression, name_len=0; lz4, name_len=3; zstd, name_len=4

# Message Record: RECORD_PREFIX + Message Header
# Message Header: channel_id(2)
#                 + sequence(4)
#                 + log_time(8)
#                 + publish_time(8)
MESSAGE_HEADER = 2 + 4 + 8 + 8


##########
# Byte stream to integer
##########
u16 = struct.Struct("<H").unpack_from # 2 bytes -> int
u32 = struct.Struct("<I").unpack_from # 4 bytes -> int
u64 = struct.Struct("<Q").unpack_from # 8 bytes -> int


##########
# Object storage read functions
##########
# Per-process connection pool limit: concurrent requests beyond this still create new connections, but they are closed after use and cannot be reused
S3_MAX_POOL_CONNECTIONS = 200
# Maximum attempts for one request when the response body read times out / the stream breaks, or throttling persists after botocore's own retries are exhausted;
# in practice the OSS gateway can stall on individual object ranges for over ten minutes; together with S3_READ_TIMEOUT and backoff of up to 30s, this tolerates about 45 minutes of continuous failures
S3_READ_ATTEMPTS = 60
# Socket timeout (seconds) per read; botocore defaults to 60s, stuck connections rarely recover on their own, shortening it switches to a new connection and retries faster
S3_READ_TIMEOUT = 20
# In practice the gateway keeps stalling for over ten minutes on large range reads of cold data, while small reads (<= 16KiB) at the same position return immediately;
# so after S3_SPLIT_AFTER_ATTEMPTS timeouts on the same range, the read is split into S3_SPLIT_BYTES pieces read sequentially and concatenated
S3_SPLIT_AFTER_ATTEMPTS = 2
S3_SPLIT_BYTES = 16 * 1024
# HTTP status codes treated as throttling / temporary server unavailability, worth retrying after backoff
S3_THROTTLE_STATUS = {429, 500, 503}

# Time (ms) of the most recent TCP connection establishment in the current thread; None means this request reused an idle pooled connection
_tcp_connect_local = threading.local()
# Both http / https connections inherit the same urllib3 _new_conn (DNS resolution + TCP three-way handshake)
_new_conn_original = AWSHTTPConnection._new_conn
assert AWSHTTPSConnection._new_conn is _new_conn_original, "botocore 连接类结构已变化"


def _new_conn_timed(self):
    """Wrap urllib3 connection establishment and record the real connect time in the current thread; get_object connects synchronously in the calling thread, so threads do not interfere with each other"""
    connect_start = time.perf_counter()
    sock = _new_conn_original(self)
    _tcp_connect_local.connect_ms = (time.perf_counter() - connect_start) * 1000
    return sock


# Only replace botocore's own connection subclass, without affecting other urllib3 users in the process
AWSHTTPConnection._new_conn = _new_conn_timed
AWSHTTPSConnection._new_conn = _new_conn_timed


@lru_cache(maxsize=1)
def s3_client():
    """A boto3 client carries its own connection pool internally; one per process is enough"""
    return make_client(Config(max_pool_connections=S3_MAX_POOL_CONNECTIONS, read_timeout=S3_READ_TIMEOUT,
                              retries={"mode": "legacy", "max_attempts": 10}))  # OSS throttling returns 429 TooManyRequests; only legacy mode retries it


def pool_stats() -> list[dict[str, Any]]:
    """Read the live status of each connection pool in s3_client() (one per host:port)

    idle: number of idle connections in the pool; in_use: pool slots taken (capped at maxsize, overflow temporary connections not counted)
    created / requests: cumulative new connections / cumulative requests
    Relies on botocore / urllib3 private attributes; an assert fails directly if the structure changes after an upgrade
    """
    http_session = getattr(getattr(s3_client(), "_endpoint", None), "http_session", None)
    manager = getattr(http_session, "_manager", None)
    assert manager is not None and hasattr(manager, "pools"), "botocore 内部结构已变化, 无法读取连接池"

    stats = []
    for pool_key in manager.pools.keys():
        conn_pool = manager.pools.get(pool_key)
        if conn_pool is None or conn_pool.pool is None:  # The pool has been evicted or closed
            continue
        # Hold the queue lock to take a snapshot, avoiding interleaving with other threads' get/return of connections; do not call qsize() inside the lock (the same non-reentrant lock)
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
    """Log at DEBUG the connect / first-byte / transfer time, throughput and pool status of a single range read"""
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


def _bucket_key(mcap_url: str) -> str:
    """Index url -> real key inside the bucket: the same batch of data has different prefixes in the two buckets (the Wulanchabu bucket under shihaoran/,
    the Beijing bucket at the top level), while the index url only records the key in the bucket used at collection time. When KEY_PREFIX is non-empty and the url lacks that
    prefix, it is prepended; legacy index urls that already contain the prefix (e.g. shihaoran/raw/...) are returned as-is."""
    return mcap_url if not KEY_PREFIX or mcap_url.startswith(f"{KEY_PREFIX}/") else f"{KEY_PREFIX}/{mcap_url}"


def fetch_mcap_bytes(mcap_url: str, length: int, offset: int) -> bytes:
    """Read a byte range from the bucket; when the logger has DEBUG enabled, record the timing and pool status of this read

    ttfb spans from sending the request to receiving the response headers (including possible connection setup, signing and server-side processing); transfer is the time spent reading the response body.
    At end of file S3 truncates the range, so the returned byte count may be smaller than length (_play_messages relies on this behavior)
    """
    # length<=0 builds an invalid Range header; S3 may return the whole object
    assert length > 0 and offset >= 0, f"非法读取范围: length={length}, offset={offset}"
    byte_range = f"bytes={offset}-{offset + length - 1}"
    for attempt_idx in range(S3_READ_ATTEMPTS):
        _tcp_connect_local.connect_ms = None  # Clear this thread's connection-establishment record from the previous request
        request_start = time.perf_counter()
        try:
            response = s3_client().get_object(Bucket=BUCKET_NAME, Key=_bucket_key(mcap_url), Range=byte_range)
            header_received = time.perf_counter()
            data_bytes = response["Body"].read()
            break
        except (ReadTimeoutError, ResponseStreamingError, IncompleteReadError, ClientError) as error:
            # For ClientError only retry throttling-type errors (e.g. TooManyRequests); other errors such as NoSuchKey are raised as usual
            is_throttled = (not isinstance(error, ClientError)
                            or error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in S3_THROTTLE_STATUS)
            if not is_throttled or attempt_idx == S3_READ_ATTEMPTS - 1:
                raise
            if (not isinstance(error, ClientError) and length > S3_SPLIT_BYTES
                    and attempt_idx + 1 >= S3_SPLIT_AFTER_ATTEMPTS):
                logger.warning(f"读取 {mcap_url} {byte_range} 已连续失败 {attempt_idx + 1} 次, "
                               f"改为按 {S3_SPLIT_BYTES} 字节分片读取")
                return _fetch_mcap_bytes_in_pieces(mcap_url, length, offset)
            # Exponential backoff + random jitter, avoiding a request storm when many workers retry at the same time
            backoff_seconds = min(2 ** attempt_idx, 30) * random.uniform(0.5, 1.5)
            logger.warning(f"读取 {mcap_url} {byte_range} 失败({type(error).__name__}), "
                           f"{backoff_seconds:.1f}s 后第 {attempt_idx + 2} 次尝试")
            time.sleep(backoff_seconds)
    read_end = time.perf_counter()
    if logger.isEnabledFor(logging.DEBUG):
        _log_fetch_timing(mcap_url, offset, len(data_bytes),
                          request_start, header_received, read_end)
    return data_bytes


def _fetch_mcap_bytes_in_pieces(mcap_url: str, length: int, offset: int) -> bytes:
    """Split one large range read into several small reads of at most S3_SPLIT_BYTES, read them in order and concatenate; stop at end of file"""
    pieces = []
    for piece_offset in range(offset, offset + length, S3_SPLIT_BYTES):
        piece_length = min(S3_SPLIT_BYTES, offset + length - piece_offset)
        piece = fetch_mcap_bytes(mcap_url, piece_length, piece_offset)  # Smaller than S3_SPLIT_BYTES, so it will not be split again
        pieces.append(piece)
        if len(piece) < piece_length:
            break
    return b"".join(pieces)


##########
# MCAP utility functions
##########
# Read the MCAP byte length
def fetch_mcap_length(mcap_url: str) -> int:
    """Byte size of the object"""
    return s3_client().head_object(
        Bucket=BUCKET_NAME,
        Key=mcap_url
    )["ContentLength"]


# Decompress compressed chunks
def decompress_chunk_record(
    mcap_url: str,
    records_start: int,
    records_length: int,
    compression: str,
    uncompressed_size: int,
) -> bytes:
    """Decompress the whole records section of a chunk; only the most recent chunk is cached, for reuse by consecutive reads of the same chunk (e.g. ffmpeg GOP)."""
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


# Locate the position offsets of the Summary Records
def fetch_SummaryRecords(mcap_url: str) -> dict[bytes, tuple[int, int]]:
    """ Locate the position offsets of multiple Records in the Summary Section
    Basic structure at the end of an MCAP:
    [Schema Record] (optional, repeated)  <-- 3. Summary and index section (Summary Section)
    [Channel Record] (optional, repeated)
    [Chunk Index Record]  (file physical offsets of all chunks)
    [Message Index Record]  (timestamps and offsets of messages in each chunk; not present for now)
    ...
    [Summary Offset Record] (points to the start of each summary Record type above)
    [Footer Record]
    [Magic Record]
    """
    file_size = fetch_mcap_length(mcap_url)

    # Find the Footer at the end of the MCAP
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

    # Use the Footer to locate the Summary Offset Records
    summary_offset_length = footer_offset_start - summary_offset_start
    if summary_offset_length <= 0:
        return {}

    summary_offset_bytes = fetch_mcap_bytes(mcap_url,
                                            summary_offset_length,
                                            summary_offset_start)

    # Use the Summary Offset Records to locate the Summary Records
    summary_records_loca: dict[bytes, tuple[int, int]] = {}
    pos = 0
    while pos + RECORD_PREFIX <= summary_offset_length:
        op = summary_offset_bytes[pos]
        if op != OP_SUMMARY_OFFSET:
            # Stop parsing when a non-Summary-Offset Record is encountered
            break
        payload_len, = u64(summary_offset_bytes, pos + OPCODE_PREFIX)

        # Each payload is fixed at 17 bytes:
        # group_opcode(1) + group_start(8) + group_length(8)
        payload_pos = pos + RECORD_PREFIX
        group_opcode = summary_offset_bytes[payload_pos: payload_pos + OPCODE_PREFIX]

        if group_opcode[0] in SUMMARY_OFFSET_OPCODES:
            group_start,  = u64(summary_offset_bytes, payload_pos + 1)
            group_length, = u64(summary_offset_bytes, payload_pos + 9)
            summary_records_loca[group_opcode] = (group_start, group_length)

        pos += RECORD_PREFIX + payload_len

    return summary_records_loca


# Parse Schemas from the Summary
def parse_SchemaRecord_in_SummarySection(mcap_url: str, record_start: int, record_length: int) -> list[dict]:
    """Parse all Schema Records in the Summary
    Returned fields: schema_id, name, encoding, msgdef
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
            # name / encoding / data are all prefixed fields: uint32 length + content
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


# Parse Channels from the Summary
def parse_ChannelRecord_in_SummarySection(mcap_url: str, record_start: int, record_length: int) -> list[dict]:
    """Parse all Channel Records in the Summary
    Returned fields: channel_id, schema_id, topic, encoding
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
            # topic / message_encoding are both prefixed strings: uint32 length + UTF-8 bytes
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


# Parse Chunks from the Summary
def parse_ChunkIndexRecord_in_SummarySection(mcap_url: str, record_start: int, record_length: int) -> list[dict]:
    """Parse all Chunk Index Records in the Summary, which contain 9 fields:
    msg_start_time, msg_end_time, chunk_start_offset, chunk_length,
    msg_idx_offsets, msg_idx_length, compression, compressed_size, uncompressed_size

    Returned fields:
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
            pos = 8 + 8  # Skip msg_start_time / msg_end_time
            chunk_start_offset, = u64(payload, pos); pos += 8
            chunk_length,       = u64(payload, pos); pos += 8
            # message_index_offsets: map<uint16 channel_id, uint64 file_offset>
            # The u32 field is the total byte length of the map, not the entry count; each entry is 10 bytes (2+8)
            map_byte_len, = u32(payload, pos); pos += 4
            map_end = pos + map_byte_len
            msg_idx_offsets: dict[int, int] = {}
            while pos + 10 <= map_end:
                ch_id,  = u16(payload, pos); pos += 2
                offset, = u64(payload, pos); pos += 8
                msg_idx_offsets[ch_id] = offset
            pos = map_end
            msg_idx_length, = u64(payload, pos); pos += 8
            # compression is a prefixed string: uint32 length + UTF-8 bytes
            compression_len, = u32(payload, pos); pos += 4
            compression = payload[pos:pos + compression_len].decode(
                "utf-8", errors="replace")
            pos += compression_len
            # compressed_size is the byte size of the records section in the file; uncompressed_size is used as the decompression upper bound
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


# Use the Chunk Index Record to parse the Message Index Records inside one chunk
def parse_MessageIdx_from_ChunkIndexRecord(
    mcap_url: str,
    chunk: dict[str, Any],
) -> dict[int, list[tuple[int, int]]]:
    """Read the whole group of Message Index Records of one chunk

    Args:
        chunk: one item of the list returned by parse_ChunkIndexRecord_in_SummarySection()

    Returns:
        channel_id -> [(log_time, uncompressed_byte_offset), ...], kept in the original write order
    """
    # The MessageIndex records of one chunk are laid out consecutively; the minimum offset in msg_idx_offsets is the start of the whole group
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
