"""
Locate one message using ("mcap_url", "chunk_file_offset", "uncompressed_byte_offset", "record_length").
"""
from __future__ import annotations
import threading

from rosbags.typesys import (
    Stores, get_typestore, get_types_from_msg,
)

from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    # MCAP Record types
    OP_MESSAGE,
    # Common MCAP structure lengths
    OPCODE_PREFIX, RECORD_PREFIX, MESSAGE_HEADER, CHUNK_HEADER_FIXED,
    # Byte stream to integer
    u32, u64,
    # MCAP utility functions
    fetch_mcap_bytes, decompress_chunk_record
)

# The most recent chunk read into memory as a whole: (mcap_url, chunk_file_offset, records_bytes).
# During sequential playback the player has just finished reading a chunk and immediately fetches its messages one by one (keyframe checks, key parsing);
# on a hit it slices directly, saving two S3 round trips per message; only one chunk is stored, so memory stays constant
_last_chunk_records: tuple[str, int, bytes] | None = None


def remember_chunk_records(mcap_url: str, chunk_file_offset: int, records_bytes: bytes) -> None:
    """Keep the chunk just read whole so fetch_data_bytes() can slice it instead of hitting S3."""
    global _last_chunk_records
    # The whole tuple is assigned at once: readers see either the old chunk or the new one, never a half-merged state
    _last_chunk_records = (mcap_url, chunk_file_offset, records_bytes)


def fetch_data_bytes(
    mcap_url: str,
    chunk_file_offset: int,
    uncompressed_byte_offset: int,
    msg_record_length: int,
) -> bytes:
    """Locate and read a single message in O(1) by (chunk_file_offset, uncompressed_byte_offset).
    Uncompressed chunks need no scanning: two pread calls finish locating and reading;
    compressed chunks (zstd/lz4) are decompressed whole first, then sliced at the same offset within the decompressed stream.
    chunk_file_offset is the chunk record's absolute byte position in the file.

    Returns:
        cdr_bytes
    """
    last_chunk = _last_chunk_records
    if last_chunk is not None and last_chunk[:2] == (mcap_url, chunk_file_offset):
        # Hit the chunk just played: records_bytes is the same byte stream as read by the two paths below, so slice directly
        msg_record = last_chunk[2][uncompressed_byte_offset:
                                   uncompressed_byte_offset + RECORD_PREFIX + msg_record_length]
        return _message_cdr(msg_record, chunk_file_offset, uncompressed_byte_offset)

    # --- Parse the chunk header to locate the start of records ---
    chunk_head = fetch_mcap_bytes(
        mcap_url,
        CHUNK_HEADER_FIXED + 4, # max(name_len)=4
        chunk_file_offset + RECORD_PREFIX # Skip the Chunk Record Prefix
    )
    # Read the prefix value from the Chunk Header's compression field.
    # name_len=0, no compression; name_len=3, lz4; name_len=4, zstd
    name_len, = u32(chunk_head, 28)
    # Read the compression algorithm: empty string, 'lz4', 'zstd'
    compression = chunk_head[32 : 32 + name_len].decode() # '' or 'lz4' or 'zstd'
    # Read the Chunk Header records_length
    records_length, = u64(chunk_head, 32 + name_len)
    # Start from the first record
    records_start = chunk_file_offset + RECORD_PREFIX + CHUNK_HEADER_FIXED + name_len

    if compression:
        # Compressed chunk: uncompressed_byte_offset is an offset within the decompressed stream, so decompress the chunk whole first
        records_bytes = decompress_chunk_record(
            mcap_url, records_start, records_length, compression,
            u64(chunk_head, 16)[0],   # uncompressed_size, the zstd decompression upper bound
        )
        msg_record = records_bytes[uncompressed_byte_offset:
                            uncompressed_byte_offset + RECORD_PREFIX + msg_record_length]
    else:
        # Uncompressed chunk: uncompressed_byte_offset is the in-file offset; one range read targets the message record
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
    """Slice one message's CDR out of a records-section byte segment read from an uncompressed chunk.

    span_bytes is a segment of the records section starting at span_start; the sliced record's opcode and length fields must match the index,
    otherwise return None (e.g. the chunk is actually compressed, so in-file and in-stream offsets differ); the caller falls back to fetch_data_bytes.
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
    Read and deserialize messages in an mcap randomly by locations.

    All topics have their CDR bytes deserialized into rosbags message objects before returning, including FFMPEGPacket.
    Decoding FFMPEGPacket video (-> image pixels) is the caller's responsibility.
    fetch_message may be called concurrently from multiple threads: the only shared mutable state (typestore registration) is lock-protected.
    """

    def __init__(self):
        self._typestore = get_typestore(Stores.ROS2_HUMBLE)
        self._registered_types: set[str] = set()  # Names of registered custom types, avoiding duplicate register calls
        self._register_lock = threading.Lock()    # When multiple threads first hit the same type, only one performs the register

    def _ensure_type_registered(self, msg_type: str, msg_def: str) -> None:
        """If msg_type is not in the standard library and not yet registered, register it dynamically into the typestore with msg_def"""
        if msg_type in self._registered_types:
            return
        with self._register_lock:
            if msg_type in self._registered_types:  # Already registered by another thread while waiting for the lock
                return
            try:
                self._typestore.deserialize_cdr(b"\x00\x01\x00\x00", msg_type)
            except Exception:
                self._typestore.register(get_types_from_msg(msg_def, msg_type))
            self._registered_types.add(msg_type)

    def fetch_message(self, msg_type: str, msg_def: str, locations: list[tuple]):
        """
        Takes the msg_type/msg_def/locations produced by mcap_player.play_messages as input, returns the deserialized message list.

        Takes the msg_type/msg_def/locations produced by mcap_player.play_messages as input; for each locating tuple in locations
        it calls fetch_data_bytes + deserialize_cdr once, returning a list[ros2 message]. The caller decides further processing by msg_type (e.g. FFMPEGPacket video decoding).
        A locations length of 1 returns a single-element list; length > 1 (ffmpeg GOP) returns the whole segment's frame list.
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
        """Read one range covering a run of adjacent messages in the same chunk, then slice and deserialize one by one.

        span_messages is [(msg_type, msg_def, uncompressed_byte_offset, record_length), ...] and may span topics;
        for uncompressed chunks the records-section start is computed directly, saving the request that would read the chunk header.
        Messages failing the slice check (compressed chunks) fall back to per-message fetch_data_bytes. The return order matches span_messages.
        """
        assert span_messages, "span_messages 不能为空"
        span_start = min(span_message[2] for span_message in span_messages)
        span_end = max(span_message[2] + RECORD_PREFIX + span_message[3] for span_message in span_messages)
        # In an uncompressed chunk's header, compression is an empty string (name_len=0) and the records section immediately follows
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
