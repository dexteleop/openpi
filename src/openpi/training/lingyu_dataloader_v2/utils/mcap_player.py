"""Play every message in an mcap one by one, advancing purely by each record's own length.

It never looks at file metadata (Footer / Summary / Statistics / MessageIndex), so even a file from a power cut,
a half-recorded mcap, or one whose statistics were never finished can still play out every message already on disk,
ending naturally at the first corrupt record.

Every Message record in an mcap lies within exactly one Chunk and has a unique start offset in that Chunk's
records byte stream, therefore
(chunk_file_offset, uncompressed_byte_offset) is the primary key locating this message.
chunk_file_offset is the absolute byte position of the Chunk record in the file.

Usage::
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
    # MCAP Record types
    OP_CHANNEL, OP_MESSAGE, OP_CHUNK, OP_SCHEMA,
    # Common MCAP structure lengths
    MAGIC_SIZE, OPCODE_PREFIX, RECORD_PREFIX, CHUNK_HEADER_FIXED,
    # Byte stream to integer
    u16, u32, u64,
    # MCAP utility functions
    fetch_mcap_bytes, fetch_mcap_length, decompress_chunk_record
)

# Information needed for each message
_PER_MESSAGE_KEYS = (
    "chunk_file_offset", "uncompressed_byte_offset", "record_length",  # message_location
    "topic_name",  # resolved from channel_id via Channel records
    "msg_type",    # ROS2 type string, e.g. "sensor_msgs/msg/JointState"; passed to get_message()
    "msg_def",     # ROS2 msg definition string, used to register custom types in typestore
    "log_time",    # timestamp_ns
)

# Prefetch bytes for sequential playback: when the _fetch_buffered buffer misses, read this many extra bytes beyond the requested length.
# Why it is needed: the file layout is | Chunk | MessageIndex x C | Chunk | ..., one MessageIndex per channel;
#   they are top-level records at the same level as Chunks. The main loop does not use their content but must still read each 9-byte prefix to get the length and skip it;
#   without buffering, after every chunk there would be about 50 S3 round trips fetching just 9 bytes.
# Effect: fetch all MessageIndex records after this chunk plus the next chunk's prefix + head in one go, after which only
#   one more request is needed for the next chunk's records. Uncompressed chunks prefetch while reading records; for compressed
#   chunks decompress_chunk_record reads separately, and the prefetch is triggered by a prefix miss on the first following MessageIndex.
# Value basis: total MessageIndex size ~= 15 x channel count + 16 x message count, measured 20-30KB; 64KB leaves about 2x headroom.
#   When exceeded, it only costs one extra request per 64KB; parsing results stay the same.
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
    Yield each message's locating info, parsed format and filter info in the order messages are stored in the mcap
    Input: the file path of the rosbag-mounted file
    Output: for one message, all data in _PER_MESSAGE_KEYS,
          yielding only topics selected by filter_topics() plus extra_topics
    """
    selected_mcap_topics = filter_topics().keys() | set(extra_topics)
    file_size = fetch_mcap_length(mcap_path)
    channel_topics: dict[int, str] = {}    # channel_id -> topic_name, accumulated as Channel records are encountered
    channel_schemas: dict[int, int] = {}   # channel_id -> schema_id
    schema_names: dict[int, str] = {}      # schema_id  -> msg_type string
    schema_msgdefs: dict[int, str] = {}    # schema_id  -> msg definition string
    pos = MAGIC_SIZE    # pos points at the first Record
    read_buffer: tuple[int, bytes] = (0, b"")   # (buffer start offset, bytes), the sequential-read prefetch buffer
    while pos + RECORD_PREFIX <= file_size:
        # --- Fetch the Record Prefix ---
        header, read_buffer = _fetch_buffered(mcap_path, read_buffer, RECORD_PREFIX, pos)
        if len(header) < RECORD_PREFIX:
            break
        top_record_length, = u64(header, OPCODE_PREFIX) # Get the current Record's length; a Record may also contain nested Records

        # --- Message Records exist only inside Chunk Records, so only Chunk Records are read ---
        if header[0] == OP_CHUNK:
            chunk_file_offset = pos  # the chunk's absolute offset in the file, used as the locating primary key

            # --- Parse the chunk header to locate the start of records ---
            chunk_head, read_buffer = _fetch_buffered(
                mcap_path, read_buffer,
                CHUNK_HEADER_FIXED + 4, # max(name_len)=4
                pos + RECORD_PREFIX
            )
            # name_len=0, no compression; name_len=3, lz4; name_len=4, zstd
            name_len, = u32(chunk_head, 28)
            # Read the compression algorithm: empty string, 'lz4', 'zstd'
            compression = chunk_head[32 : 32 + name_len].decode() # '' or 'lz4' or 'zstd'
            # Read the Chunk Header records_length
            records_length, = u64(chunk_head, 32 + name_len)
            # Start from the first record
            records_start = pos + RECORD_PREFIX + CHUNK_HEADER_FIXED + name_len
            # Read the whole Chunk Record
            if compression:
                records_bytes = decompress_chunk_record(
                    mcap_path, records_start, records_length, compression,
                    u64(chunk_head, 16)[0],  # uncompressed_size, the zstd decompression upper bound
                )
            else:
                # Also prefetch the following MessageIndex records and the next chunk head; usually one request suffices for this chunk
                records_bytes, read_buffer = _fetch_buffered(
                    mcap_path, read_buffer, records_length, records_start)
            # Handed to fetch_data_bytes for reuse: when downstream fetches this chunk's messages one by one it slices directly, without requesting S3 again
            remember_chunk_records(mcap_path, chunk_file_offset, records_bytes)

            uncompressed_byte_offset, records_length = 0, len(records_bytes)
            while uncompressed_byte_offset + RECORD_PREFIX <= records_length:
                inner_record_length, = u64(records_bytes, uncompressed_byte_offset + OPCODE_PREFIX)
                body_at = uncompressed_byte_offset + RECORD_PREFIX
                # A half-written corrupt record from a power cut; stop here
                if body_at + inner_record_length > records_length:
                    break
                inner_op = records_bytes[uncompressed_byte_offset]

                # --- Read the channel_id -> topic_name, schema_id mapping ---
                # No mapping is built for unselected topics; their messages are skipped below based on this
                if inner_op == OP_CHANNEL:
                    b = records_bytes[body_at : body_at + inner_record_length]
                    cid, = u16(b, 0); sid, = u16(b, 2); tl, = u32(b, 4)
                    topic_name = b[8 : 8 + tl].decode()
                    if topic_name in selected_mcap_topics:
                        channel_topics[int(cid)] = topic_name
                        channel_schemas[int(cid)] = int(sid)

                # --- Read the schema_id -> msg_type mapping ---
                elif inner_op == OP_SCHEMA:
                    b = records_bytes[body_at : body_at + inner_record_length]
                    sid, = u16(b, 0); nl, = u32(b, 2)
                    name = b[6 : 6 + nl].decode()
                    enc_len, = u32(b, 6 + nl)
                    dat_len, = u32(b, 6 + nl + 4 + enc_len)
                    dat = b[6 + nl + 4 + enc_len + 4 : 6 + nl + 4 + enc_len + 4 + dat_len]
                    schema_names[int(sid)] = name
                    schema_msgdefs[int(sid)] = dat.decode("utf-8", errors="replace")

                # --- Read messages, yielding locating info ---
                elif inner_op == OP_MESSAGE:
                    cid, = u16(records_bytes, body_at)
                    # A cid not in the mapping means the topic is unselected; skip to the next message
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

        pos += RECORD_PREFIX + top_record_length  # pos points to the next Record


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
    return 'h265'  # Kept identical to the original behavior by default


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
        # Keyframe detection must deserialize FFMPEGPacket.data; the fetcher carries its own typestore and custom type registrations
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
        Record the frame-data indices that can decode this frame image, for later operations

        Returns [I] when this frame is an IDR.
        Returns [I P ...] when an IDR (keyframe) is detected.
        Returns [] if no IDR has been seen yet for this message.
        """
        # Only inspect the data field's bitstream: bytes of header/encoding and similar fields may accidentally form 00 00 01 fake start codes,
        # causing missed IDR detection and unbounded GOP list growth (OOM); data's offset within the message varies with string lengths, so deserialization is required
        payload = bytes(self._fetcher.fetch_message(msg_type, msg_def, [loc])[0].data)
        if _has_idr(payload):
            ffmpeg_buffers[topic_name] = [loc]
        elif topic_name not in ffmpeg_buffers:
            return []  # No keyframe seen yet, so this frame cannot be decoded; discard
        else:
            ffmpeg_buffers[topic_name].append(loc)
        return list(ffmpeg_buffers[topic_name])

    def play_messages(self, extra_topics: tuple[str, ...] = ()):
        """
        Output: topic_name, msg_type, msg_def, log_time, locations

        extra_topics: mcap topics to yield in addition to the user-selected data (e.g. key signals marking episode start/end),
                      these topics are not training data and thus are not written into USER_SELECTED_TOPICS.
        """
        mcap_video_topics_gop = load_mcap_video_topics_gop()  # Keys are already mcap topic names, no mapping needed
        ffmpeg_buffers: dict[str, list] = {}

        # Play the messages of one mcap sequentially
        msg_iter = _play_messages(self.mcap_url, extra_topics)

        for msg in msg_iter:
            topic = msg["topic_name"]
            loc = (self.mcap_url,
                   msg["chunk_file_offset"],
                   msg["uncompressed_byte_offset"],
                   msg["record_length"])
            if mcap_video_topics_gop.get(topic,1) > 1:  # GOP>1: inter-frame coding, must trace back to a keyframe
                locations = self._accumulate_frames_from_keyframe(
                    ffmpeg_buffers, topic, msg["msg_type"], msg["msg_def"], loc)
                if not locations:  # No keyframe to rely on, skip this frame
                    continue
            else:
                locations = [loc]
            # Convert locating integers and timestamps to strings uniformly before yielding
            locations_str = [(loc[0], str(loc[1]), str(loc[2]), str(loc[3]))
                             for loc in locations]
            yield topic, msg["msg_type"], msg["msg_def"], str(msg["log_time"]), locations_str
