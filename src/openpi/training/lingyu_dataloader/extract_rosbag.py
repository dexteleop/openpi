#!/usr/bin/env python3
"""
Extract ROS2 bag data and generate a self-explainable dataset in a single pass.

Each Parquet file embeds full metadata in its file-level schema metadata:
  - dataset_meta  : global info + episodes (same content in every parquet)
  - topic_channel : topic, msg_type, msg_def (full ros2msg definition)
  - field_mapping : column name -> ROS type string

Episodes are determined from XR hand inputs and stored in the metadata;
data rows do NOT carry an ``episode`` column.

Output layout:
    dataset/
    ├── data.parquet              # all topics merged (one row group per topic)
    └── <video_name>.mp4          # remuxed HEVC video

Usage:
    python extract_rosbag.py <bag_path> [output_dir]
    python extract_rosbag.py /mnt/data6t/shihaoran_data/robot_20251103-J0GV_20260330-135132-828453_20260330-135359-953453 /mnt/data6t/shihaoran_data/test_lerobot3
    python extract_rosbag.py /mnt/data6t/shihaoran_data/robot_20251103-J0GV_20260427-094752-838208_20260427-095244-401208 /mnt/data6t/shihaoran_data/test_parquetvideo
    python extract_rosbag.py /mnt/data6t/shihaoran_data/rec_20260402_182236_7521abbe /mnt/data522g/shihaoran_data/test_rec

Dependencies:
    pip install rosbags pyarrow numpy
    ffmpeg must be on PATH.
"""

import os
import queue
import sys
import json
import subprocess
import threading
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict, OrderedDict
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from rosbags.interfaces.typing import Nodetype

# ─── Constants ────────────────────────────────────────────────────────────────

FPS = 45  # 保证MP4每一帧的间隔都是固定的，防止MP4无法被单独查看
ROBOT_TYPE = "TeleAvatar"
ROS_DISTRO = "humble"

X_BUTTON = 2  # XR hand input: episode start
Y_BUTTON = 3  # XR hand input: episode end

MERGED_PARQUET = "data.parquet"
BATCH_SIZE = 1000

# ROS primitive -> PyArrow type
ROS_TO_ARROW = {
    "bool":    pa.bool_(),
    "int8":    pa.int8(),
    "uint8":   pa.uint8(),
    "int16":   pa.int16(),
    "uint16":  pa.uint16(),
    "int32":   pa.int32(),
    "uint32":  pa.uint32(),
    "int64":   pa.int64(),
    "uint64":  pa.uint64(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "string":  pa.string(),
}

# ─── Utility: topic name ─────────────────────────────────────────────────────


def sanitize_topic_name(topic: str) -> str:
    return topic.strip("/").replace("/", "_")


# ─── Utility: ROS message helpers ─────────────────────────────────────────────


def msg_to_dict(val):
    """Recursively convert a ROS message to a plain Python dict/list."""
    if hasattr(val, "__dataclass_fields__"):
        return {
            f: msg_to_dict(getattr(val, f))
            for f in val.__dataclass_fields__
            if f != "__msgtype__"
        }
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (list, tuple)):
        return [msg_to_dict(v) for v in val]
    elif isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    else:
        return val


def infer_ros_type(val):
    """Infer a ROS-like type string from a Python/numpy value."""
    if isinstance(val, bool):
        return "bool"
    elif isinstance(val, (np.int8,)):
        return "int8"
    elif isinstance(val, (np.uint8,)):
        return "uint8"
    elif isinstance(val, (np.int16,)):
        return "int16"
    elif isinstance(val, (np.uint16,)):
        return "uint16"
    elif isinstance(val, (np.int32,)):
        return "int32"
    elif isinstance(val, (np.uint32,)):
        return "uint32"
    elif isinstance(val, (np.int64,)):
        return "int64"
    elif isinstance(val, (np.uint64,)):
        return "uint64"
    elif isinstance(val, (int,)):
        return "int64"
    elif isinstance(val, np.float32):
        return "float32"
    elif isinstance(val, (float, np.float64)):
        return "float64"
    elif isinstance(val, str):
        return "string"
    elif isinstance(val, np.ndarray):
        dtype_map = {
            np.dtype("float64"): "float64[]",
            np.dtype("float32"): "float32[]",
            np.dtype("int64"):   "int64[]",
            np.dtype("int32"):   "int32[]",
            np.dtype("uint8"):   "uint8[]",
            np.dtype("uint16"):  "uint16[]",
            np.dtype("int8"):    "int8[]",
            np.dtype("bool"):    "bool[]",
        }
        return dtype_map.get(val.dtype, f"{val.dtype}[]")
    elif isinstance(val, (list, tuple)):
        if len(val) > 0 and isinstance(val[0], str):
            return "string[]"
        return "json"
    else:
        return "string"


def resolve_field_types(typestore, msgtype, prefix=""):
    """Resolve flattened column specs from the ROS message definition.

    Types come from the msgdef registered in the typestore, NOT from sample
    values, so an empty variable-length array no longer leaves a column's type
    ambiguous (an empty list carries no element type).  Column names match the
    dot-separated keys produced by flatten_msg().

    Returns OrderedDict: column name -> (arrow_type, ros_type_str, kind)
    where kind is "scalar", "list" or "json".
    """
    spec = OrderedDict()
    for fname, ftype in typestore.fielddefs[msgtype][1]:
        key = f"{prefix}.{fname}" if prefix else fname
        node = ftype[0]

        if node == Nodetype.NAME:
            # Nested message: expand into <key>.<subfield> columns
            spec.update(resolve_field_types(typestore, ftype[1], key))
        elif node in (Nodetype.SEQUENCE, Nodetype.ARRAY):
            inner = ftype[1][0]
            if inner[0] == Nodetype.BASE:
                prim = inner[1][0]
                elem = ROS_TO_ARROW.get(prim, pa.string())
                spec[key] = (pa.list_(elem), f"{prim}[]", "list")
            else:
                # Sequence of nested messages -> JSON string column
                spec[key] = (pa.string(), f"{inner[1]}[]", "json")
        else:
            prim = ftype[1][0]
            spec[key] = (ROS_TO_ARROW.get(prim, pa.string()), prim, "scalar")
    return spec


def flatten_msg(msg, spec=None, prefix=""):
    """Flatten a ROS message into an OrderedDict with dot-separated keys.

    - Nested ROS messages: recursively flattened
    - numpy arrays: kept as numpy (converted to list at write time)
    - Lists of nested messages: serialized as JSON string
    - Lists of primitives (e.g. name: string[]): kept as list

    ``spec`` comes from resolve_field_types().  When given, the JSON-vs-list
    decision follows the message definition rather than the runtime value, so
    an EMPTY sequence is handled exactly like a populated one.  When None the
    decision falls back to inspecting values (msgdef unavailable).
    """
    result = OrderedDict()
    for field in msg.__dataclass_fields__:
        if field == "__msgtype__":
            continue
        val = getattr(msg, field)
        key = f"{prefix}.{field}" if prefix else field

        if hasattr(val, "__dataclass_fields__"):
            result.update(flatten_msg(val, spec, key))
        elif isinstance(val, np.ndarray):
            result[key] = val
        elif isinstance(val, (list, tuple)):
            if spec is not None:
                as_json = spec.get(key, (None, None, "list"))[2] == "json"
            else:
                as_json = len(val) > 0 and hasattr(val[0], "__dataclass_fields__")
            if as_json:
                result[key] = json.dumps(
                    [msg_to_dict(v) for v in val], ensure_ascii=False
                )
            else:
                result[key] = list(val)
        else:
            result[key] = val
    return result


def is_ffmpeg_topic(conn) -> bool:
    return "ffmpeg" in conn.msgtype.lower() or "ffmpeg" in conn.topic.lower()


# ─── HEVC access-unit inspection ─────────────────────────────────────────────
#
# Each FFMPEGPacket payload is one annex-B access unit of the raw HEVC stream.
# Recording rarely begins on an IDR, so the leading packets reference a frame
# that was never captured.  ffmpeg's mp4 muxer discards those packets and still
# exits 0, which would leave the sidecar timestamps longer than the mp4 and
# shift every frame_index.  Detecting IDRs here lets the extractor drop exactly
# the same packets, together with their timestamps.

_NAL_START3 = b"\x00\x00\x01"


def _iter_nal_types(data: bytes):
    """Yield nal_unit_type for every NAL unit in an annex-B access unit.

    Searching only the 3-byte start code also covers 4-byte ones: 00 00 00 01
    contains 00 00 01 at offset +1, so a single matcher suffices.
    """
    pos = data.find(_NAL_START3)
    n = len(data)
    while pos != -1 and pos + 3 < n:
        yield (data[pos + 3] >> 1) & 0x3F  # type is bits 6..1 of the header
        pos = data.find(_NAL_START3, pos + 3)


def _has_idr(data: bytes) -> bool:
    """True if this access unit is an IDR, i.e. a self-contained decode start.

    The first VCL NAL determines the access-unit type, because the non-VCL NALs
    (VPS 32 / SPS 33 / PPS 34 / SEI 39) always precede it; returning on that
    first VCL NAL keeps the scan short and immune to later bytes.

    FFMPEGPacket carries a ``flags`` field, but it is AV_PKT_FLAG_KEY, which is
    also set on CRA (21) -- a keyframe that may be followed by RASL frames and
    is therefore not a safe cut point.  Parsing the NAL types is stricter.
    """
    for nal_type in _iter_nal_types(data):
        if nal_type < 32:
            return nal_type in (19, 20)  # IDR_W_RADL / IDR_N_LP
    return False


# Map an FFMPEGPacket ``encoding`` string to a valid ffmpeg input demuxer.
# The field may carry pixel-format hints (e.g. 'hevc;nv12;bgr8;rgb8') or use an
# alias ('h265'); keep only the codec token and normalize known aliases.
_FFMPEG_FORMAT_ALIASES = {"h265": "hevc", "h264": "h264", "avc": "h264"}


def ffmpeg_input_format(encoding: str) -> str:
    codec = (encoding or "").split(";")[0].strip().lower()
    return _FFMPEG_FORMAT_ALIASES.get(codec, codec) or "hevc"


def init_typestore(bag_path: str):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(bag_path) as reader:
        add_types = {}
        for conn in reader.connections:
            if conn.msgdef is not None:
                try:
                    msgdef = conn.msgdef
                    if hasattr(msgdef, "data"):
                        msgdef = msgdef.data
                    add_types.update(get_types_from_msg(msgdef, conn.msgtype))
                except Exception:
                    pass
        typestore.register(add_types)
    return typestore


# ─── Episode detection ────────────────────────────────────────────────────────


def find_episode_topic(parquet_rows):
    """Identify the topic to use for episode detection (XR hand inputs)."""
    for candidate in ["/xr/left_hand_inputs", "/xr/right_hand_inputs"]:
        if candidate in parquet_rows and parquet_rows[candidate]:
            return candidate
    for topic in parquet_rows:
        if "hand_inputs" in topic:
            return topic
    return None


def detect_episodes_from_rows(rows):
    """Detect episodes from accumulated hand-input rows via button transitions.

    X button (index 2) 0->1 = episode start
    Y button (index 3) 0->1 = episode end

    Only the RISING EDGE counts: Joy publishes at high rate, so holding X down
    yields many messages with buttons[X] == 1, and just the one that follows a 0
    opens an episode.  Bits other than X and Y are ignored, so a press mixed
    with other buttons still registers.

    Malformed and boundary cases follow the online recorder's rules:
      - a ``buttons`` array too short to hold both indices is skipped, never
        indexed (a single bad message must not abort the whole extraction);
      - the first message only seeds the reference state, so a button already
        held down when the bag starts is not mistaken for a press;
      - a second X before the matching Y is a re-record: it replaces the
        pending start instead of nesting.
    """
    if not rows:
        return []

    min_len = max(X_BUTTON, Y_BUTTON) + 1
    events = []
    prev = None
    n_short = 0
    for row in rows:
        buttons = row.get("buttons")
        if buttons is None:
            continue
        ts = row["__timestamp_ns__"]
        if isinstance(buttons, np.ndarray):
            btn = buttons.astype(np.int32)
        else:
            btn = np.array(buttons, dtype=np.int32)

        # Guard the indexing. Without this a driver hiccup that emits a short
        # buttons array raises IndexError here, and because this runs after the
        # bag has been read and the videos encoded, the whole run is lost.
        if btn.shape[0] < min_len:
            n_short += 1
            continue

        # No earlier message means no transition to observe, so the first row
        # only becomes the reference for the next one.
        if prev is None:
            prev = btn
            continue

        if prev[X_BUTTON] == 0 and btn[X_BUTTON] == 1:
            events.append((ts, "start"))
        if prev[Y_BUTTON] == 0 and btn[Y_BUTTON] == 1:
            events.append((ts, "end"))
        prev = btn

    if n_short:
        print(f"  WARNING: skipped {n_short} hand-input message(s) with fewer "
              f"than {min_len} buttons")

    episodes = []
    pending_start = None
    for ts, kind in events:
        if kind == "start":
            # A start while one is already pending means the operator pressed X
            # again to redo the take; the abandoned attempt is reported so it is
            # never silently absent from the dataset.
            if pending_start is not None:
                print(f"  WARNING: RE-RECORD at ts={ts}, discarding unfinished "
                      f"START at ts={pending_start}")
            pending_start = ts
        elif kind == "end":
            if pending_start is not None:
                episodes.append({"start_ts": pending_start, "end_ts": ts})
                pending_start = None
            else:
                print(f"  WARNING: END at ts={ts} has no matching START, skipping")

    if pending_start is not None:
        print(f"  WARNING: START at ts={pending_start} has no matching END, skipping")

    return episodes


# ─── Metadata builders ───────────────────────────────────────────────────────


def build_episodes_meta(episodes):
    """Map episode index -> {start, end}."""
    result = {}
    for idx, ep in enumerate(episodes):
        result[str(idx)] = {
            "start": ep["start_ts"],
            "end": ep["end_ts"],
        }
    return result


def build_dataset_meta(total_episodes, total_videos, total_topics, episodes_meta):
    """Global dataset metadata (identical content in every parquet)."""
    return {
        "robot_type": ROBOT_TYPE,
        "ros_distro": ROS_DISTRO,
        "total_episodes": total_episodes,
        "total_videos": total_videos,
        "total_topics": total_topics,
        "episodes": episodes_meta,
    }


def _encode_metadata(dataset_meta, topic, msgtype, msgdef_text, field_types):
    """Encode 3 metadata dicts into bytes for parquet schema metadata.

    Keys: dataset_meta, schema, field_mapping
    """
    return {
        b"dataset_meta":  json.dumps(dataset_meta).encode(),
        b"topic_channel":       json.dumps({
            "topic": topic,
            "msg_type": msgtype,
            "msg_def": msgdef_text or "",
        }).encode(),
        b"field_mapping": json.dumps(dict(field_types)).encode(),
    }


# ─── Schema & Table building ─────────────────────────────────────────────────


def build_pa_schema(topic, msgtype, msgdef_text, dataset_meta_dict,
                    spec=None, first_row=None):
    """Build a PyArrow schema with full self-explainable metadata.

    Column order: __timestamp_ns__ | <message fields ...>
    field_mapping order: <message fields ...> | __timestamp_ns__

    ``spec`` (from resolve_field_types) types every column from the message
    definition.  ``first_row`` is only consulted as a fallback when the msgdef
    is unavailable, in which case types are inferred from values.
    """
    field_types = OrderedDict()
    pa_fields = [
        pa.field("__timestamp_ns__", pa.int64()),
    ]

    if spec is not None:
        for key, (arrow_type, ros_type, _kind) in spec.items():
            field_types[key] = ros_type
            pa_fields.append(pa.field(key, arrow_type))
    else:
        for key, val in first_row.items():
            if key == "__timestamp_ns__":
                continue
            ros_type = infer_ros_type(val)
            field_types[key] = ros_type

            if ros_type.endswith("[]") and ros_type != "string[]":
                elem_str = ros_type[:-2]
                elem_pa = ROS_TO_ARROW.get(elem_str, pa.float64())
                pa_fields.append(pa.field(key, pa.list_(elem_pa)))
            elif ros_type == "string[]":
                pa_fields.append(pa.field(key, pa.list_(pa.string())))
            elif ros_type == "json":
                pa_fields.append(pa.field(key, pa.string()))
                field_types[key] = "json"
            elif ros_type in ROS_TO_ARROW:
                pa_fields.append(pa.field(key, ROS_TO_ARROW[ros_type]))
            else:
                pa_fields.append(pa.field(key, pa.string()))

    # __timestamp_ns__ last in field_mapping (matches design doc)
    field_types["__timestamp_ns__"] = "int64"

    metadata = _encode_metadata(
        dataset_meta_dict, topic, msgtype, msgdef_text, field_types,
    )
    return pa.schema(pa_fields, metadata=metadata)


def rows_to_table(rows, pa_schema):
    """Convert list of flattened dicts to a PyArrow Table with explicit schema."""
    arrays = []
    for field in pa_schema:
        col_name = field.name
        raw = [r.get(col_name) for r in rows]

        if isinstance(field.type, pa.ListType):
            converted = []
            for v in raw:
                if isinstance(v, np.ndarray):
                    converted.append(v.tolist())
                elif v is None:
                    converted.append(None)
                else:
                    converted.append(v)
            arrays.append(pa.array(converted, type=field.type))
        else:
            arrays.append(pa.array(raw, type=field.type))

    return pa.table(
        {field.name: arr for field, arr in zip(pa_schema, arrays)},
        schema=pa_schema,
    )


# ─── Merge all parquets into data.parquet ────────────────────────────────────


def _merge_parquets(output_dir, dataset_meta_override=None):
    """Merge all per-topic parquet files into a single data.parquet.

    Each topic becomes one row group.  A ``topic`` column (string) is prepended.
    """
    # Collect source parquets (exclude the output itself)
    paths = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".parquet") and f != MERGED_PARQUET
    )
    if not paths:
        return

    # ── detect column names claimed by topics with incompatible types ──
    # A name shared by several topics under different types cannot be unioned
    # (e.g. `data` is list<float32> on Float32MultiArray, int32 on Int32,
    # string on String).  Every occurrence of such a name is prefixed with its
    # topic, so the merged schema is unambiguous regardless of file ordering.
    col_types = defaultdict(set)
    for path in paths:
        for field in pq.ParquetFile(path).schema_arrow:
            col_types[field.name].add(field.type)
    conflict_cols = {n for n, t in col_types.items() if len(t) > 1}
    if conflict_cols:
        print(f"[Merge] Topic-qualifying {len(conflict_cols)} conflicting "
              f"column(s): {', '.join(sorted(conflict_cols))}")

    # ── scan schemas ──
    union_cols = OrderedDict()   # col_name -> pa.DataType
    file_infos = []
    topics_meta = {}
    dataset_meta = None

    for path in paths:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        raw_meta = schema.metadata or {}

        topic_channel = (
            json.loads(raw_meta[b"topic_channel"])
            if b"topic_channel" in raw_meta else {}
        )
        topic = topic_channel.get(
            "topic", os.path.splitext(os.path.basename(path))[0],
        )
        field_mapping = (
            json.loads(raw_meta[b"field_mapping"])
            if b"field_mapping" in raw_meta else {}
        )
        if dataset_meta is None and b"dataset_meta" in raw_meta:
            dataset_meta = json.loads(raw_meta[b"dataset_meta"])

        rename_map = {}

        for field in schema:
            out_name = rename_map.get(field.name, field.name)
            if out_name in conflict_cols:
                out_name = f"{sanitize_topic_name(topic)}__{out_name}"
                rename_map[field.name] = out_name
            if out_name not in union_cols:
                union_cols[out_name] = field.type

        # Keep field_mapping aligned with the renamed columns
        if rename_map:
            field_mapping = {
                rename_map.get(k, k): v for k, v in field_mapping.items()
            }

        file_infos.append({
            "path": path,
            "topic": topic,
            "num_rows": pf.metadata.num_rows,
            "rename_map": rename_map,
        })
        topics_meta[topic] = {
            "msg_type": topic_channel.get("msg_type", ""),
            "msg_def": topic_channel.get("msg_def", ""),
            "field_mapping": field_mapping,
            "row_count": pf.metadata.num_rows,
        }

    # ── build union schema ──
    pa_fields = [pa.field("topic", pa.string())]
    for name, typ in union_cols.items():
        pa_fields.append(pa.field(name, typ))

    # Use caller-provided dataset_meta if source files lack it (streaming mode)
    # or only have an empty placeholder.
    if not dataset_meta and dataset_meta_override is not None:
        dataset_meta = dict(dataset_meta_override)

    merged_meta = {}
    if dataset_meta is not None:
        dataset_meta["total_topics"] = len(file_infos)
        merged_meta[b"dataset_meta"] = json.dumps(dataset_meta).encode()
    merged_meta[b"topics"] = json.dumps(
        topics_meta, ensure_ascii=False,
    ).encode()

    union_schema = pa.schema(pa_fields, metadata=merged_meta)

    # ── write merged file ──
    out_path = os.path.join(output_dir, MERGED_PARQUET)
    writer = pq.ParquetWriter(out_path, union_schema)
    total_rows = 0

    for info in file_infos:
        table = pq.read_table(info["path"])
        num = len(table)
        rename_map = info["rename_map"]

        src = {}
        for i, f in enumerate(table.schema):
            src[rename_map.get(f.name, f.name)] = table.column(i)

        aligned = []
        for f in union_schema:
            if f.name == "topic":
                aligned.append(
                    pa.array([info["topic"]] * num, type=pa.string()),
                )
            elif f.name in src:
                aligned.append(src[f.name])
            else:
                aligned.append(pa.nulls(num, type=f.type))

        merged_table = pa.table(
            {f.name: a for f, a in zip(union_schema, aligned)},
            schema=union_schema.remove_metadata(),
        )
        # Pin one row group per topic: generate_episode.py maps the i-th topic
        # to the i-th row group, and pyarrow would otherwise split any table
        # over ~1.05M rows into several row groups, silently breaking it.
        writer.write_table(merged_table, row_group_size=max(num, 1))
        total_rows += num

    writer.close()

    out_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"[Merge] {MERGED_PARQUET}: {total_rows:,} rows, "
          f"{len(file_infos)} row groups, {out_size:.1f} MB")

    # ── remove individual parquets ──
    for info in file_infos:
        os.remove(info["path"])
    print(f"[Merge] Removed {len(file_infos)} individual parquet files")


# ─── Buffered row group writer ──────────────────────────────────────────────


def _flush_buffer(topic, ctx):
    """Flush a topic's row buffer to its ParquetWriter as a row group."""
    if not ctx["buffer"]:
        return
    table = rows_to_table(ctx["buffer"], ctx["schema"])
    ctx["writer"].write_table(table)
    row_count = len(ctx["buffer"])
    ctx["row_count"] += row_count
    ctx["buffer"] = []


# ─── Pipe writer thread ──────────────────────────────────────────────────────


def _pipe_writer(proc, q):
    """Drain a queue by writing bytes to proc.stdin; close stdin on sentinel.

    If ffmpeg exits early (e.g. codec mismatch) the pipe breaks; swallow the
    error and keep draining the queue so the producer never blocks forever on a
    full queue.  The non-zero exit code is reported later during finalization.
    """
    broken = False
    while True:
        data = q.get()
        if data is None:
            break
        if not broken:
            try:
                proc.stdin.write(data)
            except (BrokenPipeError, ValueError, OSError):
                broken = True  # ffmpeg gone; drop remaining frames
        q.task_done()
    try:
        proc.stdin.close()
    except (BrokenPipeError, ValueError, OSError):
        pass
    q.task_done()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <bag_path> [output_dir]")
        sys.exit(1)

    bag_path = sys.argv[1]
    output_dir = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(os.path.dirname(bag_path), "extracted")
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Bag:    {bag_path}")
    print(f"Output: {output_dir}\n")

    t_program_start = time.monotonic()

    typestore = init_typestore(bag_path)

    # ── 1. Classify topics & collect connection metadata ──────────────
    t_classify_start = time.monotonic()
    topic_msgtype = {}
    topic_msgdef = {}
    ffmpeg_topics = set()

    with Reader(bag_path) as reader:
        for c in reader.connections:
            topic_msgtype[c.topic] = c.msgtype

            raw_def = c.msgdef
            if raw_def is not None:
                if hasattr(raw_def, "data"):
                    raw_def = raw_def.data
                if isinstance(raw_def, bytes):
                    raw_def = raw_def.decode("utf-8", errors="replace")
                topic_msgdef[c.topic] = raw_def
            else:
                topic_msgdef[c.topic] = ""

            if is_ffmpeg_topic(c):
                ffmpeg_topics.add(c.topic)

            tag = "[VIDEO]" if c.topic in ffmpeg_topics else "[DATA] "
            print(f"  {tag} {c.topic}  ({c.msgtype})")
    print()
    t_classify_end = time.monotonic()
    print(f"[Duration] Classify topics: {t_classify_end - t_classify_start:.2f}s")

    # ── 2. Open ffmpeg pipes for video topics ─────────────────────────
    t_open_start = time.monotonic()
    video_procs = {}
    video_queues = {}
    video_threads = {}
    video_counts = {}
    video_timestamps = defaultdict(list)

    # Per-camera IDR gate: a topic starts feeding the pipe at its own first IDR.
    # Cameras are gated independently because their IDRs are not aligned in the
    # bag, and a shared gate would truncate the GOP of whichever camera hit its
    # IDR earlier.
    cameras_with_idr = set()
    video_skipped = defaultdict(int)

    # Probe the first packet of each video topic to read its codec (the
    # FFMPEGPacket ``encoding`` field, e.g. 'hevc' or 'mjpeg').  The ffmpeg
    # input demuxer (-f) must match the actual codec, not a hard-coded one.
    video_encoding = {}
    with Reader(bag_path) as reader:
        for conn, ts, rawdata in reader.messages():
            topic = conn.topic
            if topic in ffmpeg_topics and topic not in video_encoding:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                video_encoding[topic] = msg.encoding
                if len(video_encoding) == len(ffmpeg_topics):
                    break

    for topic in sorted(ffmpeg_topics):
        out_path = os.path.join(output_dir, sanitize_topic_name(topic) + ".mp4")
        raw_enc = video_encoding.get(topic, "hevc")
        enc = ffmpeg_input_format(raw_enc)
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-nostats", "-loglevel", "error",
                "-f", enc, "-r", str(FPS),
                "-i", "pipe:0",
                "-c:v", "copy",
                "-movflags", "+faststart",
                out_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        q: queue.Queue = queue.Queue(maxsize=16)
        video_queues[topic] = q
        video_threads[topic] = threading.Thread(
            target=_pipe_writer, args=(proc, q), daemon=True,
        )
        video_threads[topic].start()
        video_procs[topic] = (proc, out_path)
        video_counts[topic] = 0
        print(f"[Video] Pipe: {topic} -> {os.path.basename(out_path)}  ({raw_enc!r} -> -f {enc})")

    t_open_end = time.monotonic()
    print(f"[Duration] Open pipes (video only): {t_open_end - t_open_start:.2f}s")

    # ── 3. Single-pass read ───────────────────────────────────────────
    #
    # Non-video topics are streamed to disk: each topic gets its own
    # ParquetWriter and row buffer.  Once the buffer reaches BATCH_SIZE
    # rows it is flushed as a row group, keeping memory bounded.
    #
    # Hand-input topics (for episode detection) are kept fully in memory
    # because episode detection needs to scan all rows after the bag is read.
    #
    parquet_buffers = {}   # topic -> {schema, writer, buffer, row_count}
    parquet_rows = {}      # hand-input topics only (for episode detection)
    topic_spec = {}        # topic -> resolve_field_types() result (None if N/A)

    print("\nReading bag (single pass) ...")
    total = 0
    t_read_start = time.monotonic()
    t_chunk_start = t_read_start
    with Reader(bag_path) as reader:
        for conn, ts, rawdata in reader.messages():
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            topic = conn.topic

            if topic in ffmpeg_topics:
                data = bytes(msg.data)
                # Withhold packets until this camera's first IDR.  They would be
                # dropped by the muxer anyway (no reference frame yet, and the
                # mp4 would start with garbage); dropping them here discards
                # their timestamps too, so video_counts, len(video_timestamps)
                # and the muxed packet count stay equal and frame_index keeps
                # addressing the right mp4 frame.
                if topic not in cameras_with_idr:
                    if not _has_idr(data):
                        video_skipped[topic] += 1
                        continue
                    cameras_with_idr.add(topic)
                video_queues[topic].put(data)
                video_timestamps[topic].append(ts)
                video_counts[topic] += 1
            else:
                # Resolve column types from the msgdef once per topic; fall
                # back to value-based inference if the definition is missing.
                if topic not in topic_spec:
                    try:
                        topic_spec[topic] = resolve_field_types(
                            typestore, conn.msgtype,
                        )
                    except Exception:
                        topic_spec[topic] = None
                spec = topic_spec[topic]

                row = flatten_msg(msg, spec)
                row["__timestamp_ns__"] = ts

                # Hand-input topics: keep in memory for episode detection.
                if "hand_inputs" in topic:
                    if topic not in parquet_rows:
                        parquet_rows[topic] = []
                    parquet_rows[topic].append(row)
                    continue

                # First message for this topic: open a ParquetWriter.
                if topic not in parquet_buffers:
                    msgtype = topic_msgtype.get(topic, "unknown")
                    msgdef_text = topic_msgdef.get(topic, "")
                    schema = build_pa_schema(
                        topic, msgtype, msgdef_text, {},
                        spec=spec, first_row=row,
                    )
                    out_path = os.path.join(
                        output_dir, sanitize_topic_name(topic) + ".parquet",
                    )
                    writer = pq.ParquetWriter(out_path, schema)
                    parquet_buffers[topic] = {
                        "schema": schema,
                        "writer": writer,
                        "buffer": [],
                        "row_count": 0,
                    }

                parquet_buffers[topic]["buffer"].append(row)
                if len(parquet_buffers[topic]["buffer"]) >= BATCH_SIZE:
                    _flush_buffer(topic, parquet_buffers[topic])

            total += 1
            if total % 5000 == 0:
                chunk_elapsed = time.monotonic() - t_chunk_start
                avg_per_msg = chunk_elapsed / 5000
                print(f"  ... {total} messages  (last 5k: {chunk_elapsed:.2f}s, {avg_per_msg*1000:.2f}ms/msg)", flush=True)
                t_chunk_start = time.monotonic()

    # Flush remaining rows for each topic.
    for topic, ctx in parquet_buffers.items():
        _flush_buffer(topic, ctx)
        ctx["writer"].close()
        print(
            f"[Parquet] {topic}: {ctx['row_count']} msgs -> "
            f"{sanitize_topic_name(topic)}.parquet",
        )

    t_read_end = time.monotonic()
    print(f"  Total: {total} messages  ({t_read_end - t_read_start:.2f}s)\n")

    # ── 4. Detect episodes ────────────────────────────────────────────
    t_ep_start = time.monotonic()
    print("Detecting episodes ...")
    ep_topic = find_episode_topic(parquet_rows)
    episodes = []
    if ep_topic:
        print(f"  Using: {ep_topic}")
        episodes = detect_episodes_from_rows(parquet_rows[ep_topic])
        print(f"  Detected {len(episodes)} episode(s)")
    else:
        print("  WARNING: No XR hand input topic found, 0 episodes")

    t_ep_end = time.monotonic()
    print(f"[Duration] Detect episodes: {t_ep_end - t_ep_start:.2f}s")

    episodes_meta = build_episodes_meta(episodes)

    # Build the single dataset_meta dict (shared by all parquets)
    all_data_topics = len(parquet_buffers) + len(parquet_rows)
    dataset_meta_dict = build_dataset_meta(
        total_episodes=len(episodes),
        total_videos=len(ffmpeg_topics),
        total_topics=all_data_topics + len(ffmpeg_topics),
        episodes_meta=episodes_meta,
    )

    # ── 5. Finalize videos + sidecar timestamp parquets ───────────────
    t_video_start = time.monotonic()
    for topic in sorted(ffmpeg_topics):
        proc, out_path = video_procs[topic]
        q = video_queues[topic]
        q.put(None)  # sentinel to stop writer thread
        video_threads[topic].join()
        ret = proc.wait()
        count = video_counts[topic]

        if ret != 0:
            stderr_text = proc.stderr.read().decode(errors="replace")
            print(f"[Video] WARNING {topic}: ffmpeg exit code {ret}")
            for line in stderr_text.strip().splitlines()[-5:]:
                print(f"    {line}")
        else:
            print(
                f"[Video] {topic}: {count} packets -> {os.path.basename(out_path)}"
            )

        # Report the IDR gate's effect so a truncated head is never silent.
        if video_skipped[topic]:
            print(
                f"[Video] {topic}: skipped {video_skipped[topic]} leading "
                f"packet(s) before the first IDR"
            )
        if topic not in cameras_with_idr:
            print(
                f"[Video] WARNING {topic}: no IDR found in the entire bag, "
                f"no frames written"
            )

        # ── Sidecar timestamps with full metadata ──
        ts_list = video_timestamps[topic]
        ts_path = os.path.join(
            output_dir, sanitize_topic_name(topic) + "_timestamps.parquet"
        )

        # Backstop for drops the IDR gate cannot anticipate (e.g. a corrupt
        # packet mid-stream): compare the sidecar length against what the muxer
        # actually stored.  Count container PACKETS, not decoded frames -- a
        # full decode of an open-GOP HEVC stream reports 2-3 fewer frames
        # (reorder-buffer frames that never flush), which would look like a
        # muxer-side drop.
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "packet=pos", "-of", "csv=p=0", out_path],
                capture_output=True, text=True,
            )
            muxed = len([ln for ln in probe.stdout.splitlines() if ln.strip()])
        except (OSError, ValueError) as exc:
            print(f"[Video] WARNING {topic}: cannot verify frame count ({exc})")
        else:
            if muxed != len(ts_list):
                print(
                    f"[Video] WARNING {topic}: mp4 holds {muxed} packet(s) but "
                    f"sidecar has {len(ts_list)} timestamp(s) "
                    f"(delta={len(ts_list) - muxed}); frame_index is misaligned"
                )

        ts_field_types = OrderedDict([
            ("frame_index", "int64"),
            ("__timestamp_ns__", "int64"),
        ])
        ts_meta = _encode_metadata(
            dataset_meta = dataset_meta_dict,
            topic = topic,
            msgtype = "Timestamp",
            msgdef_text = "A timestamp corresponding to one FFmpeg data",
            field_types = ts_field_types,
        )
        ts_pa_schema = pa.schema([
            pa.field("__timestamp_ns__", pa.int64()),
            pa.field("frame_index", pa.int64()),
        ], metadata=ts_meta)

        ts_table = pa.table({
            "__timestamp_ns__": pa.array(
                np.array(ts_list, dtype=np.int64)
            ),
            "frame_index":      pa.array(
                np.arange(len(ts_list), dtype=np.int64)
            ),
        }, schema=ts_pa_schema)
        pq.write_table(ts_table, ts_path)
        print(
            f"        sidecar: {os.path.basename(ts_path)} ({len(ts_list)} entries)"
        )

    t_video_end = time.monotonic()
    print(f"[Duration] Finalize videos + timestamps: {t_video_end - t_video_start:.2f}s")

    # ── 7. Update parquet metadata with dataset_meta ──────────────────
    # Data is already streamed to disk during step 3.
    # Only the hand-input topic parquet (if any) still needs writing.
    for topic in sorted(parquet_rows.keys()):
        rows = parquet_rows[topic]
        if not rows:
            continue

        msgtype = topic_msgtype.get(topic, "unknown")
        msgdef_text = topic_msgdef.get(topic, "")

        pa_schema = build_pa_schema(
            topic, msgtype, msgdef_text, dataset_meta_dict,
            spec=topic_spec.get(topic), first_row=rows[0],
        )
        table = rows_to_table(rows, pa_schema)

        fname = sanitize_topic_name(topic) + ".parquet"
        out_path = os.path.join(output_dir, fname)
        pq.write_table(table, out_path)
        print(f"[Parquet] {topic}: {len(rows)} msgs -> {fname}  ({msgtype})")

    # ── 8. Merge all parquets into data.parquet ──────────────────────
    print()
    t_merge_start = time.monotonic()
    _merge_parquets(output_dir, dataset_meta_dict)
    t_merge_end = time.monotonic()

    t_total = t_merge_end - t_program_start
    print(f"\n[Duration] Merge parquets:    {t_merge_end - t_merge_start:.2f}s")

    # ── Timeline Summary ─────────────────────────────────────────────
    print(f"\n[Timeline Summary]")
    print(f"{'Phase':<35} {'Duration':>8}  {'Cumulative':>9}")
    print(f"{'-'*35} {'-'*8}  {'-'*9}")

    def _row(label, duration, cumulative):
        print(f"{label:<35} {duration:>7.2f}s  {cumulative:>8.2f}s")

    t_classify_dur  = t_classify_end - t_classify_start
    t_open_dur      = t_open_end - t_open_start
    t_read_dur      = t_read_end - t_read_start
    t_ep_dur        = t_ep_end - t_ep_start
    t_video_dur     = t_video_end - t_video_start
    t_merge_dur     = t_merge_end - t_merge_start

    _row("Classify topics",          t_classify_dur, t_classify_end - t_program_start)
    _row("Open pipes",               t_open_dur,     t_open_end - t_program_start)
    _row("Read bag (single pass)",   t_read_dur,     t_read_end - t_program_start)
    _row("Detect episodes",          t_ep_dur,       t_ep_end - t_program_start)
    _row("Finalize videos + sidecar", t_video_dur,    t_video_end - t_program_start)
    _row("Merge parquets",           t_merge_dur,    t_merge_end - t_program_start)

    print(f"{'-'*35} {'-'*8}  {'-'*9}")
    _row("Total wall time",          t_total,        t_total)
    print()
    print("\nAll done.")


if __name__ == "__main__":
    main()
