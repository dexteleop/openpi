#!/usr/bin/env python3
"""
Build WebDataset .tar files from rosbag episodes.

Each (s,a) sample contains:
  - 000000.json (video paths + frame indices for on-the-fly decoding)
  - 000000.state_action.npz (62-dim state + action_seq_len x 62-dim action seq)

State and action share ONE 62-dim layout (the LeRobot v2 convention). Nothing
is selected, sliced or unit-converted here: every recorded column is written
raw and downstream transforms (e.g. TeleavatarInputs) pick what they need.
    [ 0:16]  positions   left_arm(7) left_gripper(1) right_arm(7) right_gripper(1)
    [16:32]  velocities  same order
    [32:48]  efforts     same order
    [48:55]  left  ee pose   position xyz(3) + orientation quaternion xyzw(4)
    [55:62]  right ee pose   position xyz(3) + orientation quaternion xyzw(4)

State reads the MEASURED topics (joint_states + current_ee_pose); action reads
the COMMANDED topics (joint_cmd + target_ee_pose). Arms are position
controlled and grippers are force controlled, so in the action vector the
gripper command lives in the effort block (indices 39 / 47) and its position
slots (7 / 15) are constantly 0 -- that asymmetry is preserved as recorded.

This script only PACKS raw vectors: it writes no norm_stats.json and computes
no mean/std/quantiles. Produce normalization stats separately
(scripts/compute_norm_stats.py) if the training pipeline needs them.

Usage:
    python generate_tars.py <root_dir> <output_dir> [fps] [action_seq_len]

    <root_dir> may be either a single dataset directory (containing
    data.parquet) or a parent directory holding many such datasets as
    sub-directories; all discovered datasets are merged into one tar corpus.

    python generate_tars.py /DATA/disk1/haoran/infra_dataset /DATA/disk1/haoran/infra_wds 30 30
"""

import os
import sys
import json
import numpy as np
from io import BytesIO
import tarfile
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from .generate_episode import generate_episodes_idxs
except :
    from openpi.training.lingyu_dataloader.generate_episode import generate_episodes_idxs

SAMPLES_PER_SHARD = 100
VIDEO_TOPICS = {
    "xr": "/xr_video_topic/ffmpeg",
    "left": "/left/color/image_raw/ffmpeg",
    "right": "/right/color/image_raw/ffmpeg",
}

# --- 62-dim vector layout ------------------------------------------------
# Both state and action use the same layout, so one builder serves both; only
# the source topics differ (measured vs commanded). Written raw: no dim
# selection, no unit conversion.
#   [ 0:16] positions   [16:32] velocities   [32:48] efforts   [48:62] ee poses
#
# ONE list per vector, in layout order. Whether a topic is a JointState (which
# contributes one slot to each of the three field blocks) or a Pose (one 7-dim
# block at the end) is read from the parquet's msg_type metadata, not encoded by
# splitting the list -- see load_topic_kinds.
# The three arrays sensor_msgs/JointState actually defines. There is no
# acceleration array in that message type, so the recording contains none and
# the vector has none; the third block is effort (torque, Nm).
JOINT_FIELDS = ["position", "velocity", "effort"]
POSE_COLUMNS = [
    "position.x", "position.y", "position.z",
    "orientation.x", "orientation.y", "orientation.z", "orientation.w",
]
# A JointState topic of N joints contributes N dims to EACH of the three field
# blocks (3N dims total, spread across the vector), not N dims in one place. A
# Pose topic contributes one contiguous 7-dim block. Hence 16 joints * 3 fields
# + 2 poses * 7 = 62. See vector_dim().
#
# Measured topics -> state vector.
STATE_TOPICS = [
    "/left_arm/joint_states",       # JointState,  7 joints -> 7 dims per field
    "/left_gripper/joint_states",   # JointState,  1 joint  -> 1 dim  per field
    "/right_arm/joint_states",      # JointState,  7 joints -> 7 dims per field
    "/right_gripper/joint_states",  # JointState,  1 joint  -> 1 dim  per field
    "/left_arm/current_ee_pose",    # Pose,        7-dim block
    "/right_arm/current_ee_pose",   # Pose,        7-dim block
]
# Commanded topics -> action vector. Same order and shapes as STATE_TOPICS, so
# both vectors are 62-dim and index-compatible.
ACTION_TOPICS = [
    "/left_arm/joint_cmd",          # JointState,  7 joints -> 7 dims per field
    "/left_gripper/joint_cmd",      # JointState,  1 joint  -> 1 dim  per field
    "/right_arm/joint_cmd",         # JointState,  7 joints -> 7 dims per field
    "/right_gripper/joint_cmd",     # JointState,  1 joint  -> 1 dim  per field
    "/left_arm/target_ee_pose",     # Pose,        7-dim block
    "/right_arm/target_ee_pose",    # Pose,        7-dim block
]

# msg_type (as recorded in the parquet metadata) -> how the topic is read.
KIND_BY_MSG_TYPE = {
    "sensor_msgs/msg/JointState": "joint",
    "geometry_msgs/msg/Pose": "pose",
}


def load_topic_kinds(dataset_dir):
    """Return {topic: "joint" | "pose"} for every topic in STATE/ACTION_TOPICS.

    The kind comes from the msg_type the recorder stored in the parquet
    metadata, so the topic lists above stay flat: adding a topic means adding
    one name, not also picking which list it belongs in.
    """
    filepath = os.path.join(dataset_dir, "data.parquet")
    meta = pq.ParquetFile(filepath).schema_arrow.metadata
    topics_meta = json.loads(meta[b"topics"]) if b"topics" in meta else {}
    kinds = {}
    for topic in STATE_TOPICS + ACTION_TOPICS:
        if topic not in topics_meta:
            raise KeyError(f"{topic} not present in {filepath}")
        msg_type = topics_meta[topic]["msg_type"]
        if msg_type not in KIND_BY_MSG_TYPE:
            raise ValueError(f"{topic}: unsupported msg_type {msg_type}")
        kinds[topic] = KIND_BY_MSG_TYPE[msg_type]
    return kinds


def preload_joint_topic(dataset_dir, topic_name, topic_to_rg):
    """Load a JointState topic as {field: list of per-row float32 vectors}.

    All three fields (position / velocity / effort) are read in one pass, since
    the 62-dim vector needs every one of them. Values are stored exactly as
    recorded -- no unit conversion.
    """
    filepath = os.path.join(dataset_dir, "data.parquet")
    pf = pq.ParquetFile(filepath)
    # topic_to_rg maps a topic to a LIST of row-group indices (a topic may span
    # several groups when it exceeds pyarrow's per-group row limit), so use the
    # plural read_row_groups which accepts a list, not read_row_group.
    table = pf.read_row_groups(topic_to_rg[topic_name], columns=JOINT_FIELDS)
    return {
        field: [np.array(row, dtype=np.float32)
                for row in table.column(field).to_pylist()]
        for field in JOINT_FIELDS
    }


def preload_pose_topic(dataset_dir, topic_name, topic_to_rg):
    """Load a Pose topic as a list of per-row 7-dim float32 vectors.

    Layout per row: position xyz (3) + orientation quaternion xyzw (4), i.e.
    POSE_COLUMNS in order. Pose messages are flat scalar columns in the
    parquet file, not list columns, so they are stacked here.
    """
    filepath = os.path.join(dataset_dir, "data.parquet")
    pf = pq.ParquetFile(filepath)
    table = pf.read_row_groups(topic_to_rg[topic_name], columns=POSE_COLUMNS)
    stacked = np.stack(
        [table.column(col).to_numpy() for col in POSE_COLUMNS], axis=1,
    ).astype(np.float32)
    return list(stacked)


def preload_topics(dataset_dir, topics, kinds, topic_to_rg):
    """Preload every topic in `topics`, dispatching on its kind.

    Returns one dict keyed by topic; a "joint" entry holds {field: rows} and a
    "pose" entry holds a list of 7-dim rows. Duplicates are loaded once.
    """
    data = {}
    for topic in topics:
        if topic in data:
            continue
        if kinds[topic] == "joint":
            data[topic] = preload_joint_topic(dataset_dir, topic, topic_to_rg)
        else:
            data[topic] = preload_pose_topic(dataset_dir, topic, topic_to_rg)
    return data


def topic_dims(topics, kinds, data):
    """Return {topic: dim} using the dims actually present in the data."""
    dims = {}
    for topic in topics:
        if kinds[topic] == "joint":
            dims[topic] = data[topic]["position"][0].shape[0]
        else:
            dims[topic] = len(POSE_COLUMNS)
    return dims


def vector_dim(topics, kinds, dims):
    """Total width of the vector built from `topics`.

    Each JointState topic contributes its dim once per field in JOINT_FIELDS;
    each Pose topic contributes one 7-dim block.
    """
    return sum(dims[t] * (len(JOINT_FIELDS) if kinds[t] == "joint" else 1)
               for t in topics)


def describe_vector_layout(topics, kinds, dims):
    """Return one "[lo:hi] source" line per slice of the 62-dim vector.

    Derived from the topic list and the dims actually seen in the data, so the
    printed layout always matches what is concatenated at runtime.
    """
    lines = []
    offset = 0
    for field in JOINT_FIELDS:
        for topic in topics:
            if kinds[topic] != "joint":
                continue
            dim = dims[topic]
            lines.append(f"[{offset:2d}:{offset + dim:2d}]  {topic}.{field}")
            offset += dim
    for topic in topics:
        if kinds[topic] != "pose":
            continue
        dim = dims[topic]
        lines.append(f"[{offset:2d}:{offset + dim:2d}]  {topic}.pose")
        offset += dim
    return lines


def build_action_sequence(frame_idx, frames, action_seq_len, kinds, data,
                          action_dim):
    """Build the (action_seq_len, action_dim) action sequence for one sample.

    Step k reads frame_idx + k, i.e. the commanded action k steps into the
    future. Two cases make a step unavailable: the index runs past the episode
    end, or the frame is missing one of the ACTION topics (build_state returns
    None). Both fall back to hold-last padding -- repeat the most recent valid
    action -- so the sequence stays physically continuous instead of jumping to
    zero. Only when step 0 itself is unavailable is there nothing to hold, and
    the zeros remain.

    Kept as its own function so the selection/padding decisions stay isolated
    from the packing loop.
    """
    last_action = np.zeros(action_dim, dtype=np.float32)
    action_list = []
    for k in range(action_seq_len):
        fidx = frame_idx + k
        if fidx < len(frames):
            step_vec = build_state(frames[fidx], ACTION_TOPICS, kinds, data)
            if step_vec is not None:
                last_action = step_vec
        action_list.append(last_action)
    return action_list


def build_state(frame, topics, kinds, data):
    """Assemble one 62-dim state vector for a frame, or None if any topic is missing.

    Concatenation order is [all positions, all velocities, all efforts, poses],
    matching describe_vector_layout and the LeRobot v2 convention: the joint
    fields sweep `topics` once per field, then the Pose topics append their
    7-dim blocks. Returning None (instead of a partially filled vector) keeps
    the caller's "skip incomplete frames" behaviour explicit.
    """
    blocks = []
    for field in JOINT_FIELDS:
        for topic in topics:
            if kinds[topic] != "joint":
                continue
            row_idx = frame.get(topic)
            rows = data[topic][field]
            if row_idx is None or row_idx >= len(rows):
                return None
            blocks.append(rows[row_idx])
    for topic in topics:
        if kinds[topic] != "pose":
            continue
        row_idx = frame.get(topic)
        rows = data[topic]
        if row_idx is None or row_idx >= len(rows):
            return None
        blocks.append(rows[row_idx])
    return np.concatenate(blocks).astype(np.float32)


def discover_datasets(root_dir):
    """Discover dataset directories under root_dir without hard-coding names.

    A directory is treated as one dataset if it directly contains a
    data.parquet file. If root_dir itself contains data.parquet, it is
    treated as a single dataset (backward compatible with the old usage).

    Returns:
        Sorted list of dataset directory paths. Order carries no meaning;
        sorting only makes shard numbering reproducible across runs.
    """
    if os.path.exists(os.path.join(root_dir, "data.parquet")):
        return [root_dir]

    datasets = []
    for entry in sorted(os.scandir(root_dir), key=lambda e: e.name):
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, "data.parquet")):
            datasets.append(entry.path)
    return datasets


# Written next to the shards, alongside sample_index.json. The loader reads it
# at worker startup to build every VideoDecoder up front instead of paying the
# build cost mid-training.
VIDEO_INDEX_FILENAME = "video_index.json"


def mp4_path_for(dataset_dir, topic_name):
    """Recorded topic -> the mp4 the extractor wrote for it.

    Same sanitization as the recorder and as the loader's get_mp4_path: strip
    the leading slash and turn the remaining slashes into underscores. Kept
    here so the index can be built without importing the loader.
    """
    sanitized = topic_name.strip("/").replace("/", "_")
    return os.path.join(dataset_dir, sanitized + ".mp4")


def build_video_index(datasets):
    """Map every (dataset_dir, topic) pair to the mp4 that holds it.

    The key is "<dataset_dir>|<topic>" rather than the bare filename on
    purpose: every episode names its files identically
    (left_color_image_raw_ffmpeg.mp4 and friends), so keying by name alone
    would collapse all episodes onto three entries and lose all but one.

    Only files that exist on disk are recorded, so a camera that was never
    recorded for an episode simply has no entry and the loader falls back to
    its zero-fill path.
    """
    video_index = {}
    missing = 0
    for dataset_dir in datasets:
        for topic in VIDEO_TOPICS.values():
            mp4_path = mp4_path_for(dataset_dir, topic)
            if os.path.exists(mp4_path):
                video_index[f"{dataset_dir}|{topic}"] = mp4_path
            else:
                missing += 1
    return video_index, missing


def process_one_dataset(dataset_dir, output_dir, fps, action_seq_len,
                        global_idx, tar, sample_index):
    """Process a single dataset directory, appending samples to shared shards.

    The tar sample layout is unchanged: each sample still emits a
    {key}.json (video paths + frame indices) and a {key}.state_action.npz.

    Args:
        global_idx: running sample counter shared across all datasets, so
            shard names stay unique and continuous.
        tar: currently-open tarfile handle (or None), kept open across
            datasets so shards are filled to SAMPLES_PER_SHARD regardless of
            dataset boundaries.
        sample_index: dict mapping sample_id to tar location info.

    Returns:
        (global_idx, tar): updated sample counter and current tar handle.
    """
    print(f"\n===== Dataset: {dataset_dir} =====")
    print("\nGenerating episode indices...")
    episodes, topic_to_rg = generate_episodes_idxs(dataset_dir, fps)

    print("\nPreloading joint and pose data...")
    kinds = load_topic_kinds(dataset_dir)
    data = preload_topics(dataset_dir, STATE_TOPICS + ACTION_TOPICS, kinds,
                          topic_to_rg)

    # Vector dims come from the data, not from a hardcoded constant.
    dims = topic_dims(STATE_TOPICS + ACTION_TOPICS, kinds, data)
    state_dim = vector_dim(STATE_TOPICS, kinds, dims)
    action_dim = vector_dim(ACTION_TOPICS, kinds, dims)
    print(f"\nState layout ({state_dim}-dim):")
    for line in describe_vector_layout(STATE_TOPICS, kinds, dims):
        print(f"  {line}")
    print(f"\nAction layout ({action_dim}-dim):")
    for line in describe_vector_layout(ACTION_TOPICS, kinds, dims):
        print(f"  {line}")

    print("\nBuilding tar files...")

    for eidx_str, frames in episodes.items():
        if len(frames) < fps:
            print(f"  Episode {eidx_str}: too short, skipping")
            continue

        print(f"\nProcessing episode {eidx_str} ({len(frames)} frames)...")

        total_frames = len(frames)
        for frame_idx, frame in enumerate(frames):
            # State: (1, state_dim) from the measured topics.
            state_vec = build_state(frame, STATE_TOPICS, kinds, data)
            if state_vec is None:
                continue
            state_arr = state_vec.reshape(1, state_dim)

            # Action: (action_seq_len, action_dim) from the commanded topics,
            # one row per future frame. See build_action_sequence for the
            # selection and hold-last padding rules.
            action_list = build_action_sequence(
                frame_idx, frames, action_seq_len, kinds, data, action_dim,
            )
            if not action_list:
                continue
            action_arr = np.array(action_list, dtype=np.float32)

            # Open new shard when needed
            if global_idx % SAMPLES_PER_SHARD == 0:
                if tar is not None:
                    tar.close()
                shard_idx = global_idx // SAMPLES_PER_SHARD
                tar_path = os.path.join(output_dir, f"teleavatar_{shard_idx:06d}.tar")
                tar = tarfile.open(tar_path, "w")

            sample_key = f"{global_idx:06d}"

            # Record sample index: sample_id -> tar location
            shard_idx = global_idx // SAMPLES_PER_SHARD
            sample_idx_in_shard = global_idx % SAMPLES_PER_SHARD
            sample_index[str(global_idx)] = {
                "tar_file": f"teleavatar_{shard_idx:06d}.tar",
                "shard_index": shard_idx,
                "sample_index_in_shard": sample_idx_in_shard
            }

            # Write video metadata as JSON (video paths + frame indices)
            video_meta = {"dataset_path": dataset_dir}
            for topic in VIDEO_TOPICS.values():
                if topic in frame:
                    video_meta[topic] = frame[topic]
            json_bytes = json.dumps(video_meta).encode("utf-8")
            info = tarfile.TarInfo(name=f"{sample_key}.json")
            info.size = len(json_bytes)
            tar.addfile(info, BytesIO(json_bytes))

            # Write state_action.npz
            buf = BytesIO()
            np.savez(buf, state=state_arr, action=action_arr)
            npz_bytes = buf.getvalue()
            info = tarfile.TarInfo(name=f"{sample_key}.state_action.npz")
            info.size = len(npz_bytes)
            tar.addfile(info, BytesIO(npz_bytes))

            global_idx += 1
            print(f"  Episode {eidx_str}: frame {frame_idx+1}/{total_frames} "
                   f"({(frame_idx+1)/total_frames*100:.1f}%)", end="\r")

    return global_idx, tar


def generate_tars(root_dir, output_dir, fps, action_seq_len=30):
    """Traverse every dataset under root_dir and write a single tar corpus.

    All datasets share one continuous shard numbering and one open tar handle,
    so the output is a flat set of teleavatar_XXXXXX.tar shards. Each sample
    carries its own dataset_path, so shards may freely mix samples from
    different datasets.

    No normalization stats are computed here: this script only packs raw
    vectors. Generate norm_stats.json separately
    (scripts/compute_norm_stats.py) if the training pipeline needs it.

    Two index files are generated in output_dir:
      - sample_index.json: sample_id -> its location in the tar shards.
      - video_index.json:  "<dataset_dir>|<topic>" -> mp4 path, so the loader
        can build every video decoder at worker startup.
    """
    datasets = discover_datasets(root_dir)
    if not datasets:
        print(f"Error: no dataset (data.parquet) found under '{root_dir}'")
        sys.exit(1)

    print(f"Discovered {len(datasets)} dataset(s):")
    for d in datasets:
        print(f"  - {d}")

    # Shared state across all datasets
    global_idx = 0
    tar = None
    sample_index = {}  # sample_id -> tar location mapping

    for dataset_dir in datasets:
        global_idx, tar = process_one_dataset(
            dataset_dir, output_dir, fps, action_seq_len, global_idx, tar, sample_index,
        )

    if tar is not None:
        tar.close()

    # Save sample index to JSON file
    index_path = os.path.join(output_dir, "sample_index.json")
    with open(index_path, "w") as f:
        json.dump(sample_index, f, indent=2)
    print(f"\nSample index saved to: {index_path}")
    print(f"Total samples indexed: {len(sample_index)}")

    # Save video index: every mp4 the loader may have to decode from.
    video_index, missing = build_video_index(datasets)
    video_index_path = os.path.join(output_dir, VIDEO_INDEX_FILENAME)
    with open(video_index_path, "w") as f:
        json.dump(video_index, f, indent=2)
    print(f"Video index saved to: {video_index_path}")
    print(f"Total videos indexed: {len(video_index)}"
          + (f" ({missing} topic(s) had no mp4 on disk)" if missing else ""))

    print(f"\nDone! Total: {global_idx} samples from {len(datasets)} dataset(s)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <root_dir> <output_dir> [fps] [action_seq_len]")
        sys.exit(1)

    root_dir = sys.argv[1]
    output_dir = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    action_seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    generate_tars(root_dir, output_dir, fps, action_seq_len)