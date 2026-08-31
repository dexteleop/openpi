#!/usr/bin/env python3
"""
Generate (state, action) pairs from a data.parquet dataset at a fixed fps.

For each episode, a time grid is created at fps intervals (e.g. 1/30s).
At each grid point, the closest observation from every topic is found,
producing a synchronized (s, a) snapshot.

All internal time calculations use seconds. Timestamps are converted from
nanoseconds to seconds when read from parquet files.

Usage:
    python generate_episode.py <dataset_dir> [fps]
    python generate_episode.py /mnt/data6t/shihaoran_data/test_lerobot3 30
"""

import os
import sys
import json
import numpy as np

import pyarrow.parquet as pq

# State-action time offset in seconds (originally 1e9 // 90 nanoseconds)
SA_TIME_OFFSET_S = 1.0 / 90.0

# Staleness bound for a tick's nearest observation, in units of the frame
# interval, or None to disable.  `searchsorted` always returns the last message
# at or before the reference instant, however old it is, so without a bound a
# topic that stopped publishing would keep contributing its final value to every
# later frame.
#
# NOTE: the online recorder has no such per-topic bound -- its _cur_state_msg
# cache carries a stale value forward indefinitely.  Set this to None for
# bit-exact online semantics; 1.0 additionally drops frames whose topic went
# silent across the tick.  Measured worst case on MK_tower_floor2 (3 bags,
# 47757 frames, 15 topics) is 0.865 frame intervals, so 1.0 drops nothing there.
MAX_STALENESS_FRAMES = 1.0
STATE_KEYWORDS = [
    '/left_arm/joint_states', '/left_arm/current_ee_pose', '/left_gripper/joint_states',
    '/right_arm/joint_states','/right_arm/current_ee_pose','/right_gripper/joint_states',
    '/chassis/joint_states', '/kinco/actual_velocity',
    'ffmpeg',
]
ACTION_KEYWORDS = [
    '/left_arm/joint_cmd', '/left_arm/target_ee_pose', '/left_gripper/joint_cmd',
    '/right_arm/joint_cmd','/right_arm/target_ee_pose','/right_gripper/joint_cmd',
    '/chassis/joint_cmd', '/kinco/cmd_velocity',
]
# Topics that advance the tick clock without contributing a column.
#
# The online recorder's `if is_recording:` block (convert:1472) sits at the same
# level as its Joy handler, so EVERY message it reads can close a tick window --
# including the hand-input topic it only uses for X/Y detection.  Leaving these
# out of the arrival timeline shifts ~14% of tick reference instants, so they are
# classified separately: they drive the clock, they never become data.
CLOCK_KEYWORDS = [
    'hand_inputs',
]


def classify_topic(topic):
    for kw in ACTION_KEYWORDS:
        if kw in topic:
            return "action"
    for kw in STATE_KEYWORDS:
        if kw in topic:
            return "state"
    for kw in CLOCK_KEYWORDS:
        if kw in topic:
            return "clock"
    return "other"


def load_dataset_meta(dataset_dir, fps=30):
    # Time interval in seconds (1/fps)
    interval_s = 1.0 / fps

    filepath = os.path.join(dataset_dir, "data.parquet")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        sys.exit(1)
    meta = pq.ParquetFile(filepath).schema_arrow.metadata

    topics_meta = json.loads(meta[b"topics"]) if b"topics" in meta else {}

    dataset_meta = json.loads(meta[b"dataset_meta"]) if b"dataset_meta" in meta else {}
    episodes_meta = dataset_meta.get("episodes", {})

    # Build topic -> row-group-index map from the ACTUAL "topic" column of each
    # row group, not from the metadata dict order.  A single topic may span
    # several row groups (pyarrow splits any group past 1048576 rows), so a
    # topic maps to a LIST of row groups; relying on the enumeration index would
    # shift every topic after such a split by one and read the wrong data.
    pf = pq.ParquetFile(filepath)
    topic_to_rg = {}
    for j in range(pf.num_row_groups):
        t = pf.read_row_group(j, columns=["topic"]).column("topic")[0].as_py()
        topic_to_rg.setdefault(t, []).append(j)

    topics_role = {}
    for topic_name in topics_meta.keys():
        topics_role[topic_name] = classify_topic(topic_name)

    return filepath, topics_role, topic_to_rg, episodes_meta, interval_s


def _search_closest_le(ts_arr: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """For each target, find index of the closest timestamp <= target (ts_arr must be sorted)."""
    indices = np.searchsorted(ts_arr, targets, side="right") - 1
    indices[indices < 0] = -1
    return indices


def _episode_ticks(start_ts, end_ts, arrival_ts, interval_s):
    """Replay the online recorder's tick state machine over one episode.

    Mirrors convert_rosbag_to_lerobot.py:1472-1517.  The recorder cannot sample
    at a grid point directly -- it only learns the grid point has passed when the
    next message of ANY subscribed topic arrives, and samples at that instant.
    So the reference time for a tick is not the grid point but the first arrival
    at or after it, and the grid then re-anchors from that arrival.

    Args:
        arrival_ts: sorted arrival times of every subscribed topic, in seconds.

    Returns:
        (state_ref, action_ref, tick_idx) parallel arrays.  state_ref[i] is the
        instant `_materialize_state` runs for tick tick_idx[i]; action_ref[i] is
        the instant `add_action` runs.  Ticks whose state window stayed silent
        are absent, exactly as the recorder skips them.
    """
    state_target = start_ts + interval_s
    action_target = state_target + SA_TIME_OFFSET_S
    k = 1
    in_adding = False
    s_ref = None
    state_ref, action_ref, tick_idx = [], [], []

    lo = int(np.searchsorted(arrival_ts, state_target, side="left"))
    for t in arrival_ts[lo:]:
        if t < state_target:
            continue
        if t <= action_target:
            if not in_adding:
                # State window: sample state here (convert:1476-1482).
                s_ref = t
                in_adding = True
        elif in_adding:
            # Action window closed by this arrival: sample action, emit the row,
            # then skip whole intervals the arrival already ran past.
            state_ref.append(s_ref)
            action_ref.append(t)
            tick_idx.append(k)
            skipped = int((t - action_target) // interval_s)
            state_target += (skipped + 1) * interval_s
            k += skipped + 1
            action_target = state_target + SA_TIME_OFFSET_S
            in_adding = False
        else:
            # Nothing arrived inside the state window: the tick is dropped.
            skipped = int((t - action_target) // interval_s) + 1
            state_target += skipped * interval_s
            k += skipped
            action_target = state_target + SA_TIME_OFFSET_S

    return (np.array(state_ref, dtype=np.float64),
            np.array(action_ref, dtype=np.float64),
            np.array(tick_idx, dtype=np.int64))


def generate_episode_snapshots(filepath, topics_role, topic_to_rg, episodes_meta,
                               interval_s):
    """Build per-episode frames of topic -> row index, aligned to the recorder.

    Frames are produced on the tick sequence the online recorder would have
    produced (see _episode_ticks), and each topic contributes the newest message
    at or before that tick's reference instant -- the same value its
    _cur_state_msg / _cur_action_msg cache would have held.
    """
    pf = pq.ParquetFile(filepath)

    # Pass 1: the tick reference instants depend on when messages ARRIVE across
    # all topics, so the arrival timeline has to be known before any frame can be
    # placed.  "clock" topics count here even though they contribute no column.
    # Only the timestamp column is read, and only once per topic.
    sampled = [t for t, role in topics_role.items() if role != "other"]
    print(f"Building arrival timeline from {len(sampled)} topic(s) "
          f"({sum(1 for t in sampled if topics_role[t] == 'clock')} clock-only) ...")
    topic_ts = {}
    for topic_name in sampled:
        ts_ns = pf.read_row_groups(
            topic_to_rg[topic_name], columns=["__timestamp_ns__"],
        ).column("__timestamp_ns__").to_numpy()
        topic_ts[topic_name] = ts_ns / 1e9
    arrival_ts = (np.sort(np.concatenate([v for v in topic_ts.values() if len(v)]))
                  if topic_ts else np.zeros(0, dtype=np.float64))

    # Initialize empty frame lists and per-tick reference times for all episodes
    episode_frames = {}
    episode_targets = {}
    # Sort by numeric episode index so [episode x] / [ep x] logs (and the
    # frame-building order that flows from here) follow 0,1,2,...,10 instead
    # of lexicographic string order ("0","1","10","2",...).
    for eidx_str, ep_info in sorted(episodes_meta.items(), key=lambda kv: int(kv[0])):
        # Convert timestamps from nanoseconds to seconds
        start_ts = ep_info["start"] / 1e9
        end_ts = ep_info["end"] / 1e9

        # Restrict to arrivals the recorder would have seen while recording.
        lo = int(np.searchsorted(arrival_ts, start_ts, side="left"))
        hi = int(np.searchsorted(arrival_ts, end_ts, side="left"))
        state_arr, action_arr, ticks = _episode_ticks(
            start_ts, end_ts, arrival_ts[lo:hi], interval_s,
        )

        # The recorder stops at the Y press, so a tick whose action reference
        # would fall past the episode end never produces a row.
        keep = action_arr <= end_ts
        state_arr, action_arr, ticks = state_arr[keep], action_arr[keep], ticks[keep]

        n_frames = len(state_arr)
        print(f"[episode {eidx_str}] start={start_ts:.6f}s, end={end_ts:.6f}s, "
              f"{n_frames} tick(s), last tick index={ticks[-1] if n_frames else 0}")
        episode_frames[eidx_str] = [{} for _ in range(n_frames)]
        episode_targets[eidx_str] = (state_arr, action_arr, n_frames,
                                     start_ts, end_ts)

    print()

    # Pass 2: for every tick, take each topic's newest message at or before that
    # tick's reference instant -- the value the recorder's cache would hold.
    for topic_name, role in topics_role.items():
        print(f"[topic {topic_name}]", end='')
        if role == "other":
            print("  [skip]")
            continue
        if role == "clock":
            # Already folded into the arrival timeline; contributes no column.
            print("  [clock only]")
            continue

        ts = topic_ts[topic_name]   # already read in pass 1

        # Inner loop: iterate over episodes (reuse the already-loaded timestamp array)
        for eidx_str in episode_targets.keys():
            state_arr, action_arr, n_frames, start_ts, end_ts = episode_targets[eidx_str]
            if not n_frames or not len(ts):
                continue
            # No episode mask here: the recorder's _cur_state_msg cache is never
            # cleared at the X press, so a message from before the episode is a
            # legitimate source. MAX_STALENESS_FRAMES below is what bounds it.
            targets = state_arr if role == "state" else action_arr
            indices = _search_closest_le(ts, targets)

            # Reject a match older than the staleness bound: the topic was silent
            # across this tick, so the frame must not be built from a value
            # carried over from long before. Dropping the topic makes the frame
            # incomplete and the dedup pass below removes it.
            if MAX_STALENESS_FRAMES is not None:
                fresh = indices >= 0
                if fresh.any():
                    age = targets[fresh] - ts[indices[fresh]]
                    too_old = age > MAX_STALENESS_FRAMES * interval_s
                    if too_old.any():
                        indices[np.flatnonzero(fresh)[too_old]] = -1
                        print(f"  [ep {eidx_str}] {int(too_old.sum())} stale", end='')

            for i in range(n_frames):
                if indices[i] >= 0:
                    # topic-local index: the 0-based position of this data in the topic's full array
                    episode_frames[eidx_str][i][topic_name] = int(indices[i])

            matched = np.sum(indices >= 0)
            print(f"  [ep {eidx_str}] matched {matched}/{n_frames}", end='')

        print()

    # dedup: keep frame only if the frame contains every expected topic
    # ("clock" topics drive the tick clock but never populate a frame)
    expected_topics = [name for name, role in topics_role.items()
                       if role not in ("other", "clock")]
    expected_count = len(expected_topics)
    episodes = {}
    # Numeric sort keeps the final episodes dict (and thus the tar-building
    # order in generate_tars.py) in 0,1,2,...,10 order.
    for eidx_str in sorted(episode_frames.keys(), key=int):
        frames = episode_frames[eidx_str]
        deduped = []
        for frame in frames:
            cur_indices = {name: frame.get(name) for name in expected_topics if name in frame}
            is_complete = len(cur_indices) == expected_count
            if is_complete:
                deduped.append(frame)
        episodes[eidx_str] = deduped
    return episodes


def generate_episodes_idxs(dataset_dir, fps):
    print("Loading data.parquet's metadata ...")
    filepath, topics_role, topic_to_rg, episodes_meta, interval_s = load_dataset_meta(
        dataset_dir, fps
    )
    print("\nGenerating episode snapshots ...")
    episodes = generate_episode_snapshots(
        filepath, topics_role, topic_to_rg, episodes_meta, interval_s
    )
    return episodes, topic_to_rg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <dataset_dir> [fps]")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # default fps is 30

    if not os.path.isdir(dataset_dir):
        print(f"Error: '{dataset_dir}' is not a directory")
        sys.exit(1)

    episodes, topic_to_rg = generate_episodes_idxs(dataset_dir, fps)
    
    print("\nShowing episodes ...")
    for eidx, episode in episodes.items():
        print(f"Episode {eidx}: {len(episode)} frames")
        print(f'episodes[{eidx}][0]:', episode[0])
        print(f'episodes[{eidx}][-1]:', episode[-1])