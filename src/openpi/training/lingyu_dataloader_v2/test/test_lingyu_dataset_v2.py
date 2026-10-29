"""Verify LingyuDatasetV2's len and sampling results, cross-checked against the global index.

This test is read-only, so collection and numbering must have finished first (when data is already collected, just rebuild the index with --index-only)::
    python -m openpi.training.lingyu_dataloader_v2.build_sample_idx [--index-only]

Checks::
    1. len(dataset) == the sum of num_samples in episode_index.parquet
    2. locate_sample agrees exactly with a direct DuckDB lookup of (prompt, episode_id, sample_idx)
    3. 5 random samples can all be fetched and are single-frame structures without a leading frame dimension:
       images are (H, W, 3) uint8, state is (state_dim,), action is (chunk, action_dim)
    4. state/action dims == the sum of field lengths in their concat configs, i.e. the concat order is really used
    5. Two adjacent samples within one episode differ (what was fetched is not the same frame/message)
Each random sample is saved to lingyu_dataset_v2_output/sample_<global number>/: each camera as png, state/action as one json
Sampling time splits into three parts: reading the sample's locations index by (data_file, row_group) /
reading raw mcap bytes from S3 in parallel threads and deserializing (whole wall-clock span) / video decoding + field extraction + assembling into a sample
"""
import gc
import json
import random
import shutil
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pytest
import torch
from PIL import Image

import openpi.training.config as _config
from openpi.training.lingyu_dataloader_v2 import lingyu_dataset_v2
from openpi.training.lingyu_dataloader_v2.build_sample_idx import GLOBAL_INDEX_NAME, WAREHOUSE_DIR
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import (
    LingyuDatasetV2, filter_sample_messages, load_sample_topics, load_video_decode_config)
from openpi.training.lingyu_dataloader_v2.mcap_config.config import (
    load_mcap_state_and_action_topics_fields, load_mcap_video_topics_gop)
from openpi.training.lingyu_dataloader_v2.model_config.config import load_action_chunk_length
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
from openpi.training.lingyu_dataloader_v2.utils import _mcap_utils, mcap_message_fetcher

logger = get_logger(__file__)

_NUM_RANDOM_SAMPLES = 5  # Number of samples randomly spot-checked
_NUM_MEMORY_SAMPLES = 3  # Samples whose memory test_span_fetch_memory measures (plus 1 warm-up sample); deep call-stack snapshots are slow, about 7 minutes per sample
_TRACE_FRAMES = 64  # Call-stack depth tracemalloc records per memory block
_OUTPUT_DIR = Path(__file__).resolve().parent / "lingyu_dataset_v2_output"
_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"  # GPU crop/scale parameters come from this training config

# The original functions wrapped by the timing wrappers; swapped for the _timed_* versions below during sampling, accumulating into _elapsed_seconds
_ORIGINAL_FETCH_SAMPLE_MESSAGES = lingyu_dataset_v2.fetch_sample_messages
_ORIGINAL_LOAD_SAMPLE_TOPICS = lingyu_dataset_v2.load_sample_topics
_elapsed_seconds = {"s3": 0.0, "index": 0.0}
# The original functions wrapped by the counting wrappers; swapped for _counted_fetch_mcap_bytes during test_span_fetch, accumulating into _fetch_stats
_ORIGINAL_FETCH_MCAP_BYTES = _mcap_utils.fetch_mcap_bytes
_fetch_stats = {"requests": 0, "bytes": 0}
_fetch_stats_lock = threading.Lock()


def test():
    """Randomly fetch 5 samples, verify number lookup and output structure, and save the samples to disk."""
    index_path = Path(WAREHOUSE_DIR, GLOBAL_INDEX_NAME)
    if not index_path.exists():
        pytest.skip(f"没有 {index_path}, 请先运行 build_sample_idx.py")

    # Like training, GPU crop/scale parameters come from the openpi training config
    train_config = _config.get_config(_CONFIG_NAME)
    video_key_to_crop, image_resolution = load_video_decode_config(
        train_config.data.create(train_config.assets_dirs, train_config.model))
    dataset = LingyuDatasetV2(video_key_to_crop=video_key_to_crop, image_resolution=image_resolution)
    logger.info(f"len(dataset) = {len(dataset)}, {len(dataset.episodes)} 个 episode")

    # --- 1. Total count matches the index ---
    total_samples = duckdb.sql(f"SELECT sum(num_samples) FROM '{index_path}'").fetchone()[0]
    assert len(dataset) == total_samples, "len(dataset) 与全局索引的 sample 总数不符"

    # Clear the previous run's output so the directory holds only this run's random samples
    shutil.rmtree(_OUTPUT_DIR, ignore_errors=True)
    random_samples_idx = sorted(random.sample(range(len(dataset)), _NUM_RANDOM_SAMPLES))
    logger.info(f"随机抽取的全局 sample 编号: {random_samples_idx}")
    for global_sample_idx in random_samples_idx:
        # --- 2. Lookups agree with DuckDB ---
        located = duckdb.sql(
            f"SELECT prompt, episode_id, {global_sample_idx} - sample_offset AS sample_idx "
            f"FROM '{index_path}' "
            f"WHERE sample_offset <= {global_sample_idx} "
            f"  AND {global_sample_idx} < sample_offset + num_samples").fetchall()
        assert len(located) == 1, f"全局 sample {global_sample_idx} 反查到 {len(located)} 个 episode"
        assert dataset.locate_sample(global_sample_idx) == located[0], \
            f"locate_sample({global_sample_idx}) 与 DuckDB 反查不一致"

        # --- 3. Sample and verify structure ---
        _elapsed_seconds.update(s3=0.0, index=0.0)
        start_time = time.perf_counter()
        with (patch.object(lingyu_dataset_v2, "fetch_sample_messages", _timed_fetch_sample_messages),
              patch.object(lingyu_dataset_v2, "load_sample_topics", _timed_load_sample_topics)):
            sample = dataset[global_sample_idx]
        elapsed = time.perf_counter() - start_time
        # Total time minus index reading and S3 reading leaves the cost of turning raw bytes into samples
        convert_elapsed = elapsed - _elapsed_seconds["index"] - _elapsed_seconds["s3"]

        images = sample["observation"]["images"]
        state, action = sample["observation"]["state"], sample["action"]
        assert images, "没有解码出任何图像"
        for camera_key, frame in images.items():
            assert frame.ndim == 3 and frame.shape[2] == 3, \
                f"{camera_key} 不是 (H, W, 3): {tuple(frame.shape)}"
            assert frame.dtype == np.uint8, f"{camera_key} 不是 uint8: {frame.dtype}"
        assert state.ndim == 1 and state.dtype == torch.float32, \
            f"state 不是 (state_dim,) float32: {tuple(state.shape)}"
        assert action.shape[0] == load_action_chunk_length(), \
            f"action 步数与 ACTION_CHUNK_LENGTH 不符: {tuple(action.shape)}"

        # --- 4. Dims == the sum of each concat config's field lengths ---
        for vector_name, vector, concat_config in (
            ("state", state, dataset.state_concat),("action", action, dataset.action_concat),):
            assert vector.shape[-1] == sum(
                # Each concat entry is that topic field's length; summing them gives the vector's total width
                len(part) for part in _concat_parts(dataset, global_sample_idx, concat_config)), \
                f"{vector_name} 维度与拼接配置不符: {tuple(vector.shape)}"

        logger.info(f"sample {global_sample_idx} -> {sample['prompt']!r}, "
                    f"{ {key: tuple(frame.shape) for key, frame in images.items()} }, "
                    f"state={tuple(state.shape)}, action={tuple(action.shape)}, "
                    f"总耗时 {elapsed:.3f}s = 读索引 {_elapsed_seconds['index']:.3f}s "
                    f"+ S3 读原始数据 {_elapsed_seconds['s3']:.3f}s + 转换为样本 {convert_elapsed:.3f}s")
        _save_sample(sample, global_sample_idx, located[0])

    # --- 5. Adjacent samples within one episode differ in content ---
    prompt, episode_id, sample_idx = dataset.locate_sample(0)
    assert dataset.locate_sample(1)[:2] == (prompt, episode_id), "前两个样本不在同一 episode, 无法比较"
    assert not torch.equal(dataset[0]["observation"]["state"], dataset[1]["observation"]["state"]), \
        "相邻样本的 state 完全相同, 取样可能一直落在同一条 message 上"


def test_span_fetch():
    """Messages fetched merged-by-chunk vs one-by-one are byte-identical; count both ways' S3 requests and downloaded bytes."""
    index_path = Path(WAREHOUSE_DIR, GLOBAL_INDEX_NAME)
    if not index_path.exists():
        pytest.skip(f"没有 {index_path}, 请先运行 build_sample_idx.py")

    dataset = LingyuDatasetV2(load_images=False)  # Used only to look up samples' physical locations, no images read
    used_topic_names = set(load_mcap_video_topics_gop()) | set(load_mcap_state_and_action_topics_fields())
    request_totals = {"span": [0, 0], "single": [0, 0]}
    for global_sample_idx in sorted(random.sample(range(len(dataset)), _NUM_RANDOM_SAMPLES)):
        episode = dataset.locate_episode(global_sample_idx)
        sample_topics = load_sample_topics(
            episode["data_file"], episode["row_group"], global_sample_idx - episode["sample_offset"])
        used_topics = {topic: message for topic, message in sample_topics.items()
                       if message is not None and topic in used_topic_names}

        _fetch_stats.update(requests=0, bytes=0)
        with _count_fetch_bytes(), ThreadPoolExecutor(max_workers=lingyu_dataset_v2.SAMPLE_FETCH_WORKERS) as pool:
            span_messages = lingyu_dataset_v2.fetch_sample_messages(used_topics, pool)
        span_stats = (_fetch_stats["requests"], _fetch_stats["bytes"])

        # Control group: the pre-change per-message read, one fetch_message per location
        _fetch_stats.update(requests=0, bytes=0)
        with _count_fetch_bytes():
            single_messages = {topic: [lingyu_dataset_v2.fetcher.fetch_message(
                message["msg_type"], message["msg_def"], [tuple(location.values())])[0]
                for location in message["locations"]] for topic, message in used_topics.items()}
        single_stats = (_fetch_stats["requests"], _fetch_stats["bytes"])

        # Re-serialize to CDR and compare byte by byte; identical bytes mean identical message content
        typestore = lingyu_dataset_v2.fetcher._typestore
        assert span_messages.keys() == single_messages.keys(), "两种读取方式的 topic 集合不同"
        for topic, message in used_topics.items():
            assert len(span_messages[topic]) == len(single_messages[topic]) == len(message["locations"]), \
                f"{topic} 的 message 条数与 locations 不符"
            for span_message, single_message in zip(span_messages[topic], single_messages[topic]):
                assert typestore.serialize_cdr(span_message, message["msg_type"]) == \
                       typestore.serialize_cdr(single_message, message["msg_type"]), f"{topic} 的 message 内容不一致"

        for mode_name, mode_stats in (("span", span_stats), ("single", single_stats)):
            request_totals[mode_name][0] += mode_stats[0]
            request_totals[mode_name][1] += mode_stats[1]
        logger.info(f"sample {global_sample_idx}: {sum(len(m['locations']) for m in used_topics.values())} 个 location, "
                    f"合并读取 {span_stats[0]} 次请求 / {span_stats[1] / 1024:.0f}KiB, "
                    f"逐条读取 {single_stats[0]} 次请求 / {single_stats[1] / 1024:.0f}KiB, message 逐字节一致")

    logger.info(f"{_NUM_RANDOM_SAMPLES} 个样本合计: 合并读取 {request_totals['span'][0]} 次 / "
                f"{request_totals['span'][1] / 1024 / 1024:.1f}MiB, 逐条读取 {request_totals['single'][0]} 次 / "
                f"{request_totals['single'][1] / 1024 / 1024:.1f}MiB")
    assert request_totals["span"][0] < request_totals["single"][0], "合并读取没有减少请求次数"


def test_span_fetch_memory():
    """The merged-read whole-segment bytes are released once fetch_sample_messages returns; the result holds only each message's own CDR copy.

    and compare against the sample's S3 download bytes. tracemalloc itself uses a lot of memory, so RSS is not looked at.
    Use tracemalloc to measure per-sample Python memory: peak (during reads) / still held by the result after return / residual after the result is released,
    """
    index_path = Path(WAREHOUSE_DIR, GLOBAL_INDEX_NAME)
    if not index_path.exists():
        pytest.skip(f"没有 {index_path}, 请先运行 build_sample_idx.py")

    dataset = LingyuDatasetV2(load_images=False)  # Used only to look up samples' physical locations, no images read
    state_and_action_topics = set(load_mcap_state_and_action_topics_fields())
    sample_idxs = random.sample(range(len(dataset)), _NUM_MEMORY_SAMPLES + 1)
    tracemalloc.start(_TRACE_FRAMES)  # Record deep enough call stacks to recognize blocks allocated via fetch_mcap_bytes
    for sample_order, global_sample_idx in enumerate(sample_idxs):
        episode = dataset.locate_episode(global_sample_idx)
        sample_topics = load_sample_topics(
            episode["data_file"], episode["row_group"], global_sample_idx - episode["sample_offset"])
        used_topics = {topic: message for topic, message in sample_topics.items()
                       if message is not None and topic in state_and_action_topics}

        _fetch_stats.update(requests=0, bytes=0)
        gc.collect()
        tracemalloc.reset_peak()
        traced_before = tracemalloc.get_traced_memory()[0]
        before_snapshot = tracemalloc.take_snapshot()
        with _count_fetch_bytes(), ThreadPoolExecutor(max_workers=lingyu_dataset_v2.SAMPLE_FETCH_WORKERS) as pool:
            topic_messages = lingyu_dataset_v2.fetch_sample_messages(used_topics, pool)
        traced_after, traced_peak = tracemalloc.get_traced_memory()
        result_snapshot = tracemalloc.take_snapshot()
        del topic_messages
        gc.collect()
        traced_released = tracemalloc.get_traced_memory()[0]
        if sample_order == 0:
            continue  # The first sample includes one-time costs of S3 client creation, typestore registration etc., not counted

        downloaded = _fetch_stats["bytes"]
        # While the result is alive, the surviving blocks newly allocated during this sample whose stack passes through _mcap_utils (i.e. fetched by fetch_mcap_bytes) = the unreleased whole-segment bytes
        download_filter = [tracemalloc.Filter(True, _mcap_utils.__file__, all_frames=True)]
        span_alive = sum(stat.size_diff for stat in result_snapshot.filter_traces(download_filter).compare_to(
            before_snapshot.filter_traces(download_filter), "filename"))
        logger.info(f"sample {global_sample_idx}: S3 下载 {downloaded / 1024:.0f}KiB({_fetch_stats['requests']} 次), "
                    f"读取期间峰值 +{(traced_peak - traced_before) / 1024:.0f}KiB, "
                    f"返回后结果共持有 {(traced_after - traced_before) / 1024:.0f}KiB(其中下载段仍存活 {span_alive / 1024:.1f}KiB), "
                    f"结果释放后残留 {(traced_released - traced_before) / 1024:.0f}KiB")
        assert span_alive < downloaded / 10, "返回结果仍引用着下载的整段字节, 整段字节未被释放"
    tracemalloc.stop()


@contextmanager
def _count_fetch_bytes():
    """Swap fetch_mcap_bytes in mcap_message_fetcher and _mcap_utils for counting versions, accumulating request counts and downloaded bytes."""
    with (patch.object(mcap_message_fetcher, "fetch_mcap_bytes", _counted_fetch_mcap_bytes),
          patch.object(_mcap_utils, "fetch_mcap_bytes", _counted_fetch_mcap_bytes)):
        yield


def _counted_fetch_mcap_bytes(mcap_url: str, length: int, offset: int) -> bytes:
    """Counting fetch_mcap_bytes: called concurrently from threads; counters accumulate under a lock."""
    data_bytes = _ORIGINAL_FETCH_MCAP_BYTES(mcap_url, length, offset)
    with _fetch_stats_lock:
        _fetch_stats["requests"] += 1
        _fetch_stats["bytes"] += len(data_bytes)
    return data_bytes


def _timed_fetch_sample_messages(sample_topics: dict, thread_pool) -> dict:
    """Timed fetch_sample_messages: the whole wall-clock span of reading one sample's all locations from S3 in parallel."""
    start_time = time.perf_counter()
    topic_messages = _ORIGINAL_FETCH_SAMPLE_MESSAGES(sample_topics, thread_pool)
    _elapsed_seconds["s3"] += time.perf_counter() - start_time
    return topic_messages


def _timed_load_sample_topics(data_file: str, row_group: int, sample_idx: int) -> dict[str, dict]:
    """Timed load_sample_topics: reading that sample's locations index by (data_file, row_group)."""
    start_time = time.perf_counter()
    sample_topics = _ORIGINAL_LOAD_SAMPLE_TOPICS(data_file, row_group, sample_idx)
    _elapsed_seconds["index"] += time.perf_counter() - start_time
    return sample_topics


def _save_sample(sample: dict, global_sample_idx: int, located: tuple):
    """Save one sample under sample_<global number>/: one png per camera, state/action together in one json."""
    sample_dir = _OUTPUT_DIR / f"sample_{global_sample_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for camera_key, frame in sample["observation"]["images"].items():
        Image.fromarray(frame).save(sample_dir / f"{camera_key}.png")  # Already (H, W, 3) uint8

    prompt, episode_id, sample_idx = located
    json_path = sample_dir / "state_action.json"
    json_path.write_text(json.dumps({
        "prompt": prompt, "episode_id": episode_id, "sample_idx": sample_idx,
        "state": sample["observation"]["state"].tolist(), "action": sample["action"].tolist(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"sample {global_sample_idx} 已保存到 {sample_dir}")


def _concat_parts(dataset, global_sample_idx, concat_config):
    """Extract the 1-D array for each entry of the sample's concat configs, for dimension checks."""
    episode = dataset.locate_episode(global_sample_idx)
    sample_topics = load_sample_topics(
        episode["data_file"], episode["row_group"], global_sample_idx - episode["sample_offset"])
    # Pick only topics used by concat, so verifying dims does not decode the three video streams again
    topic_data = filter_sample_messages(
        {topic: sample_topics[topic] for topic, _ in concat_config})
    return [topic_data[topic][0][field_name] for topic, field_name in concat_config]


if __name__ == "__main__":
    test()
