"""核对 LingyuDatasetV2 的 len 与取样结果, 并与全局索引交叉验证。

本测试只读, 故须先跑完采集与编号(数据已采集时只需 --index-only 重建索引)::
    python -m openpi.training.lingyu_dataloader_v2.build_sample_idx [--index-only]

核对::
    1. len(dataset) == episode_index.parquet 里 num_samples 之和
    2. locate_sample 与 DuckDB 直接反查的 (prompt, episode_id, sample_idx) 完全一致
    3. 随机 5 个样本都能取出, 且为不带前导帧维度的单帧结构:
       images 为 (H, W, 3) uint8, state 为 (state_dim,), action 为 (chunk, action_dim)
    4. state/action 的维度 == 各自拼接配置里字段长度之和, 即拼接顺序真的被用上了
    5. 同一 episode 内相邻的两个样本各不相同(取到的不是同一帧/同一条 message)
每个随机样本存到 lingyu_dataset_v2_output/sample_<全局编号>/: 各 camera 存为 png, state/action 存为 json
取样耗时拆为三段: 按 (data_file, row_group) 读 sample 的 locations 索引 /
多线程并行从 S3 读 mcap 原始字节并反序列化(整段墙钟时间) / 视频解码+取字段+拼接成样本
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

import openpi.training.config_lingyu as _config
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

_NUM_RANDOM_SAMPLES = 5  # 随机抽查的样本数
_NUM_MEMORY_SAMPLES = 3  # test_span_fetch_memory 统计内存的样本数(另加 1 个预热样本); 深调用栈快照很慢, 每个样本约 7 分钟
_TRACE_FRAMES = 64  # tracemalloc 每个内存块记录的调用栈深度
_OUTPUT_DIR = Path(__file__).resolve().parent / "lingyu_dataset_v2_output"
_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"  # GPU 裁剪/缩放参数取自这个训练配置

# 被计时包装的原函数; 取样期间替换成下方 _timed_* 版本, 耗时累加进 _elapsed_seconds
_ORIGINAL_FETCH_SAMPLE_MESSAGES = lingyu_dataset_v2.fetch_sample_messages
_ORIGINAL_LOAD_SAMPLE_TOPICS = lingyu_dataset_v2.load_sample_topics
_elapsed_seconds = {"s3": 0.0, "index": 0.0}
# 被计数包装的原函数; test_span_fetch 期间替换成 _counted_fetch_mcap_bytes, 累计进 _fetch_stats
_ORIGINAL_FETCH_MCAP_BYTES = _mcap_utils.fetch_mcap_bytes
_fetch_stats = {"requests": 0, "bytes": 0}
_fetch_stats_lock = threading.Lock()


def test():
    """随机取 5 个样本, 核对编号反查与输出结构, 并把样本落盘。"""
    index_path = Path(WAREHOUSE_DIR, GLOBAL_INDEX_NAME)
    if not index_path.exists():
        pytest.skip(f"没有 {index_path}, 请先运行 build_sample_idx.py")

    # 与训练一样, GPU 裁剪/缩放参数取自 openpi 训练配置
    train_config = _config.get_config(_CONFIG_NAME)
    video_key_to_crop, image_resolution = load_video_decode_config(
        train_config.data.create(train_config.assets_dirs, train_config.model))
    dataset = LingyuDatasetV2(video_key_to_crop=video_key_to_crop, image_resolution=image_resolution)
    logger.info(f"len(dataset) = {len(dataset)}, {len(dataset.episodes)} 个 episode")

    # --- 1. 总数与索引一致 ---
    total_samples = duckdb.sql(f"SELECT sum(num_samples) FROM '{index_path}'").fetchone()[0]
    assert len(dataset) == total_samples, "len(dataset) 与全局索引的 sample 总数不符"

    # 清空上一次运行的输出, 目录里只留本次随机抽到的样本
    shutil.rmtree(_OUTPUT_DIR, ignore_errors=True)
    random_samples_idx = sorted(random.sample(range(len(dataset)), _NUM_RANDOM_SAMPLES))
    logger.info(f"随机抽取的全局 sample 编号: {random_samples_idx}")
    for global_sample_idx in random_samples_idx:
        # --- 2. 反查结果与 DuckDB 一致 ---
        located = duckdb.sql(
            f"SELECT prompt, episode_id, {global_sample_idx} - sample_offset AS sample_idx "
            f"FROM '{index_path}' "
            f"WHERE sample_offset <= {global_sample_idx} "
            f"  AND {global_sample_idx} < sample_offset + num_samples").fetchall()
        assert len(located) == 1, f"全局 sample {global_sample_idx} 反查到 {len(located)} 个 episode"
        assert dataset.locate_sample(global_sample_idx) == located[0], \
            f"locate_sample({global_sample_idx}) 与 DuckDB 反查不一致"

        # --- 3. 取样, 核对结构 ---
        _elapsed_seconds.update(s3=0.0, index=0.0)
        start_time = time.perf_counter()
        with (patch.object(lingyu_dataset_v2, "fetch_sample_messages", _timed_fetch_sample_messages),
              patch.object(lingyu_dataset_v2, "load_sample_topics", _timed_load_sample_topics)):
            sample = dataset[global_sample_idx]
        elapsed = time.perf_counter() - start_time
        # 总耗时扣掉读索引与读 S3, 余下即由原始字节转换为样本的耗时
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

        # --- 4. 维度 == 拼接配置里各字段长度之和 ---
        for vector_name, vector, concat_config in (
            ("state", state, dataset.state_concat),("action", action, dataset.action_concat),):
            assert vector.shape[-1] == sum(
                # 拼接的每一项就是该 topic 那个字段的长度, 逐项相加即向量总宽
                len(part) for part in _concat_parts(dataset, global_sample_idx, concat_config)), \
                f"{vector_name} 维度与拼接配置不符: {tuple(vector.shape)}"

        logger.info(f"sample {global_sample_idx} -> {sample['prompt']!r}, "
                    f"{ {key: tuple(frame.shape) for key, frame in images.items()} }, "
                    f"state={tuple(state.shape)}, action={tuple(action.shape)}, "
                    f"总耗时 {elapsed:.3f}s = 读索引 {_elapsed_seconds['index']:.3f}s "
                    f"+ S3 读原始数据 {_elapsed_seconds['s3']:.3f}s + 转换为样本 {convert_elapsed:.3f}s")
        _save_sample(sample, global_sample_idx, located[0])

    # --- 5. 同一 episode 内相邻样本内容不同 ---
    prompt, episode_id, sample_idx = dataset.locate_sample(0)
    assert dataset.locate_sample(1)[:2] == (prompt, episode_id), "前两个样本不在同一 episode, 无法比较"
    assert not torch.equal(dataset[0]["observation"]["state"], dataset[1]["observation"]["state"]), \
        "相邻样本的 state 完全相同, 取样可能一直落在同一条 message 上"


def test_span_fetch():
    """按 chunk 合并读取与逐条读取得到的 message 逐字节一致, 并统计两种方式的 S3 请求次数与下载量。"""
    index_path = Path(WAREHOUSE_DIR, GLOBAL_INDEX_NAME)
    if not index_path.exists():
        pytest.skip(f"没有 {index_path}, 请先运行 build_sample_idx.py")

    dataset = LingyuDatasetV2(load_images=False)  # 只用它反查 sample 的物理位置, 不读图像
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

        # 对照组: 改动前的逐条读取, 每个 location 单独 fetch_message
        _fetch_stats.update(requests=0, bytes=0)
        with _count_fetch_bytes():
            single_messages = {topic: [lingyu_dataset_v2.fetcher.fetch_message(
                message["msg_type"], message["msg_def"], [tuple(location.values())])[0]
                for location in message["locations"]] for topic, message in used_topics.items()}
        single_stats = (_fetch_stats["requests"], _fetch_stats["bytes"])

        # 重新序列化成 CDR 逐字节比对, 字节相同即 message 内容完全相同
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
    """合并读取的整段字节在 fetch_sample_messages 返回后即被释放, 返回结果只持有各条 message 自己的 CDR 副本。

    用 tracemalloc 统计每个样本的 Python 内存: 峰值(读取期间)/返回后仍被结果持有/结果释放后的残留,
    并与该样本的 S3 下载字节对照。tracemalloc 自身会大量占内存, 故不看 RSS。
    """
    index_path = Path(WAREHOUSE_DIR, GLOBAL_INDEX_NAME)
    if not index_path.exists():
        pytest.skip(f"没有 {index_path}, 请先运行 build_sample_idx.py")

    dataset = LingyuDatasetV2(load_images=False)  # 只用它反查 sample 的物理位置, 不读图像
    state_and_action_topics = set(load_mcap_state_and_action_topics_fields())
    sample_idxs = random.sample(range(len(dataset)), _NUM_MEMORY_SAMPLES + 1)
    tracemalloc.start(_TRACE_FRAMES)  # 记录足够深的调用栈, 才能认出经由 fetch_mcap_bytes 分配的内存块
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
            continue  # 首个样本含 S3 client 创建、typestore 注册等一次性开销, 不计入

        downloaded = _fetch_stats["bytes"]
        # 结果仍在时, 本次取样新增且调用栈经过 _mcap_utils(即 fetch_mcap_bytes 下载)的存活内存块 = 未释放的整段字节
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
    """把 mcap_message_fetcher 与 _mcap_utils 里的 fetch_mcap_bytes 换成计数版, 累计请求次数与下载字节。"""
    with (patch.object(mcap_message_fetcher, "fetch_mcap_bytes", _counted_fetch_mcap_bytes),
          patch.object(_mcap_utils, "fetch_mcap_bytes", _counted_fetch_mcap_bytes)):
        yield


def _counted_fetch_mcap_bytes(mcap_url: str, length: int, offset: int) -> bytes:
    """计数版 fetch_mcap_bytes: 多线程并发调用, 计数在锁内累加。"""
    data_bytes = _ORIGINAL_FETCH_MCAP_BYTES(mcap_url, length, offset)
    with _fetch_stats_lock:
        _fetch_stats["requests"] += 1
        _fetch_stats["bytes"] += len(data_bytes)
    return data_bytes


def _timed_fetch_sample_messages(sample_topics: dict, thread_pool) -> dict:
    """计时版 fetch_sample_messages: 一个 sample 全部 location 并行读 S3 的整段墙钟时间。"""
    start_time = time.perf_counter()
    topic_messages = _ORIGINAL_FETCH_SAMPLE_MESSAGES(sample_topics, thread_pool)
    _elapsed_seconds["s3"] += time.perf_counter() - start_time
    return topic_messages


def _timed_load_sample_topics(data_file: str, row_group: int, sample_idx: int) -> dict[str, dict]:
    """计时版 load_sample_topics: 按 (data_file, row_group) 读该 sample 的 locations 索引。"""
    start_time = time.perf_counter()
    sample_topics = _ORIGINAL_LOAD_SAMPLE_TOPICS(data_file, row_group, sample_idx)
    _elapsed_seconds["index"] += time.perf_counter() - start_time
    return sample_topics


def _save_sample(sample: dict, global_sample_idx: int, located: tuple):
    """把一个样本存到 sample_<全局编号>/ 下: 每个 camera 一张 png, state/action 合存一个 json。"""
    sample_dir = _OUTPUT_DIR / f"sample_{global_sample_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for camera_key, frame in sample["observation"]["images"].items():
        Image.fromarray(frame).save(sample_dir / f"{camera_key}.png")  # 已是 (H, W, 3) uint8

    prompt, episode_id, sample_idx = located
    json_path = sample_dir / "state_action.json"
    json_path.write_text(json.dumps({
        "prompt": prompt, "episode_id": episode_id, "sample_idx": sample_idx,
        "state": sample["observation"]["state"].tolist(), "action": sample["action"].tolist(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"sample {global_sample_idx} 已保存到 {sample_dir}")


def _concat_parts(dataset, global_sample_idx, concat_config):
    """取出该样本在拼接配置里每一项对应的一维数组, 供核对维度用。"""
    episode = dataset.locate_episode(global_sample_idx)
    sample_topics = load_sample_topics(
        episode["data_file"], episode["row_group"], global_sample_idx - episode["sample_offset"])
    # 只挑拼接用到的 topic, 免得为核对维度把三路视频再解一遍
    topic_data = filter_sample_messages(
        {topic: sample_topics[topic] for topic, _ in concat_config})
    return [topic_data[topic][0][field_name] for topic, field_name in concat_config]


if __name__ == "__main__":
    test()
