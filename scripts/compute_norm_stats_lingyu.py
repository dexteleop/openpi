"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import glob
import io
import logging
import tarfile

import numpy as np
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config_lingyu as _config
import openpi.training.data_loader_lingyu as _data_loader
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import (
    SAMPLE_FETCH_WORKERS, LingyuDatasetV2, concat_vector, fetch_samples_messages, load_samples_topics,
    load_state_and_action_concat)
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import mcap_key_from_index
from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import ros2_message_filter
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class BatchSampleDataset(torch.utils.data.Dataset):
    """一个 item = batch_size 个连续 sample: 定位一次读齐, mcap 消息跨 sample 合并成大段拉取。

    顺序遍历(全量)时 batch 内 sample 连续且物理相邻: parquet row group 只读一次;
    全部 sample 的 location 一起按 chunk 合并(MERGE_GAP), 连续消息并成少量大段,
    下载量略多但 S3 请求数大幅减少, 比逐 sample 取样更快更稳。视频不读(compute_norm_stats 场景)。
    """

    def __init__(self, warehouse_dir: str, batch_size: int, sample_transform):
        self._raw_dataset = LingyuDatasetV2(warehouse_dir, load_images=False)
        self._batch_size = batch_size
        self._sample_transform = sample_transform
        self._state_concat, self._action_concat = load_state_and_action_concat()
        self._action_topics = {topic for topic, _ in self._action_concat}
        self._state_action_topics = {topic for topic, _ in self._state_concat} | self._action_topics

    def __len__(self) -> int:
        return len(self._raw_dataset) // self._batch_size

    def _episode_ranges(self, batch_start: int, batch_end: int) -> list[tuple[dict, int, int]]:
        """把 [batch_start, batch_end) 按所在 episode 切成 (episode 行, episode 内起点, 数量) 列表。"""
        episode_ranges = []
        position = batch_start
        while position < batch_end:
            episode = self._raw_dataset.locate_episode(position)
            episode_end = episode["sample_offset"] + episode["num_samples"]
            sample_count = min(batch_end, episode_end) - position
            episode_ranges.append((episode, position - episode["sample_offset"], sample_count))
            position += sample_count
        return episode_ranges

    def _episode_samples(self, episode: dict, local_start: int, sample_count: int) -> list[dict]:
        """取一个 episode 内连续 sample_count 个 sample, 各自走 transforms 后返回样本 dict 列表。"""
        samples_topics = []
        for sample_topics in load_samples_topics(episode["data_file"], episode["row_group"], local_start, sample_count):
            used_topics = {}
            for topic, message in sample_topics.items():
                if message is None or topic not in self._state_action_topics:
                    continue
                # 索引每 sample 存 100 步动作, 只取前 action_chunk_length 步(与逐 sample 取样相同)
                if topic in self._action_topics:
                    message = {**message, "locations": message["locations"][:self._raw_dataset.action_chunk_length]}
                used_topics[topic] = message
            samples_topics.append(used_topics)
        with ThreadPoolExecutor(max_workers=SAMPLE_FETCH_WORKERS) as thread_pool:
            samples_messages = fetch_samples_messages(
                samples_topics, thread_pool, mcap_key_from_index(episode["data_file"]))

        episode_samples = []
        for topic_messages in samples_messages:
            topic_data = {topic: [ros2_message_filter(topic, ros2_message) for ros2_message in ros2_messages]
                          for topic, ros2_messages in topic_messages.items()}
            state = concat_vector(topic_data, self._state_concat, 0)
            action = np.stack([concat_vector(topic_data, self._action_concat, step_idx)
                               for step_idx in range(self._raw_dataset.action_chunk_length)])
            episode_samples.append(self._sample_transform({
                "observation": {"images": {}, "state": torch.from_numpy(state)},
                "action": torch.from_numpy(action),
                "prompt": episode["prompt"],
            }))
        return episode_samples

    def __getitem__(self, batch_idx: int) -> dict:
        batch_start = batch_idx * self._batch_size
        batch_end = min(batch_start + self._batch_size, len(self._raw_dataset))
        batch_samples = [sample
                         for episode, local_start, sample_count in self._episode_ranges(batch_start, batch_end)
                         for sample in self._episode_samples(episode, local_start, sample_count)]
        assert len(batch_samples) == batch_end - batch_start, "batch 内 sample 数与预期不符"
        return {key: np.stack([np.asarray(sample[key]) for sample in batch_samples])
                for key in ("state", "actions")}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    sample_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ]
    )
    if max_frames is not None and max_frames < len(LingyuDatasetV2(data_config.iceberg_dir, load_images=False)):
        # 抽样模式: sample 随机不连续, 跨 sample 合并无收益, 走原逐 sample 路径;
        # TransformedDataset 内部还会 compose 一次, 故这里传列表而不是已 compose 的单个对象
        dataset = _data_loader.TransformedDataset(
            LingyuDatasetV2(data_config.iceberg_dir, load_images=False),
            [sample_transform],
        )
        num_batches = max_frames // batch_size
        shuffle = True
        data_loader = _data_loader.TorchDataLoader(
            dataset, local_batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, num_batches=num_batches)
        return data_loader, num_batches

    # 全量模式: 一个 item = 一个 batch 的连续 sample, row group 与 mcap 段都按 batch 合并拉取
    batch_dataset = BatchSampleDataset(data_config.iceberg_dir, batch_size, sample_transform)
    num_batches = len(batch_dataset)
    data_loader = _data_loader.TorchDataLoader(
        batch_dataset, local_batch_size=1, num_workers=num_workers,
        shuffle=False, num_batches=num_batches, framework="pytorch")
    return data_loader, num_batches


def compute_serial_norm_stats(
    data_config: _config.DataConfig, config: _config.TrainConfig, max_frames: int | None = None,
) -> dict[str, normalize.NormStats]:
    """全部 sample 切成 batch, 逐个喂给每个 key 的一个 RunningStats。"""
    # parquet 索引已本地缓存(build_sample_idx.py 构建索引时自动下载), 对 minio 只剩 mcap 字节读取;
    # 但 mcap 数据也在本地单块 SSD 上, 256 worker x 2 线程的并发读会把磁盘队列打满(实测卡死),
    # 64(核数)是本机吞吐最优的并发, 封顶不沿用训练 config 的 256
    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, min(config.num_workers, 64), max_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    return {key: stats.get_statistics() for key, stats in stats.items()}


def create_norm_data_config(config: _config.TrainConfig) -> _config.DataConfig:
    """强制走只含 state/action 的 transforms, 训练 config 里无需打开 compute_norm_stats。"""
    assert isinstance(config.data, _config.LingyuTeleavatarV2DataConfig), \
        f"{config.name} 不是 LingyuTeleavatarV2DataConfig, 不能用本脚本计算 norm stats"
    return dataclasses.replace(config.data, compute_norm_stats=True).create(config.assets_dirs, config.model)


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = create_norm_data_config(config)

    norm_stats = compute_serial_norm_stats(data_config, config, max_frames)

    # repo_id 为 None 时直接写到 assets_dirs 下
    output_path = config.assets_dirs if data_config.repo_id is None else config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    # force=True: orbax 经 absl 在 import 期给 root 预装了默认 handler, 不强制覆盖时本配置整体 no-op,
    # INFO 全被默认 WARNING 级别过滤, log 文件与终端都无输出
    logging.basicConfig(force=True, level=logging.INFO, handlers=[
        logging.FileHandler(__file__.removesuffix(".py") + ".log", mode="w", encoding="utf-8"),
        logging.StreamHandler()])
    tyro.cli(main)
