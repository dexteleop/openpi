# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""按全局 sample 编号随机取样的 torch Dataset: episode_index.parquet + Iceberg -> 训练样本。

取样链路(编号与 build_sample_idx.py 生成的全局索引一一对应)::
    全局 sample 编号 i
      -> episode_index.parquet: sample_offset <= i < sample_offset + num_samples
      -> (prompt, episode_id, 该 episode 内的 sample_idx)
      -> Iceberg 表 {prompt 派生的 namespace}/episodes 那一行的 samples[sample_idx]
      -> {topic: {msg_type, msg_def, log_time, locations}}
      -> 视频 topic 解码出当前帧; state/action topic 反序列化后按 model_config 的拼接顺序拼成向量

__getitem__ 的输出与 lingyu_dataloader/webdataset_load_tar.py 对齐(只多一个 prompt)::
    {"observation": {"images": {"left_color": (1, C, H, W) float32 [0,1], ...},
                     "state":  (1, state_dim) float32},
     "action": (ACTION_CHUNK_LENGTH, action_dim) float32,
     "prompt": str}
图像为 channel-first、缩放到 [0,1]; state/action 原样拼接, 不做归一化与单位换算。

一次 Iceberg scan 读回的是一整行(该 episode 的全部 sample), 故按 episode 缓存最近读过的几行,
同一 episode 的相邻 sample 直接命中缓存。缓存与 fetcher 都是模块级的: DataLoader 以 spawn
起多进程时每个 worker 各建一份, 不会有任何句柄跨进程传递。

用法::
    dataset = LingyuDatasetV2()
"""
from __future__ import annotations

import logging
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import torch
from pyiceberg.expressions import EqualTo

from openpi.training.lingyu_dataloader_v2.build_sample_idx import GLOBAL_INDEX_NAME, WAREHOUSE_DIR
from openpi.training.lingyu_dataloader_v2.mcap_config.config import (
    load_mcap_video_topics_gop,
    load_mcap_state_and_action_topics_fields,
)
from openpi.training.lingyu_dataloader_v2.model_config.config import load_state_and_action_concat
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    EPISODES_TABLE_NAME, LOCATION_TYPE, load_episodes_catalog, prompt_to_namespace)
from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import ros2_message_filter
from openpi.training.lingyu_dataloader_v2.utils.video_frame_decoder import cpu_decode_current_frame

# Constants
# 视频 topic -> observation["images"] 里的键名, 与 lingyu_dataloader 的命名保持一致
VIDEO_TOPIC_TO_KEY = {
    '/left/color/image_raw/ffmpeg': 'left_color',
    '/right/color/image_raw/ffmpeg': 'right_color',
    '/xr_video_topic/ffmpeg': 'head_camera',
}
# 缓存最近几个 episode 的 samples; 一个 episode 上百 MB, 故只留很少几份
EPISODE_CACHE_SIZE = 2
# 全局索引里取样只需要这几列
INDEX_COLUMNS = ("prompt", "episode_id", "num_samples", "sample_offset")

logger = logging.getLogger(__name__)

_fetcher = MCAP_Message_Fetcher()


def load_episode_index(warehouse_dir: str) -> list[dict]:
    """读 episode_index.parquet, 返回按 sample_offset 升序的 episode 列表。"""
    index_path = Path(warehouse_dir, GLOBAL_INDEX_NAME)
    assert index_path.exists(), f"没有 {index_path}, 请先运行 build_sample_idx.py"
    index_rows = duckdb.sql(f"SELECT {', '.join(INDEX_COLUMNS)} FROM '{index_path}' "
                            f"ORDER BY sample_offset").fetchall()
    return [dict(zip(INDEX_COLUMNS, row)) for row in index_rows]


@lru_cache(maxsize=None)
def load_prompt_table(warehouse_dir: str, prompt: str):
    """打开某个 prompt(任务)的 Iceberg 表; 每个进程各开一份, 表句柄不跨进程传递。"""
    catalog = load_episodes_catalog(warehouse_dir)
    return catalog.load_table(f"{prompt_to_namespace(prompt)}.{EPISODES_TABLE_NAME}")


@lru_cache(maxsize=EPISODE_CACHE_SIZE)
def load_episode_samples(warehouse_dir: str, prompt: str, episode_id: str) -> list[dict]:
    """一次 scan 取回整个 episode 的 samples, 下标即该 episode 内的 sample_idx。"""
    # episode_id 在表内唯一(重试写重的行内容相同), 故 limit=1 取第一行即可
    episode_rows = load_prompt_table(warehouse_dir, prompt).scan(
        row_filter=EqualTo("episode_id", episode_id),
        selected_fields=("samples",), limit=1).to_arrow().to_pylist()
    assert episode_rows, f"Iceberg 表中没有 episode_id={episode_id!r}, 索引与数据不同步"
    return episode_rows[0]["samples"]


def locations_to_tuples(locations: list[dict]) -> list[tuple]:
    """Iceberg 里的 location STRUCT -> fetcher/解码器要的四元组, 顺序由 LOCATION_TYPE 决定。"""
    return [tuple(location[field_name] for field_name in LOCATION_TYPE.names)
            for location in locations]


def filter_sample_messages(sample_topics: dict[str, dict]) -> dict[str, np.ndarray | list[dict]]:
    """
    输入一个 sample 的 {topic: {msg_type, msg_def, log_time, locations}},
    输出 {视频 topic: 当前帧数组, state/action topic: [{字段名: 一维数组}, ...]}。

    视频 topic 的 locations 是一整段 GOP, 解码后只留最后一帧(即当前帧);
    state/action topic 的每个 location 各是一条独立 message, 逐条反序列化后按配置取字段,
    故 state 得到长度 1 的列表, action 得到长度 ACTION_CHUNK_LENGTH 的列表(下标即未来第几步)。
    """
    topic_data = dict()

    video_topics_gop = load_mcap_video_topics_gop()
    state_and_action_fields = load_mcap_state_and_action_topics_fields()

    for topic, message in sample_topics.items():
        if message is None:     # schema 为对齐所有行补出的空位
            continue
        locations = locations_to_tuples(message["locations"])
        if topic in video_topics_gop:
            topic_data[topic] = cpu_decode_current_frame(
                message["msg_type"], message["msg_def"], locations)
        elif topic in state_and_action_fields:
            topic_data[topic] = [
                ros2_message_filter(topic, ros2_message)
                for ros2_message in _fetcher.fetch_message(
                    message["msg_type"], message["msg_def"], locations)
            ]

    return topic_data


def concat_vector(topic_data: dict, concat_config: tuple, step_idx: int) -> np.ndarray:
    """按 model_config 的 (topic, 字段) 顺序拼出第 step_idx 步的一维向量。"""
    return np.concatenate([topic_data[topic][step_idx][field_name]
                           for topic, field_name in concat_config]).astype(np.float32)


def to_image_tensor(frame: np.ndarray) -> torch.Tensor:
    """解码出的 (H, W, 3) uint8 帧 -> (1, C, H, W) float32 [0,1]。"""
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0


class LingyuDatasetV2(torch.utils.data.Dataset):
    """以全局 sample 编号随机取样: 索引来自 episode_index.parquet, 数据来自 Iceberg + mcap。

    __init__ 只读全局索引, 不打开任何表与 mcap, 故对象里全是路径与整数, 可安全 pickle 进
    DataLoader 的 worker 进程; 表句柄、解压缓存都在首次取样时于各自进程内建立。
    """

    def __init__(self, warehouse_dir: str = WAREHOUSE_DIR):
        self.warehouse_dir = warehouse_dir
        self.episodes = load_episode_index(warehouse_dir)
        # 升序的 episode 起点, 供 bisect 由全局 sample 编号反查 episode
        self.sample_offsets = [episode["sample_offset"] for episode in self.episodes]
        self.num_samples = self.sample_offsets[-1] + self.episodes[-1]["num_samples"]
        self.state_concat, self.action_concat = load_state_and_action_concat()

        logger.info(f"{self.num_samples} samples in {len(self.episodes)} episodes "
                    f"({len(set(e['prompt'] for e in self.episodes))} prompts) from {warehouse_dir}")

    def __len__(self):
        """全部 prompt 的 sample 总数, 直接取自全局索引。"""
        return self.num_samples

    def locate_sample(self, index: int) -> tuple[str, str, int]:
        """全局 sample 编号 -> (prompt, episode_id, 该 episode 内的 sample_idx)。"""
        # sample_offset 升序且连续覆盖, 故落在最后一个不大于 index 的 episode 里
        episode = self.episodes[bisect_right(self.sample_offsets, index) - 1]
        return episode["prompt"], episode["episode_id"], index - episode["sample_offset"]

    def __getitem__(self, index: int) -> dict:
        """取出一个 sample: 反查 episode -> 解码/反序列化 -> 拼成模型需要的 state/action。"""
        prompt, episode_id, sample_idx = self.locate_sample(index)
        samples = load_episode_samples(self.warehouse_dir, prompt, episode_id)
        topic_data = filter_sample_messages(samples[sample_idx])

        # action 的步数由数据本身决定(即 locations 条数), 与 ACTION_CHUNK_LENGTH 一致
        action_steps = len(topic_data[self.action_concat[0][0]])
        state = concat_vector(topic_data, self.state_concat, 0)
        action = np.stack([concat_vector(topic_data, self.action_concat, step_idx)
                           for step_idx in range(action_steps)])

        return {
            "observation": {
                "images": {key: to_image_tensor(topic_data[topic])
                           for topic, key in VIDEO_TOPIC_TO_KEY.items() if topic in topic_data},
                "state": torch.from_numpy(state[None]),
            },
            "action": torch.from_numpy(action),
            "prompt": prompt,
        }


