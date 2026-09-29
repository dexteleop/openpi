# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""按全局 sample 编号随机取样的 torch Dataset: episode_index.parquet + Iceberg 数据文件 -> 训练样本。

取样链路(编号与 build_sample_idx.py 生成的全局索引一一对应)::
    全局 sample 编号 i
      -> episode_index.parquet: sample_offset <= i < sample_offset + num_samples
      -> (该 episode 所在的 data_file, row_group, 该 episode 内的 sample_idx)
      -> 直接 read_row_group 读出那一行的 samples 列, 只把 samples[sample_idx] 转成 Python 对象
      -> {topic: {msg_type, msg_def, log_time, locations}}
      -> 全部 topic 的全部 location 按 record_length 从大到小提交线程池, 并行从 S3 读取并反序列化,
         再按 topic 归组
      -> 视频 topic 并行解码出当前帧; state/action topic 按 model_config 的拼接顺序拼成向量

__getitem__ 的输出与 lingyu_dataloader/webdataset_load_tar.py 对齐(只多一个 prompt)::
    {"observation": {"images": {"left_color": (1, C, H, W) float32 [0,1], ...},
                     "state":  (1, state_dim) float32},
     "action": (ACTION_CHUNK_LENGTH, action_dim) float32,
     "prompt": str}
图像为 channel-first、缩放到 [0,1]; state/action 原样拼接, 不做归一化与单位换算。

不经 Iceberg scan: 它每次都要遍历全部 manifest 与候选文件的 footer 才能找到这一行, 且会把整个
episode 的上千个 sample 都转成 Python 对象; 索引里已记下物理位置, 直接读那个 row group 即可。
fetcher 是模块级的: DataLoader 以 spawn 起多进程时每个 worker 各建一份, 不会有任何句柄跨进程传递。
线程池则每次取样现建现关: 不留常驻线程, DataLoader 以 fork 起 worker 时也不会继承到失效的线程池。

用法::
    dataset = LingyuDatasetV2()
"""
from __future__ import annotations

import logging
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch

from openpi.training.lingyu_dataloader_v2.build_sample_idx import (
    GLOBAL_INDEX_NAME,
    WAREHOUSE_DIR)
from openpi.training.lingyu_dataloader_v2.mcap_config.config import (
    load_mcap_video_topics_gop,
    load_mcap_state_and_action_topics_fields)
from openpi.training.lingyu_dataloader_v2.model_config.config import (
    load_state_and_action_concat)
from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    S3_MAX_POOL_CONNECTIONS)
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import (
    MCAP_Message_Fetcher)
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    LOCATION_TYPE)
from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import (
    ros2_message_filter)
from openpi.training.lingyu_dataloader_v2.utils.video_frame_decoder import (
    cpu_decode_current_frame)


# Constants
# 将 Standard Video Topics 与 openpi 代码名称适配
VIDEO_TOPIC_TO_KEY = {
    '/left/color/image_raw/ffmpeg': 'left_color',
    '/right/color/image_raw/ffmpeg': 'right_color',
    '/xr_video_topic/ffmpeg': 'head_camera',
}
# 一个 sample 内并行读 message 的线程数。一个 sample 约 180 个 location、每个 2 次 S3 请求;
# 实测每次请求 boto3 要占约 1.1ms 持 GIL 的 CPU, 8 线程后墙钟时间不再下降, 再多只会多建连接
SAMPLE_FETCH_WORKERS = 8
# 全局索引里取样只需要这几列
INDEX_COLUMNS = ("prompt", "episode_id", "num_samples", "data_file", "row_group", "sample_offset")

logger = logging.getLogger(__name__)

fetcher = MCAP_Message_Fetcher()


def load_episode_index(warehouse_dir: str) -> list[dict]:
    """读 episode_index.parquet, 返回按 sample_offset 升序的 episode 列表。"""
    index_path = Path(warehouse_dir, GLOBAL_INDEX_NAME)
    assert index_path.exists(), f"没有 {index_path}, 请先运行 build_sample_idx.py"
    index_columns = duckdb.sql(f"SELECT * FROM '{index_path}' LIMIT 0").columns
    assert set(INDEX_COLUMNS) <= set(index_columns), \
        f"{index_path} 缺少 {set(INDEX_COLUMNS) - set(index_columns)}, 请运行 build_sample_idx.py --index-only 重建索引"
    index_rows = duckdb.sql(f"SELECT {', '.join(INDEX_COLUMNS)} FROM '{index_path}' "
                            f"ORDER BY sample_offset").fetchall()
    return [dict(zip(INDEX_COLUMNS, row)) for row in index_rows]


def load_sample_topics(data_file: str, row_group: int, sample_idx: int) -> dict[str, dict]:
    """读该 episode 所在的 row group, 只把第 sample_idx 个 sample 转成 Python 对象。"""
    samples_column = pq.ParquetFile(data_file).read_row_group(row_group, columns=["samples"]).column(0)
    # 写入端一个 episode 一行一个 row group, 故这里恰好一行
    assert len(samples_column) == 1, f"{data_file} 的 row group {row_group} 有 {len(samples_column)} 行"
    episode_samples = samples_column[0].values      # 该 episode 全部 sample 的 Arrow 数组, 不转 Python
    assert 0 <= sample_idx < len(episode_samples), \
        f"sample_idx={sample_idx} 越界, 该 episode 只有 {len(episode_samples)} 个 sample, 索引与数据不同步"
    return episode_samples[sample_idx].as_py()


def locations_to_tuples(locations: list[dict]) -> list[tuple]:
    """Iceberg 里的 location STRUCT -> fetcher/解码器要的四元组, 顺序由 LOCATION_TYPE 决定。"""
    return [tuple(location[field_name] for field_name in LOCATION_TYPE.names)
            for location in locations]


def fetch_sample_messages(sample_topics: dict[str, dict],
                          thread_pool: ThreadPoolExecutor) -> dict[str, list]:
    """把一个 sample 全部 topic 的全部 location 一起提交线程池, 并行读取并反序列化。

    返回 {topic: [ros2 message, ...]}, 每个 topic 内的顺序与其 locations 顺序一致。
    """
    # 展平成 (topic, location), 一个 location 一个任务
    flat_locations = [(topic, location) for topic, message in sample_topics.items()
                      for location in message["locations"]]
    # 按 record_length 从大到小提交: 传输久的大包先开始, 不会排到最后拖长整体耗时
    submit_order = sorted(range(len(flat_locations)),
                          key=lambda flat_idx: flat_locations[flat_idx][1]["record_length"], reverse=True)
    location_futures = dict()
    for flat_idx in submit_order:
        topic, location = flat_locations[flat_idx]
        message = sample_topics[topic]
        location_futures[flat_idx] = thread_pool.submit(
            fetcher.fetch_message, message["msg_type"], message["msg_def"], locations_to_tuples([location]))

    # 按展平前的原顺序取结果, 故各 topic 内的 message 顺序与 locations 一致
    topic_messages = {topic: [] for topic in sample_topics}
    for flat_idx, (topic, _) in enumerate(flat_locations):
        topic_messages[topic].append(location_futures[flat_idx].result()[0])
    return topic_messages


def filter_sample_messages(sample_topics: dict[str, dict]) -> dict[str, np.ndarray | list[dict]]:
    """
    输入一个 sample 的 {topic: {msg_type, msg_def, log_time, locations}},
    输出 {视频 topic: 当前帧数组, state/action topic: [{字段名: 一维数组}, ...]}。

    视频 topic 的 locations 是一整段 GOP, 解码后只留最后一帧(即当前帧);
    state/action topic 的每个 location 各是一条独立 message, 反序列化后按配置取字段,
    故 state 得到长度 1 的列表, action 得到长度 ACTION_CHUNK_LENGTH 的列表(下标即未来第几步)。
    全部 location 先并行读完, 各路视频再并行解码, 最后汇成一个 dict。
    """
    video_topics_gop = load_mcap_video_topics_gop()
    state_and_action_fields = load_mcap_state_and_action_topics_fields()
    # None 是 schema 为对齐所有行补出的空位; 配置里没用到的 topic 不读
    used_topics = {topic: message for topic, message in sample_topics.items()
                   if message is not None and (topic in video_topics_gop or topic in state_and_action_fields)}

    with ThreadPoolExecutor(max_workers=SAMPLE_FETCH_WORKERS) as thread_pool:
        topic_messages = fetch_sample_messages(used_topics, thread_pool)
        # 各路视频互不依赖, PyAV 解码时释放 GIL, 故同样放进线程池并行
        frame_futures = {topic: thread_pool.submit(cpu_decode_current_frame, packets)
                         for topic, packets in topic_messages.items() if topic in video_topics_gop}
        topic_data = {topic: [ros2_message_filter(topic, ros2_message) for ros2_message in ros2_messages]
                      for topic, ros2_messages in topic_messages.items() if topic in state_and_action_fields}
        topic_data.update({topic: future.result() for topic, future in frame_futures.items()})

    return topic_data


def concat_vector(topic_data: dict, concat_config: tuple, step_idx: int) -> np.ndarray:
    """按 model_config 的 (topic, 字段) 顺序拼出第 step_idx 步的一维向量。"""
    return np.concatenate([topic_data[topic][step_idx][field_name]
                           for topic, field_name in concat_config]).astype(np.float32)


def to_image_tensor(frame: np.ndarray) -> torch.Tensor:
    """解码出的 (H, W, 3) uint8 帧 -> (1, C, H, W) float32 [0,1]。"""
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0


class LingyuDatasetV2(torch.utils.data.Dataset):
    """以全局 sample 编号随机取样: 索引来自 episode_index.parquet, 数据来自 Iceberg 数据文件 + mcap。

    __init__ 只读全局索引, 不打开任何数据文件与 mcap, 故对象里全是路径与整数, 可安全 pickle 进
    DataLoader 的 worker 进程; 数据文件每次取样现开现读, 解压缓存在首次取样时于各自进程内建立。
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

    def locate_episode(self, index: int) -> dict:
        """全局 sample 编号 -> 它所在 episode 的索引行(INDEX_COLUMNS)。"""
        assert 0 <= index < self.num_samples, f"sample 编号 {index} 越界 [0, {self.num_samples})"
        # sample_offset 升序且连续覆盖, 故落在最后一个不大于 index 的 episode 里
        return self.episodes[bisect_right(self.sample_offsets, index) - 1]

    def locate_sample(self, index: int) -> tuple[str, str, int]:
        """全局 sample 编号 -> (prompt, episode_id, 该 episode 内的 sample_idx)。"""
        episode = self.locate_episode(index)
        return episode["prompt"], episode["episode_id"], index - episode["sample_offset"]

    def __getitem__(self, index: int) -> dict:
        """取出一个 sample: 反查 episode 物理位置 -> 解码/反序列化 -> 拼成模型需要的 state/action。"""
        episode = self.locate_episode(index)
        prompt = episode["prompt"]
        topic_data = filter_sample_messages(load_sample_topics(
            episode["data_file"], episode["row_group"], index - episode["sample_offset"]))

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


