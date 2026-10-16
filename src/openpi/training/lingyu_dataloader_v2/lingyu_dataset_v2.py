# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""按全局 sample 编号随机取样的 torch Dataset: episode_index.parquet + Iceberg 数据文件 -> 训练样本。

取样链路(编号与 build_sample_idx.py 生成的全局索引一一对应)::
    全局 sample 编号 i
      -> episode_index.parquet: sample_offset <= i < sample_offset + num_samples
      -> (该 episode 所在的 data_file, row_group, 该 episode 内的 sample_idx)
      -> 直接 read_row_group 读出那一行的 samples 列, 只把 samples[sample_idx] 转成 Python 对象
      -> {topic: {msg_type, msg_def, log_time, locations}}; location 只是 chunk 内偏移三元组,
         mcap 的桶内 key 由 data_file 反推(mcap_key_from_index): 索引与 mcap 同目录同名
      -> 全部 topic 的全部 location 按所在 chunk 分组, 组内间隔不超过 MERGE_GAP 的并成一段,
         一段一次 S3 range 请求, 按段提交线程池并行读取、切片、反序列化, 再按 topic 归组
      -> 视频 topic 发给 GPU 解码进程, 解码、裁出单目、缩放都在 GPU 上完成;
         state/action topic 按 model_config 的拼接顺序拼成向量

__getitem__ 的输出为单帧、不带前导帧维度的样本(可直接进 RepackTransform)::
    {"observation": {"images": {"left_color": (140, 224, 3) uint8, "head_camera": (224, 224, 3) uint8, ...},
                     "state":  (state_dim,) float32},
     "action": (ACTION_CHUNK_LENGTH, action_dim) float32,
     "prompt": str}
图像已按 openpi 的变换在 GPU 上裁好单目(TeleavatarInputs)并不变形缩放(ResizeImages 的 resize_with_pad),
下游这两步遇到这样的图分别原样通过、只补黑边, 故 openpi 配置不变; worker 里不再有整幅 4K 帧的解码与缩放。
裁剪与缩放参数不另行配置, 由 load_video_decode_config() 从 openpi 的 DataConfig 里取。
uint8 HWC 不转 float; state/action 原样拼接, 不做归一化与单位换算。

不经 Iceberg scan: 它每次都要遍历全部 manifest 与候选文件的 footer 才能找到这一行, 且会把整个
episode 的上千个 sample 都转成 Python 对象; 索引里已记下物理位置, 直接读那个 row group 即可。
fetcher 是模块级的: DataLoader 以 forkserver 起 worker 时每个 worker 各持一份 fork 出的副本, 不会有任何句柄跨进程传递。
线程池则每次取样现建现关: 不留常驻线程, DataLoader 以 fork 起 worker 时也不会继承到失效的线程池。

用法::
    video_key_to_crop, image_resolution = load_video_decode_config(data_config)
    dataset = LingyuDatasetV2(data_config.iceberg_dir, video_key_to_crop=video_key_to_crop,
                              image_resolution=image_resolution)
"""
from __future__ import annotations

import logging
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from openpi.policies import teleavatar_v2_policy
from openpi.training.lingyu_dataloader_v2.build_sample_idx import (
    GLOBAL_INDEX_NAME,
    PARQUET_CACHE_NAME,
    WAREHOUSE_DIR)
from openpi.training.lingyu_dataloader_v2.mcap_config.config import (
    load_mcap_video_topics_gop)
from openpi.training.lingyu_dataloader_v2.model_config.config import (
    load_action_chunk_length,
    load_state_and_action_concat)
from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    RECORD_PREFIX,
    S3_MAX_POOL_CONNECTIONS)
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    mcap_key_from_index)
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import (
    MCAP_Message_Fetcher)
from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import (
    ros2_message_filter)
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
    make_arrow_filesystem)
from openpi.training.lingyu_dataloader_v2.utils.video_frame_decoder import (
    cpu_decode_current_frame,
    eye_crop_and_size,
    gpu_decode_current_frames,
    start_gpu_decoders)
import openpi.transforms as _transforms


# Constants
# 将 Standard Video Topics 与 openpi 代码名称适配
VIDEO_TOPIC_TO_KEY = {
    '/left/color/image_raw/ffmpeg': 'left_color',
    '/right/color/image_raw/ffmpeg': 'right_color',
    '/xr_video_topic/ffmpeg': 'head_camera',
}
# 一个 sample 内并行读 message 的线程数。全部 worker 同时在途的请求数 = num_workers x 本值;
# 256 worker x 8 线程(2048 并发)会压垮 OSS 网关, 持续出现读超时与 TooManyRequests, 故降为 2(512 并发)
SAMPLE_FETCH_WORKERS = 2
# 同一 chunk 内相邻 message 的间隔不超过它就并成一次 range 读取。一个 sample 的 state/action
# 约 122 个 location 落在 6 个 chunk 里, 合并后约 20 次请求(原为 244 次)、约 1.1MiB; 越大请求越少、多下载的无用字节越多
MERGE_GAP = 64 * 1024
# 全局索引里取样只需要这几列
INDEX_COLUMNS = ("prompt", "episode_id", "num_samples", "data_file", "row_group", "sample_offset")

logger = logging.getLogger(__name__)

fetcher = MCAP_Message_Fetcher()


def load_video_decode_config(data_config) -> tuple[dict[str, tuple[str, bool]], tuple[int, int]]:
    """从 openpi 的 DataConfig 取出 GPU 解码要复刻的图像变换参数, 返回 ({相机名: (保留哪只眼, 是否先转 180°)}, (高, 宽))。

    取眼规则来自 teleavatar_v2_policy.STEREO_EYES, 旋转来自 TeleavatarInputs.rotate_head_camera(与它一样只转头部相机),
    (高, 宽) 来自 model_transforms 里的 ResizeImages; 改这些配置, GPU 解码随之改变, 不会与 openpi 的变换不一致。
    """
    teleavatar_inputs = next(transform for transform in data_config.data_transforms.inputs
                             if isinstance(transform, teleavatar_v2_policy.TeleavatarInputs))
    resize_images = next(transform for transform in data_config.model_transforms.inputs
                         if isinstance(transform, _transforms.ResizeImages))
    video_key_to_crop = {key: (eye, key == "head_camera" and teleavatar_inputs.rotate_head_camera)
                         for key, eye in teleavatar_v2_policy.STEREO_EYES.items()}
    return video_key_to_crop, (resize_images.height, resize_images.width)


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


@lru_cache(maxsize=1)
def index_filesystem():
    """读桶内 mcap 索引的 pyarrow S3 文件系统; 首次取样时于各 worker 进程内现建, 不随 fork 继承"""
    return make_arrow_filesystem()


def read_samples_column(data_file: str, row_group: int):
    """读 episode parquet 指定 row group 的 samples 列; 本地缓存存在就读本地, 否则回退读桶。

    minio 对 GetObject 高并发限流(429)且 pyarrow 不重试, build_sample_idx.py 构建索引时会把
    索引 parquet 整体本地化, 取样的 parquet 读取因此不再打 minio。
    """
    # data_file 是 Iceberg 登记的 s3://桶/key, pyarrow 的 S3 文件系统要的是 桶/key
    assert data_file.startswith("s3://"), f"{data_file} 不是桶内路径, 请重跑 build_sample_idx.py 重建索引"
    bucket_name, object_key = data_file.removeprefix("s3://").split("/", 1)
    local_path = Path(WAREHOUSE_DIR, PARQUET_CACHE_NAME, object_key)
    if local_path.exists():
        return pq.ParquetFile(local_path).read_row_group(row_group, columns=["samples"]).column(0)
    logger.warning(f"{object_key} 不在本地缓存, 回退读 minio(高并发易被 429 限流), "
                   f"请重跑 build_sample_idx.py 重建索引并下载缓存")
    # pre_buffer: samples 的几十根叶子列各是一个列块, 不合并就是几十次 range 请求(实测 113 ms), 合并后约 24 ms
    return pq.ParquetFile(f"{bucket_name}/{object_key}", filesystem=index_filesystem(), pre_buffer=True) \
        .read_row_group(row_group, columns=["samples"]).column(0)


def load_sample_topics(data_file: str, row_group: int, sample_idx: int) -> dict[str, dict]:
    """读该 episode 所在的 row group, 只把第 sample_idx 个 sample 转成 Python 对象。"""
    samples_column = read_samples_column(data_file, row_group)
    # 写入端一个 episode 一行一个 row group, 故这里恰好一行
    assert len(samples_column) == 1, f"{data_file} 的 row group {row_group} 有 {len(samples_column)} 行"
    episode_samples = samples_column[0].values      # 该 episode 全部 sample 的 Arrow 数组, 不转 Python
    assert 0 <= sample_idx < len(episode_samples), \
        f"sample_idx={sample_idx} 越界, 该 episode 只有 {len(episode_samples)} 个 sample, 索引与数据不同步"
    return episode_samples[sample_idx].as_py()


def load_samples_topics(data_file: str, row_group: int, sample_start: int, sample_count: int) -> list[dict]:
    """一次读该 episode 所在的 row group, 批量取第 sample_start 起连续 sample_count 个 sample 的定位。

    顺序取样一批连续 sample 时, 同一 row group 只下载一次, 不再每个 sample 重复读整个 row group。
    """
    samples_column = read_samples_column(data_file, row_group)
    assert len(samples_column) == 1, f"{data_file} 的 row group {row_group} 有 {len(samples_column)} 行"
    episode_samples = samples_column[0].values
    assert 0 <= sample_start and sample_start + sample_count <= len(episode_samples), \
        f"[{sample_start}, {sample_start + sample_count}) 越界, 该 episode 只有 {len(episode_samples)} 个 sample"
    return [episode_samples[sample_start + offset].as_py() for offset in range(sample_count)]


def group_location_spans(flat_locations: list[tuple[str, dict]],
                         mcap_key: str) -> list[tuple[str, int, list[tuple[int, dict]]]]:
    """把展平的 (topic, location) 按所在 chunk 分组, 组内按偏移排序, 间隔不超过 MERGE_GAP 的并成一段。

    返回 [(mcap_key, chunk_file_offset, [(flat_idx, location), ...]), ...], 一段对应一次 S3 range 请求;
    flat_idx 为该 location 在 flat_locations 中的下标。mcap_key 由 data_file 反推(见 mcap_key_from_index),
    索引里的 location 只是 chunk 内偏移三元组, 不携带 mcap 位置。
    """
    chunk_locations = dict()
    for flat_idx, (_, location) in enumerate(flat_locations):
        chunk_key = (mcap_key, location["chunk_file_offset"])
        chunk_locations.setdefault(chunk_key, []).append((flat_idx, location))

    location_spans = []
    for (mcap_key, chunk_file_offset), chunk_items in chunk_locations.items():
        chunk_items.sort(key=lambda item: item[1]["uncompressed_byte_offset"])
        span_end = 0
        for item_idx, (flat_idx, location) in enumerate(chunk_items):
            location_start = location["uncompressed_byte_offset"]
            location_end = location_start + RECORD_PREFIX + location["record_length"]
            # 本 chunk 的第一条, 或与上一段末尾相隔超过 MERGE_GAP: 另起一段
            if item_idx == 0 or location_start - span_end > MERGE_GAP:
                location_spans.append((mcap_key, chunk_file_offset, []))
                span_end = location_end
            location_spans[-1][2].append((flat_idx, location))
            span_end = max(span_end, location_end)
    return location_spans


def fetch_sample_messages(sample_topics: dict[str, dict],
                          thread_pool: ThreadPoolExecutor, mcap_key: str) -> dict[str, list]:
    """把一个 sample 全部 topic 的全部 location 按 chunk 合并成若干段, 一段一个任务提交线程池并行读取并反序列化。

    返回 {topic: [ros2 message, ...]}, 每个 topic 内的顺序与其 locations 顺序一致。
    """
    flat_locations = [(topic, location) for topic, message in sample_topics.items()
                      for location in message["locations"]]
    location_spans = group_location_spans(flat_locations, mcap_key)
    # 按段内 message 总字节从大到小提交: 传输久的大段先开始, 不会排到最后拖长整体耗时
    location_spans.sort(key=lambda span: sum(location["record_length"] for _, location in span[2]), reverse=True)
    logger.debug(f"{len(flat_locations)} 个 location 合并为 {len(location_spans)} 次 S3 请求")

    span_futures = []
    for mcap_key, chunk_file_offset, span_items in location_spans:
        span_messages = [(sample_topics[flat_locations[flat_idx][0]]["msg_type"],
                          sample_topics[flat_locations[flat_idx][0]]["msg_def"],
                          location["uncompressed_byte_offset"], location["record_length"])
                         for flat_idx, location in span_items]
        span_futures.append(([flat_idx for flat_idx, _ in span_items], thread_pool.submit(
            fetcher.fetch_span_messages, mcap_key, chunk_file_offset, span_messages)))

    # 各段结果先按 flat_idx 放回展平前的位置, 故各 topic 内的 message 顺序与 locations 一致
    flat_messages = [None] * len(flat_locations)
    for flat_idxs, span_future in span_futures:
        for flat_idx, ros2_message in zip(flat_idxs, span_future.result(), strict=True):
            flat_messages[flat_idx] = ros2_message
    assert all(ros2_message is not None for ros2_message in flat_messages), "有 location 未被任何一段读取"

    topic_messages = {topic: [] for topic in sample_topics}
    for (topic, _), ros2_message in zip(flat_locations, flat_messages):
        topic_messages[topic].append(ros2_message)
    return topic_messages


def fetch_samples_messages(samples_topics: list[dict],
                           thread_pool: ThreadPoolExecutor, mcap_key: str) -> list[dict[str, list]]:
    """多个 sample 的全部 location 一起按 chunk 合并成段批量拉取, 再按 (sample, topic) 拆回。

    连续 sample 的消息在 chunk 里物理相邻, 合并后是少量大段: 下载量略多但 S3 请求数大幅减少, 更快更稳。
    返回 [sample][topic] -> [ros2 message, ...], 各 topic 内的顺序与其 locations 顺序一致。
    """
    flat_locations = []      # 与 fetch_sample_messages 相同的 (topic, location) 展平结构
    flat_owners = []         # 对齐记录每个 location 属于 (sample_idx, topic)
    for sample_idx, sample_topics in enumerate(samples_topics):
        for topic, message in sample_topics.items():
            for location in message["locations"]:
                flat_locations.append((topic, location))
                flat_owners.append((sample_idx, topic))
    location_spans = group_location_spans(flat_locations, mcap_key)
    # 按段内 message 总字节从大到小提交: 传输久的大段先开始, 不会排到最后拖长整体耗时
    location_spans.sort(key=lambda span: sum(location["record_length"] for _, location in span[2]), reverse=True)
    logger.debug(f"{len(flat_locations)} 个 location 合并为 {len(location_spans)} 次 S3 请求")

    span_futures = []
    for mcap_key, chunk_file_offset, span_items in location_spans:
        span_messages = [(samples_topics[sample_idx][topic]["msg_type"],
                          samples_topics[sample_idx][topic]["msg_def"],
                          location["uncompressed_byte_offset"], location["record_length"])
                         for flat_idx, location in span_items
                         for sample_idx, topic in (flat_owners[flat_idx],)]
        span_futures.append(([flat_idx for flat_idx, _ in span_items], thread_pool.submit(
            fetcher.fetch_span_messages, mcap_key, chunk_file_offset, span_messages)))

    # 各段结果先按 flat_idx 放回展平前的位置, 再按 (sample, topic) 拆回
    flat_messages = [None] * len(flat_locations)
    for flat_idxs, span_future in span_futures:
        for flat_idx, ros2_message in zip(flat_idxs, span_future.result(), strict=True):
            flat_messages[flat_idx] = ros2_message
    assert all(ros2_message is not None for ros2_message in flat_messages), "有 location 未被任何一段读取"

    samples_messages = [{} for _ in samples_topics]
    for (sample_idx, topic), ros2_message in zip(flat_owners, flat_messages):
        samples_messages[sample_idx].setdefault(topic, []).append(ros2_message)
    return samples_messages


def cpu_decode_eye_frame(packets: list, eye: str, rotate: bool, resolution: tuple[int, int]) -> np.ndarray:
    """没有可见 GPU 时的 CPU 解码: 输出与 GPU 解码进程相同, 即已裁单目(及旋转)、已不变形缩放的 uint8 RGB。

    裁单目与旋转直接用 openpi 的 _extract_stereo_view, 缩放后的尺寸取自与 GPU 同一个 eye_crop_and_size,
    故两条路径的输出形状一致, 下游 TeleavatarInputs / ResizeImages 同样原样通过、只补黑边。
    """
    full_frame = cpu_decode_current_frame(packets)
    _, resized_size = eye_crop_and_size(full_frame.shape[1], full_frame.shape[0], eye, rotate, resolution)
    eye_frame = teleavatar_v2_policy._extract_stereo_view(full_frame, eye, rotate=rotate)
    # PIL 的 BILINEAR 缩小时会按比例放宽采样核(抗锯齿), 与 openpi resize_with_pad 的 LINEAR 一致
    return np.asarray(Image.fromarray(eye_frame).resize(resized_size, Image.Resampling.BILINEAR))


def filter_sample_messages(sample_topics: dict[str, dict], mcap_key: str,
                           gpu_decoder_address: str | None = None,
                           video_key_to_crop: dict[str, tuple[str, bool]] | None = None,
                           image_resolution: tuple[int, int] | None = None,
                           load_images: bool = False) -> dict[str, np.ndarray | list[dict]]:
    """
    输入一个 sample 的 {topic: {msg_type, msg_def, log_time, locations}} 与其 mcap 的桶内 key,
    输出 {视频 topic: 当前帧数组, state/action topic: [{字段名: 一维数组}, ...]}。

    视频 topic 的 locations 是一整段 GOP, 交给 gpu_decoder_address 处的 GPU 解码进程, 只取回最后一帧(即当前帧),
    按 video_key_to_crop 裁单目(及旋转)、按 image_resolution 缩放, 二者取自 load_video_decode_config();
    state/action topic 的每个 location 各是一条独立 message, 反序列化后按拼接配置取字段,
    故 state 得到长度 1 的列表, action 得到长度 load_action_chunk_length() 的列表(下标即未来第几步;
    索引里存的是 INDEX_ACTION_CHUNK_LENGTH 步, 此处只截取前几步)。
    全部 location 先并行读完, 各路视频再一次发给解码进程并行解码, 最后汇成一个 dict。
    load_images=False 时视频 topic 既不从 S3 读也不解码(compute_norm_stats 只需要 state/action);
    load_images=True 但 gpu_decoder_address 为 None(没有可见 GPU)时, 改在本进程用 cpu_decode_eye_frame 解码。
    """
    video_topics_gop = load_mcap_video_topics_gop() if load_images else {}
    state_concat, action_concat = load_state_and_action_concat()
    action_topics = {topic for topic, _ in action_concat}
    state_action_topics = {topic for topic, _ in state_concat} | action_topics
    # None 是 schema 为对齐所有行补出的空位; 拼接配置里没用到的 topic 不读
    used_topics = {}
    for topic, message in sample_topics.items():
        if message is None or (topic not in video_topics_gop and topic not in state_action_topics):
            continue
        # 索引每 sample 存 INDEX_ACTION_CHUNK_LENGTH 步动作, 模型只要前 load_action_chunk_length() 步,
        # 先截断 locations, 网络读取与反序列化都只做需要的部分
        if topic in action_topics:
            message = {**message, "locations": message["locations"][:load_action_chunk_length()]}
        used_topics[topic] = message

    with ThreadPoolExecutor(max_workers=SAMPLE_FETCH_WORKERS) as thread_pool:
        topic_messages = fetch_sample_messages(used_topics, thread_pool, mcap_key)
    topic_data = {topic: [ros2_message_filter(topic, ros2_message) for ros2_message in ros2_messages]
                  for topic, ros2_messages in topic_messages.items() if topic in state_action_topics}

    video_topics = [topic for topic in topic_messages if topic in video_topics_gop]
    if not video_topics:
        return topic_data
    if gpu_decoder_address:
        # 一个 sample 的各路视频合成一次请求, 解码进程里各路并行解
        video_requests = [(topic, topic_messages[topic][0].encoding, *video_key_to_crop[VIDEO_TOPIC_TO_KEY[topic]],
                           image_resolution, [bytes(packet.data) for packet in topic_messages[topic]])
                          for topic in video_topics]
        topic_data.update(zip(video_topics, gpu_decode_current_frames(gpu_decoder_address, video_requests)))
    else:
        # 各路视频互不依赖, PyAV 解码时释放 GIL, 故放进线程池并行
        eyes, rotates = zip(*(video_key_to_crop[VIDEO_TOPIC_TO_KEY[topic]] for topic in video_topics))
        with ThreadPoolExecutor(max_workers=len(video_topics)) as thread_pool:
            topic_data.update(zip(video_topics, thread_pool.map(
                cpu_decode_eye_frame, [topic_messages[topic] for topic in video_topics], eyes, rotates,
                [image_resolution] * len(video_topics))))
    return topic_data


def concat_vector(topic_data: dict, concat_config: tuple, step_idx: int) -> np.ndarray:
    """按 model_config 的 (topic, 字段) 顺序拼出第 step_idx 步的一维向量。"""
    return np.concatenate([topic_data[topic][step_idx][field_name]
                           for topic, field_name in concat_config]).astype(np.float32)


class LingyuDatasetV2(torch.utils.data.Dataset):
    """以全局 sample 编号随机取样: 索引来自 episode_index.parquet, 数据来自桶内的 mcap 索引(Iceberg 数据文件) + mcap。

    __init__ 只读全局索引, 不打开任何数据文件与 mcap, 故对象里全是路径与整数, 可安全 pickle 进
    DataLoader 的 worker 进程; 数据文件每次取样现开现读, 解压缓存在首次取样时于各自进程内建立。
    """

    def __init__(self, warehouse_dir: str = WAREHOUSE_DIR, load_images: bool = True,
                 video_key_to_crop: dict[str, tuple[str, bool]] | None = None,
                 image_resolution: tuple[int, int] | None = None):
        self.warehouse_dir = warehouse_dir
        # False: 只取 state/action, 输出的 images 为空 dict
        self.load_images = load_images
        # 读图像时必须给出 GPU 裁剪/缩放参数, 统一由 load_video_decode_config(data_config) 从 openpi 配置里取
        assert not load_images or (video_key_to_crop and image_resolution), \
            "load_images=True 时须传入 load_video_decode_config(data_config) 返回的 video_key_to_crop 与 image_resolution"
        self.video_key_to_crop = video_key_to_crop
        self.image_resolution = image_resolution
        self.episodes = load_episode_index(warehouse_dir)
        # 升序的 episode 起点, 供 bisect 由全局 sample 编号反查 episode
        self.sample_offsets = [episode["sample_offset"] for episode in self.episodes]
        self.num_samples = self.sample_offsets[-1] + self.episodes[-1]["num_samples"]
        self.state_concat, self.action_concat = load_state_and_action_concat()
        # 索引每 sample 存 100 步动作, 模型只要前 load_action_chunk_length() 步(取样时截取)
        self.action_chunk_length = load_action_chunk_length()
        # 主进程里先起好各 GPU 的解码进程, 对象只存其地址字符串, 照常可 pickle 进 worker;
        # 没有可见 GPU 时不起解码进程, 各 worker 在本进程用 CPU 解码(cpu_decode_eye_frame)
        self.gpu_decoder_addresses = start_gpu_decoders() if load_images and torch.cuda.device_count() > 0 else []

        logger.info(f"{self.num_samples} samples in {len(self.episodes)} episodes "
                    f"({len(set(e['prompt'] for e in self.episodes))} prompts) from {warehouse_dir}, "
                    f"load_images={load_images}, "
                    f"video_decode={'gpu' if self.gpu_decoder_addresses else 'cpu' if load_images else 'none'}")

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
        # 各 worker 按编号轮流分到各 GPU 的解码进程; 主进程里取样(num_workers=0)时用第一个
        worker_info = torch.utils.data.get_worker_info()
        gpu_decoder_address = self.gpu_decoder_addresses[
            (worker_info.id if worker_info else 0) % len(self.gpu_decoder_addresses)] if self.gpu_decoder_addresses else None
        # v3 索引的 location 不含 mcap 位置, 由 data_file 反推同目录同名 mcap 的桶内 key
        topic_data = filter_sample_messages(
            load_sample_topics(episode["data_file"], episode["row_group"], index - episode["sample_offset"]),
            mcap_key_from_index(episode["data_file"]),
            gpu_decoder_address, self.video_key_to_crop, self.image_resolution, self.load_images)

        state = concat_vector(topic_data, self.state_concat, 0)
        action = np.stack([concat_vector(topic_data, self.action_concat, step_idx)
                           for step_idx in range(self.action_chunk_length)])

        return {
            "observation": {
                "images": {key: topic_data[topic]
                           for topic, key in VIDEO_TOPIC_TO_KEY.items() if topic in topic_data},
                "state": torch.from_numpy(state),
            },
            "action": torch.from_numpy(action),
            "prompt": prompt,
        }

