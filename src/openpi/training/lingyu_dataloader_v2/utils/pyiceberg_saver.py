# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""把 MCAPSampleExtractor 产出的 episode 攒成批, 每批一次 Iceberg commit。

保存链路(详见 pyiceberg_saver_readme.md)::
    Python objects -> PyArrow Table(BATCH_EPISODES 行) -> table.append() -> 一次 snapshot

一行就是一个 episode::
    episode_id          string  = "{source_id}:{source_episode_seq:08d}", 全局稳定唯一, 供重试去重
    source_id           string  = mcap 文件名(去后缀)
    source_episode_seq  int64   = 该 mcap 内的 X/Y 对序号
    num_samples         int64   = len(samples), 冗余存一列, 使最终编号阶段不必读 samples 这根巨大的列
    samples             LIST<STRUCT<topic: STRUCT<msg_type, msg_def, log_time, locations>>>

episode_offset 不存进这张原始表: 它属于 finalize 阶段, 由 build_sample_idx.py 用 DuckDB
在最终 snapshot 上按 (source_id, source_episode_seq) 排序生成。

数据按 "机器人/任务" 两级切分: catalog 名即当前机器人(mcap_config 里选定的 ROBOT),
namespace 名由该 mcap 的 prompt(任务)派生, 故一台机器人的多个任务各占一个 namespace。
不合成一张大表是因为 samples 的 STRUCT schema 由机器人的 topic 集合决定, 混表会逼出并集
schema; 而跨任务的全局编号只依赖元数据列, 由 build_sample_idx.py 在索引阶段统一完成。

用法::
    table = load_episodes_table(warehouse_dir, topic_names, prompt)
    with IcebergEpisodeSaver(table, topic_names) as saver:
        saver.append(EpisodeRecord(source_id, source_episode_seq, samples))
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.table import Table

from openpi.training.lingyu_dataloader_v2.mcap_config.config import ROBOT

# Constants
BATCH_EPISODES = 16          # 每多少个 episode 触发一次 append/commit
EPISODES_TABLE_NAME = "episodes"

# 一条 message 的定位主键, 字段名与 MCAP_Player 产出的 location 四元组逐位对应
LOCATION_TYPE = pa.struct([
    pa.field("mcap_url", pa.string(), nullable=False),
    pa.field("chunk_file_offset", pa.int64(), nullable=False),
    pa.field("uncompressed_byte_offset", pa.int64(), nullable=False),
    pa.field("record_length", pa.int64(), nullable=False),
])

# 一个 topic 在某个 sample 上选中的 message; locations 条数随角色而异(obs 一段 GOP, action 一个 chunk)
TOPIC_MESSAGE_TYPE = pa.struct([
    pa.field("msg_type", pa.string(), nullable=False),
    pa.field("msg_def", pa.string(), nullable=False),
    pa.field("log_time", pa.int64(), nullable=False),
    pa.field("locations", pa.list_(LOCATION_TYPE), nullable=False),
])


def build_episodes_schema(topic_names: Sequence[str]) -> pa.Schema:
    """Build the Iceberg/Arrow schema of one episode row for the given topics."""
    # 一个 sample 是固定 topic 集合的 STRUCT; 某 topic 在某 sample 缺失时写 None, 故 nullable
    sample_type = pa.struct([
        pa.field(topic_name, TOPIC_MESSAGE_TYPE, nullable=True)
        for topic_name in topic_names
    ])
    return pa.schema([
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_episode_seq", pa.int64(), nullable=False),
        pa.field("num_samples", pa.int64(), nullable=False),
        pa.field("samples", pa.list_(sample_type), nullable=False),
    ])


def prompt_to_namespace(prompt: str) -> str:
    """Turn a prompt into a valid Iceberg namespace, e.g. "Fold the towels" -> "fold_the_towels"."""
    return re.sub(r"[^0-9a-z]+", "_", prompt.lower()).strip("_")


def load_episodes_table(warehouse_dir: str, topic_names: Sequence[str], prompt: str) -> Table:
    """Load (creating on first use) the episodes table of one prompt (task) of the current robot."""
    # catalog 由本函数而非 saver 负责, saver 只拿 Table, 与 sqlite/REST/S3 等环境解耦
    warehouse_path = Path(warehouse_dir)
    warehouse_path.mkdir(parents=True, exist_ok=True)
    # 同一个 catalog.db 用 catalog 名区分机器人, 故换机器人不必另建 warehouse 目录
    catalog = load_catalog(
        ROBOT,
        **{"type": "sql",
           "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
           "warehouse": f"file://{warehouse_path}"},
    )
    namespace = prompt_to_namespace(prompt)
    # namespace 名规范化后不可逆, 故把原始 prompt 原文存进 namespace 属性, 训练时可取回
    catalog.create_namespace_if_not_exists(namespace, properties={"prompt": prompt})
    return catalog.create_table_if_not_exists(
        f"{namespace}.{EPISODES_TABLE_NAME}", schema=build_episodes_schema(topic_names))


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode as produced by MCAPSampleExtractor.iter_episodes()."""
    source_id: str
    source_episode_seq: int
    # [{topic: [msg_type, msg_def, log_time, [location, ...]]}, ...], 下标即 sample_idx
    samples: list[dict[str, Any]]

    @property
    def episode_id(self) -> str:
        """Stable unique identity, so a retried commit can be de-duplicated later."""
        return f"{self.source_id}:{self.source_episode_seq:08d}"


def _message_to_arrow(message: Sequence[Any]) -> dict[str, Any]:
    """Turn one [msg_type, msg_def, log_time, locations] record into named Arrow fields."""
    msg_type, msg_def, log_time, locations = message
    return {
        "msg_type": msg_type,
        "msg_def": msg_def,
        "log_time": int(log_time),   # log_time 为字符串纳秒, int64 原样存下不损失精度
        "locations": [
            # 给四元组各位起名, 之后 SQL 里可以直接写 location.mcap_url
            dict(zip(LOCATION_TYPE.names, (url, int(chunk), int(offset), int(length))))
            for url, chunk, offset, length in locations
        ],
    }


def _episode_to_row(episode: EpisodeRecord, topic_names: Sequence[str]) -> dict[str, Any]:
    """Turn one EpisodeRecord into one Arrow row matching build_episodes_schema()."""
    return {
        "episode_id": episode.episode_id,
        "source_id": episode.source_id,
        "source_episode_seq": int(episode.source_episode_seq),
        "num_samples": len(episode.samples),
        # 每个 sample 都补齐同一套 topic 键, 缺的写 None, 保证所有行的 STRUCT schema 完全一致
        "samples": [
            {topic_name: (_message_to_arrow(sample[topic_name])
                          if topic_name in sample else None)
             for topic_name in topic_names}
            for sample in episode.samples
        ],
    }


class IcebergEpisodeSaver:
    """Buffer episodes in memory and commit them to Iceberg BATCH_EPISODES at a time."""

    def __init__(self, table: Table, topic_names: Sequence[str],
                 batch_episodes: int = BATCH_EPISODES):
        assert batch_episodes > 0, f"batch_episodes 必须为正: {batch_episodes}"
        self._table = table
        self._topic_names = tuple(topic_names)
        self._batch_episodes = batch_episodes
        self._arrow_schema = build_episodes_schema(self._topic_names)

        self._buffer: list[EpisodeRecord] = []
        self._lock = Lock()         # 只保护内存 buffer; 耗时的 commit 在锁外做
        self.num_episodes = 0       # 已 commit 的 episode 总数
        self.num_commits = 0        # 已产生的 snapshot 数

    def append(self, episode: EpisodeRecord) -> None:
        """Add one episode, committing a batch once the buffer is full."""
        with self._lock:
            self._buffer.append(episode)
            batch = None
            if len(self._buffer) >= self._batch_episodes:
                batch, self._buffer = self._buffer, []
        if batch:
            self._commit_batch(batch)

    def flush(self) -> None:
        """Commit the remaining episodes that did not fill a whole batch."""
        with self._lock:
            batch, self._buffer = self._buffer, []
        if batch:
            self._commit_batch(batch)

    def _commit_batch(self, episodes: Sequence[EpisodeRecord]) -> None:
        """Write one batch as a single Arrow table -> one Iceberg snapshot."""
        arrow_table = pa.Table.from_pylist(
            [_episode_to_row(episode, self._topic_names) for episode in episodes],
            schema=self._arrow_schema,
        )
        self._table.append(arrow_table, snapshot_properties={
            "writer": "lingyu-episode-saver",
            "episode-count": str(len(episodes)),
        })
        self.num_episodes += len(episodes)
        self.num_commits += 1

    def __enter__(self) -> "IcebergEpisodeSaver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # 出错时不 flush: 半截 batch 留在内存里作废, 避免把不完整的采集结果写成 snapshot
        if exc_type is None:
            self.flush()
