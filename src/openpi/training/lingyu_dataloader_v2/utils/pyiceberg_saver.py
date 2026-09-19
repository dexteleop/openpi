# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""把 MCAPSampleExtractor 产出的一批 episode 写成一个 Iceberg snapshot。

保存链路, 两阶段, 跨进程::
    worker 进程  EpisodeParquetWriter.write(): 一个 episode -> 一行 -> 一个 row group
                                               攒够 roll_episodes 行就关文件, 交出路径
    主进程      IcebergEpisodeSaver.extend(): table.add_files(路径) -> 一次 snapshot

之所以拆成两阶段: Arrow 转换占了整个保存开销的 95% 且是持 GIL 的纯 Python, 多线程实测
零加速(1.00x), 只有放进 worker 进程才真并行; add_files 只登记文件不重写数据, 主进程一次
调用仅 0.12 s。队列里因此只需要传文件路径, 不必把 episode 数据 pickle 回主进程。
之所以逐个 episode 增量写而不是攒一批再写: worker 的内存占用就此与文件行数无关, 恒等于
一个 episode(实测 RSS 平稳在 +0.11 GB), 于是文件想攒多大都行, commit 次数可以压得很低。

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
    # worker 进程里
    writer = EpisodeParquetWriter(table, topic_names, roll_episodes)
    closed = writer.write(EpisodeRecord(...))   # 写满一个文件时返回 (路径, episode 数)
    closed = writer.close()                     # 收尾, 交出最后那个不满的文件
    # 主进程里
    IcebergEpisodeSaver(table).extend([path], num_episodes)

一个 parquet 攒多少个 episode 由调用方决定, 见 build_sample_idx.EPISODES_PER_PARQUET。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.table import Table

from openpi.training.lingyu_dataloader_v2.mcap_config.config import ROBOT

# Constants
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


def load_episodes_catalog(warehouse_dir: str) -> Catalog:
    """Open (creating the warehouse dir on first use) the current robot's sqlite-backed catalog."""
    # catalog 由本函数而非 saver 负责, saver 只拿 Table, 与 sqlite/REST/S3 等环境解耦
    warehouse_path = Path(warehouse_dir)
    warehouse_path.mkdir(parents=True, exist_ok=True)
    # 同一个 catalog.db 用 catalog 名区分机器人, 故换机器人不必另建 warehouse 目录
    return load_catalog(
        ROBOT,
        **{"type": "sql",
           "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
           "warehouse": f"file://{warehouse_path}"},
    )


def load_episodes_table(warehouse_dir: str, topic_names: Sequence[str], prompt: str) -> Table:
    """Load (creating on first use) the episodes table of one prompt (task) of the current robot."""
    catalog = load_episodes_catalog(warehouse_dir)
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


class EpisodeParquetWriter:
    """Append episodes to one parquet file, rolling to a new file every roll_episodes.

    Meant to run inside the worker process: from_pylist is pure Python and holds the
    GIL, so it only goes parallel across processes, never across threads.
    One episode is one row, written as its own row group, so the worker only ever holds
    a single episode in memory no matter how many rows the file ends up with.
    A file is not part of the table until the parent registers it via add_files().
    """

    def __init__(self, table: Table, topic_names: Sequence[str], roll_episodes: int):
        assert roll_episodes > 0, f"roll_episodes 必须为正: {roll_episodes}"
        self._table = table
        self._topic_names = tuple(topic_names)
        self._roll_episodes = roll_episodes
        self._arrow_schema = build_episodes_schema(self._topic_names)

        self._writer: pq.ParquetWriter | None = None
        self._output_stream = None
        self._path = ""
        self._num_episodes = 0      # 当前文件里已写入的 episode 数

    def write(self, episode: EpisodeRecord) -> tuple[str, int] | None:
        """Write one episode; return (path, episode count) when the file just rolled."""
        if self._writer is None:
            # uuid 文件名: 几十个 worker 同时写同一个 data 目录, 靠它保证互不覆盖
            self._path = f"{self._table.location().rstrip('/')}/data/{uuid4()}.parquet"
            self._output_stream = self._table.io.new_output(self._path).create(overwrite=False)
            self._writer = pq.ParquetWriter(self._output_stream, self._arrow_schema,
                                            compression="zstd")
        # 一行一个 row group: 训练时按 episode 取数, 读一个 row group 就够, 无读放大
        self._writer.write_table(pa.Table.from_pylist(
            [_episode_to_row(episode, self._topic_names)], schema=self._arrow_schema))
        self._num_episodes += 1
        return self.close() if self._num_episodes >= self._roll_episodes else None

    def close(self) -> tuple[str, int] | None:
        """Finish the current file (if any) and return (path, episode count)."""
        if self._writer is None:
            return None
        self._writer.close()
        self._output_stream.close()
        self._writer, self._output_stream = None, None
        # 先取走再归零: 下一个 episode 会开一个新文件
        closed = (self._path, self._num_episodes)
        self._num_episodes = 0
        return closed


class IcebergEpisodeSaver:
    """Register worker-written parquet files into the table, one snapshot per call."""

    def __init__(self, table: Table):
        self._table = table
        self.num_episodes = 0       # 已注册的 episode 总数
        self.num_commits = 0        # 已产生的 snapshot 数

    def extend(self, parquet_paths: Sequence[str], num_episodes: int) -> None:
        """Register already-written parquet files as one Iceberg snapshot."""
        # add_files 只登记文件不重写数据; 路径带 uuid 不可能重复, 故关掉那次全表扫描
        self._table.add_files(
            file_paths=list(parquet_paths),
            snapshot_properties={"writer": "lingyu-episode-saver",
                                 "episode-count": str(num_episodes)},
            check_duplicate_files=False,
        )
        self.num_episodes += num_episodes
        self.num_commits += 1
