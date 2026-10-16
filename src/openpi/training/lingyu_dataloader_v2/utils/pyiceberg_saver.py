# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""一个 mcap 对应一个 episode parquet(索引), 再由 Iceberg 登记选中 mcap 的那些 parquet。

保存链路(构建脚本已迁至数据平台 platform-operator, 本仓库只负责登记)::
    登记  Iceberg: table.add_files(选中 mcap 的 parquet 路径) -> 一次 snapshot
          add_files 只登记文件不重写数据, 故选一批 mcap 只需秒级, 不必再回放 mcap

一行就是一个 episode::
    episode_id          string  = "{source_id}:{source_episode_seq:08d}", 全局稳定唯一, 供重试去重
    source_id           string  = mcap 文件名(去后缀)
    source_episode_seq  int64   = 该 mcap 内的 X/Y 对序号
    num_samples         int64   = len(samples), 冗余存一列, 使最终编号阶段不必读 samples 这根巨大的列
    samples             LIST<STRUCT<topic: STRUCT<msg_type, msg_def, log_time, locations>>>

索引不含任何 mcap 位置信息(LOCATION_TYPE 只有 chunk 内偏移): mcap 与其索引同目录同名存放,
未来一起搬到桶内任意位置, 索引内容都无需重建; mcap 的位置由全局索引的 data_file 反推。

episode_offset 不存进这张原始表: 它属于 finalize 阶段, 由 build_sample_idx.py 用 DuckDB
在最终 snapshot 上按 (source_id, source_episode_seq) 排序生成。

数据按 "机器人/任务" 两级切分: catalog 名即当前机器人(mcap_config 里选定的 ROBOT),
namespace 名由该 mcap 的 prompt(任务)派生, 故一台机器人的多个任务各占一个 namespace。
不合成一张大表是因为 samples 的 STRUCT schema 由机器人的 topic 集合决定, 混表会逼出并集
schema; 而跨任务的全局编号只依赖元数据列, 由 build_sample_idx.py 在索引阶段统一完成。
catalog(sqlite) 与表元数据在本地 warehouse, 数据文件即桶内的 episode parquet(s3://), 不拷贝到本地。

用法::
    table = create_episodes_table(warehouse_dir, topic_names, prompt)
    table.add_files(["s3://桶/xxx.episodes.parquet", ...])
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pyarrow as pa
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.table import Table

from openpi.training.lingyu_dataloader_v2.mcap_config.config import ROBOT
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import iceberg_s3_properties

# Constants
EPISODES_TABLE_NAME = "episodes"
# 索引与 mcap 同目录同名, 只换后缀: a/rec_0.mcap -> a/rec_0.episodes.parquet; 构建与登记两端共用
MCAP_INDEX_SUFFIX = ".episodes.parquet"


def mcap_index_key(mcap_url: str) -> str:
    """mcap 的桶内 key -> 同目录索引 parquet 的 key"""
    assert mcap_url.endswith(".mcap"), f"不是 mcap 文件: {mcap_url}"
    return mcap_url.removesuffix(".mcap") + MCAP_INDEX_SUFFIX


def mcap_key_from_index(index_key: str) -> str:
    """索引 parquet 的路径 -> 同目录同名 mcap 的桶内 key, 即 mcap_index_key() 的反函数。

    入参可为 Iceberg 登记的 data_file(s3://桶/key) 或纯桶内 key; 训练侧取样时凭它反推
    mcap 位置, 索引内容因此不必携带任何 mcap 位置信息。
    """
    if index_key.startswith("s3://"):
        index_key = index_key.removeprefix("s3://").split("/", 1)[-1]   # 去掉桶名
    assert index_key.endswith(MCAP_INDEX_SUFFIX), f"不是索引 parquet 路径: {index_key}"
    return index_key.removesuffix(MCAP_INDEX_SUFFIX) + ".mcap"

# 一条 message 在 mcap 内部的定位三元组(不含 mcap 位置: mcap 未来会在桶内不同位置出现,
# 索引与 mcap 同目录存放, 训练时由 data_file 的 key 反推同目录同名 mcap 的位置)
LOCATION_TYPE = pa.struct([
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
    # 表元数据落本地 warehouse, 数据文件在桶里(s3://), 故 FileIO 还要带上 S3 连接属性
    return load_catalog(
        ROBOT,
        **{"type": "sql",
           "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
           "warehouse": f"file://{warehouse_path}",
           **iceberg_s3_properties()},
    )


def create_episodes_table(warehouse_dir: str, topic_names: Sequence[str], prompt: str) -> Table:
    """(Re)create the empty episodes table of one prompt (task) of the current robot."""
    catalog = load_episodes_catalog(warehouse_dir)
    namespace = prompt_to_namespace(prompt)
    # namespace 名规范化后不可逆, 故把原始 prompt 原文存进 namespace 属性, 训练时可取回
    catalog.create_namespace_if_not_exists(namespace, properties={"prompt": prompt})
    table_name = f"{namespace}.{EPISODES_TABLE_NAME}"
    # 表里只该有本次选中的 mcap: 先删旧表再重建。sql catalog 的 drop_table 只删登记, 不碰桶里的 parquet
    if catalog.table_exists(table_name):
        catalog.drop_table(table_name)
    # 索引 parquet 不写列统计(数据平台产出端如此写), 指标模式设为 none: add_files 不再逐 row group 逐列
    # 收集统计, 否则每个文件要刷几千条 "statistics missing" WARNING, 登记 48 个 mcap 由 2 s 拖到 18 s
    return catalog.create_table(table_name, schema=build_episodes_schema(topic_names),
                                properties={"write.metadata.metrics.default": "none"})
