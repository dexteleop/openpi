"""直接打开 build_sample_idx.py 落盘的 Iceberg warehouse, 核对盘上结构与设计一致。

本测试只读, 不重新抽取 mcap, 故须先跑完 build_sample_idx.py(补建 mcap 索引 + 登记 + 编号)::
    python -m openpi.training.lingyu_dataloader_v2.build_sample_idx

核对::
    1. catalog 名 = 当前机器人 ROBOT; 每个 prompt 一个 namespace, 每个 namespace 一张 episodes 表
    2. namespace 属性里存着 prompt 原文, 且与 namespace 名自洽(规范化后相等)
    3. 表 schema: 5 个顶层字段, samples 里的 topic 集合 == 当前机器人选定的 topic
    4. 一张表只有一次 snapshot, 登记的数据文件数 == 选中的 mcap 数; 数据文件都是桶内的
       s3://.../xxx.episodes.parquet, 即一个 mcap 一个索引, 未被拷贝进本地 warehouse
    5. 抽一行回读: num_samples 与 samples 长度一致, locations 四元组已带上字段名
"""
from pathlib import Path

import pytest

from openpi.training.lingyu_dataloader_v2.build_sample_idx import WAREHOUSE_DIR
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    EPISODES_TABLE_NAME, LOCATION_TYPE, MCAP_INDEX_SUFFIX, load_episodes_catalog, prompt_to_namespace)
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_TOP_LEVEL_FIELDS = ["episode_id", "source_id", "source_episode_seq", "num_samples", "samples"]


def test():
    """打开 warehouse, 逐张表核对 namespace / schema / 一次登记的 mcap 索引 / 一行内容。"""
    if not Path(WAREHOUSE_DIR, "catalog.db").exists():
        pytest.skip(f"{WAREHOUSE_DIR} 里没有 catalog.db, 请先运行 build_sample_idx.py")

    catalog = load_episodes_catalog(WAREHOUSE_DIR)
    namespaces = catalog.list_namespaces()
    logger.info(f"catalog={catalog.name}, {len(namespaces)} 个 namespace(prompt)")
    assert namespaces, "catalog 里一个 namespace 都没有, 采集可能没落盘"

    topic_names = sorted(filter_topics())
    total_episodes = 0
    for (namespace,) in namespaces:
        # --- 1/2. 一个 prompt 一个 namespace 一张表, prompt 原文存在 namespace 属性里 ---
        table_ids = catalog.list_tables(namespace)
        assert table_ids == [(namespace, EPISODES_TABLE_NAME)], \
            f"{namespace} 下的表不是唯一的 {EPISODES_TABLE_NAME}: {table_ids}"
        prompt = catalog.load_namespace_properties(namespace).get("prompt")
        assert prompt and prompt_to_namespace(prompt) == namespace, \
            f"{namespace} 的 prompt 属性缺失或与 namespace 名不自洽: {prompt!r}"

        table = catalog.load_table(f"{namespace}.{EPISODES_TABLE_NAME}")

        # --- 3. schema: 顶层 5 个字段, samples 里的 topic 与当前机器人选定的一致 ---
        assert [field.name for field in table.schema().fields] == _TOP_LEVEL_FIELDS, \
            f"{namespace} 的顶层字段与设计不符"

        # --- 4. 一次 snapshot 登记全部选中 mcap 的索引, 数据文件都在桶里 ---
        snapshots = table.inspect.snapshots().to_pylist()
        assert len(snapshots) == 1, f"{namespace} 应只有一次登记 snapshot, 实际 {len(snapshots)} 次"
        summary = dict(snapshots[0]["summary"])
        assert int(summary["added-data-files"]) == int(summary["mcap-count"]), \
            f"{namespace} 登记的数据文件数与 mcap 数不符: {summary}"
        # plan_files 会跳过 0 行的索引(该 mcap 没有合格 episode), 故只校验路径形式
        data_files = [scan_task.file.file_path for scan_task in table.scan().plan_files()]
        assert all(path.startswith("s3://") and path.endswith(MCAP_INDEX_SUFFIX) for path in data_files), \
            f"{namespace} 存在不是桶内 mcap 索引的数据文件: {data_files[:3]}"

        # --- 5. 回读一行: 只取一行, 但 samples 整列都在这一行里, 结构原样可用 ---
        first_row = table.scan(limit=1).to_arrow().to_pylist()[0]
        sample_struct = table.scan(limit=1).to_arrow().schema.field("samples").type.value_type
        assert sorted(field.name for field in sample_struct) == topic_names, \
            f"{namespace} 的 samples STRUCT 与当前机器人选定的 topic 不一致"
        assert first_row["num_samples"] == len(first_row["samples"]), \
            f"{namespace} 的 num_samples 与 samples 实际长度不符"
        first_message = first_row["samples"][0][topic_names[0]]
        assert list(first_message["locations"][0]) == list(LOCATION_TYPE.names), \
            "locations 四元组没有带上字段名"

        table_episodes = int(summary["added-records"])
        total_episodes += table_episodes
        logger.info(f"[{namespace}] prompt={prompt!r} {table_episodes} 个 episode, "
                    f"{summary['mcap-count']} 个 mcap 索引, 其中 {len(data_files)} 个非空")
        logger.info(f"[{namespace}] 首行 {first_row['episode_id']}: "
                    f"{first_row['num_samples']} 个 sample, {len(topic_names)} 个 topic, "
                    f"首条 message locations={first_message['locations'][0]}")

    logger.info(f"warehouse 共 {total_episodes} 个 episode, "
                f"分布在 {len(namespaces)} 个 prompt 的表中")


if __name__ == "__main__":
    test()
