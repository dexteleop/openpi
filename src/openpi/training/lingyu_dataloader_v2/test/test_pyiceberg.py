"""Open the Iceberg warehouse written by build_sample_idx.py directly and verify the on-disk structure matches the design.

This test is read-only and does not re-extract mcaps, so collection must have finished first::
    python -m openpi.training.lingyu_dataloader_v2.build_sample_idx 2>&1 | tee test/build_sample_idx.log

Checks::
    1. catalog name == current ROBOT; one namespace per prompt, one episodes table per namespace
    2. namespace attributes hold the prompt text, self-consistent with the namespace name (equal after normalization)
    3. Table schema: 5 top-level fields; the topic set in samples == the current robot's selected topics
    4. One snapshot per EPISODES_PER_PARQUET episodes: each snapshot has at most that many rows, with short tails
       at most one per worker (files continue across mcaps and are truncated only when a worker finishes)
       (EPISODES_PER_PARQUET is defined in build_sample_idx.py, same source as the value actually used at collection)
    5. Read one row back: num_samples matches len(samples), and the location quadruples carry field names
"""
from pathlib import Path

import pytest

from openpi.training.lingyu_dataloader_v2.build_sample_idx import EPISODES_PER_PARQUET, WAREHOUSE_DIR
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    EPISODES_TABLE_NAME, LOCATION_TYPE, load_episodes_catalog, prompt_to_namespace)
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_TOP_LEVEL_FIELDS = ["episode_id", "source_id", "source_episode_seq", "num_samples", "samples"]


def test():
    """Open the warehouse and verify, table by table: namespace / schema / snapshot batching / one row's content."""
    if not Path(WAREHOUSE_DIR, "catalog.db").exists():
        pytest.skip(f"{WAREHOUSE_DIR} 里没有 catalog.db, 请先运行 build_sample_idx.py")

    catalog = load_episodes_catalog(WAREHOUSE_DIR)
    namespaces = catalog.list_namespaces()
    logger.info(f"catalog={catalog.name}, {len(namespaces)} 个 namespace(prompt)")
    assert namespaces, "catalog 里一个 namespace 都没有, 采集可能没落盘"

    topic_names = sorted(filter_topics())
    total_episodes = 0
    for (namespace,) in namespaces:
        # --- 1/2. One prompt one namespace one table; the prompt text lives in namespace attributes ---
        table_ids = catalog.list_tables(namespace)
        assert table_ids == [(namespace, EPISODES_TABLE_NAME)], \
            f"{namespace} 下的表不是唯一的 {EPISODES_TABLE_NAME}: {table_ids}"
        prompt = catalog.load_namespace_properties(namespace).get("prompt")
        assert prompt and prompt_to_namespace(prompt) == namespace, \
            f"{namespace} 的 prompt 属性缺失或与 namespace 名不自洽: {prompt!r}"

        table = catalog.load_table(f"{namespace}.{EPISODES_TABLE_NAME}")

        # --- 3. Schema: 5 top-level fields; topics in samples match the current robot's selection ---
        assert [field.name for field in table.schema().fields] == _TOP_LEVEL_FIELDS, \
            f"{namespace} 的顶层字段与设计不符"

        # --- 4. Snapshot batching: at most EPISODES_PER_PARQUET rows per batch ---
        # Worker files continue across mcaps and truncate only when the worker finishes, so short-tail snapshots are at most one per worker
        snapshots = table.inspect.snapshots().to_pylist()
        batch_sizes = [int(dict(snapshot["summary"])["episode-count"]) for snapshot in snapshots]
        assert all(size <= EPISODES_PER_PARQUET for size in batch_sizes), \
            f"{namespace} 存在超过 {EPISODES_PER_PARQUET} 个 episode 的 snapshot: {batch_sizes}"
        # The rows actually written per snapshot must also equal episode-count, preventing summary/data mismatch
        added_records = [int(dict(snapshot["summary"])["added-records"]) for snapshot in snapshots]
        assert added_records == batch_sizes, f"{namespace} 的 added-records 与 episode-count 不符"

        # --- 5. Read one row back: only one row, but the whole samples column is in it, structure usable as-is ---
        first_row = table.scan(limit=1).to_arrow().to_pylist()[0]
        sample_struct = table.scan(limit=1).to_arrow().schema.field("samples").type.value_type
        assert sorted(field.name for field in sample_struct) == topic_names, \
            f"{namespace} 的 samples STRUCT 与当前机器人选定的 topic 不一致"
        assert first_row["num_samples"] == len(first_row["samples"]), \
            f"{namespace} 的 num_samples 与 samples 实际长度不符"
        first_message = first_row["samples"][0][topic_names[0]]
        assert list(first_message["locations"][0]) == list(LOCATION_TYPE.names), \
            "locations 四元组没有带上字段名"

        table_episodes = sum(batch_sizes)
        total_episodes += table_episodes
        logger.info(f"[{namespace}] prompt={prompt!r} {table_episodes} 个 episode, "
                    f"{len(snapshots)} 次 snapshot, 分批={batch_sizes}")
        logger.info(f"[{namespace}] 首行 {first_row['episode_id']}: "
                    f"{first_row['num_samples']} 个 sample, {len(topic_names)} 个 topic, "
                    f"首条 message locations={first_message['locations'][0]}")

    logger.info(f"warehouse 共 {total_episodes} 个 episode, "
                f"分布在 {len(namespaces)} 个 prompt 的表中")


if __name__ == "__main__":
    test()
