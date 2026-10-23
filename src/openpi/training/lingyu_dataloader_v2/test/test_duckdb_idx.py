"""Read episode_index.parquet directly with DuckDB, verifying the global index is built and reverse-searchable.

There is no duckdb database file in the pipeline: DuckDB is just a query engine whose product is episode_index.parquet,
while catalog.db is Iceberg's own sqlite catalog.

Checks::
    1. Columns complete: prompt/episode_id/source_id/source_episode_seq/num_samples/data_file/row_group/
       episode_offset/sample_offset
    2. episode_offset is contiguous numbering 0..N-1, no holes, no duplicates
    3. sample_offset == the prefix sum of preceding episodes' num_samples
    4. Fetched by episode_offset, (prompt, source_id, source_episode_seq) increases monotonically
    5. episode_id is unique within one prompt (retry-duplicated rows in Iceberg have been folded)
    6. Cross-check with Iceberg: the index's episode/sample counts == the actual totals of each prompt table
    7. Reverse lookup: global sample number -> (prompt, episode_id, local sample_idx) -> that message in Iceberg
"""
from pathlib import Path

import duckdb
import pytest
from pyiceberg.expressions import EqualTo

from openpi.training.lingyu_dataloader_v2.build_sample_idx import GLOBAL_INDEX_NAME, WAREHOUSE_DIR
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    EPISODES_TABLE_NAME, load_episodes_catalog, prompt_to_namespace)
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_INDEX_COLUMNS = ["prompt", "episode_id", "source_id", "source_episode_seq",
                  "num_samples", "data_file", "row_group", "episode_offset", "sample_offset"]


def test():
    """Read the global index, verify numbering self-consistency and agreement with Iceberg, and actually reverse-look-up one message."""
    index_path = Path(WAREHOUSE_DIR, GLOBAL_INDEX_NAME)
    if not index_path.exists():
        pytest.skip(f"没有 {index_path}, 请先运行 build_sample_idx.py")

    index_rows = duckdb.sql(
        f"SELECT * FROM '{index_path}' ORDER BY episode_offset").fetchall()
    index_columns = duckdb.sql(f"SELECT * FROM '{index_path}' LIMIT 0").columns
    index = [dict(zip(index_columns, row)) for row in index_rows]
    total_samples = sum(entry["num_samples"] for entry in index)
    logger.info(f"全局索引 {len(index)} 个 episode, {total_samples} 个 sample, "
                f"{len(set(entry['prompt'] for entry in index))} 个 prompt")

    # --- 1. Columns complete, and both numbers are integers (DuckDB's SUM is HUGEINT, degrading to DOUBLE in parquet) ---
    assert list(index_columns) == _INDEX_COLUMNS, f"索引列与设计不符: {index_columns}"
    assert index, "索引是空的, 采集或编号阶段没产出"
    assert all(isinstance(index[0][column], int) for column in ("episode_offset", "sample_offset")), \
        f"编号不是整数, 下游按编号取数会退化成浮点: {type(index[0]['sample_offset'])}"

    # --- 2. episode_offset is contiguous from 0 ---
    assert [entry["episode_offset"] for entry in index] == list(range(len(index))), \
        "episode_offset 不是从 0 起的连续编号"

    # --- 3. sample_offset is a prefix sum ---
    expected_sample_offset = 0
    for entry in index:
        assert entry["sample_offset"] == expected_sample_offset, \
            f"{entry['episode_id']} 的 sample_offset 与累加值不符"
        expected_sample_offset += entry["num_samples"]
    assert expected_sample_offset == total_samples, "sample_offset 累加后与 sample 总数不符"

    # --- 4. Sort keys are monotonic ---
    identities = [(entry["prompt"], entry["source_id"], entry["source_episode_seq"])
                  for entry in index]
    assert identities == sorted(identities), \
        "索引未按 (prompt, source_id, source_episode_seq) 排序"

    # --- 5. episode_id is unique within a prompt ---
    assert len(set((entry["prompt"], entry["episode_id"]) for entry in index)) == len(index), \
        "同一 prompt 内出现重复 episode_id, 去重没生效"

    # --- 6. Cross-check with Iceberg: a whole missing table is invisible inside the index alone ---
    catalog = load_episodes_catalog(WAREHOUSE_DIR)
    prompt_tables = {
        catalog.load_namespace_properties(namespace)["prompt"]:
            catalog.load_table(f"{namespace}.{EPISODES_TABLE_NAME}")
        for (namespace,) in catalog.list_namespaces()
    }
    for prompt, table in prompt_tables.items():
        # Scan metadata columns only; the huge samples column is never read
        table_meta = table.scan(selected_fields=("episode_id", "num_samples")).to_arrow()
        # Iceberg may contain retry-duplicated rows, so deduplicate by episode_id before comparing
        unique_meta = dict(zip(table_meta["episode_id"].to_pylist(),
                               table_meta["num_samples"].to_pylist()))
        indexed = {entry["episode_id"]: entry["num_samples"]
                   for entry in index if entry["prompt"] == prompt}
        assert indexed == unique_meta, \
            f"prompt={prompt!r} 的表与索引对不上: 表 {len(unique_meta)} 个 episode, " \
            f"索引 {len(indexed)} 个"
        logger.info(f"[{prompt_to_namespace(prompt)}] 表与索引一致: {len(indexed)} 个 episode, "
                    f"{sum(indexed.values())} 个 sample")

    # --- 7. Reverse lookup: the first, middle and last global sample numbers must all land in a unique episode, with a real message fetched ---
    # The purpose of this step goes into the log: the index counts as usable only if a message is really fetched from Iceberg
    logger.info("反查链路: 全局 sample 编号 -> 索引定位 (prompt, episode_id, 局部 sample_idx) "
                "-> 回 Iceberg 取出那条真实 message")
    for global_sample_idx in (0, total_samples // 2, total_samples - 1):
        located = duckdb.sql(
            f"SELECT prompt, episode_id, {global_sample_idx} - sample_offset AS sample_idx "
            f"FROM '{index_path}' "
            f"WHERE sample_offset <= {global_sample_idx} "
            f"  AND {global_sample_idx} < sample_offset + num_samples").fetchall()
        assert len(located) == 1, f"全局 sample {global_sample_idx} 反查到 {len(located)} 个 episode"
        prompt, episode_id, sample_idx = located[0]

        episode_row = prompt_tables[prompt].scan(
            row_filter=EqualTo("episode_id", episode_id), limit=1).to_arrow().to_pylist()[0]
        sample = episode_row["samples"][sample_idx]
        topic_name, message = next((topic, msg) for topic, msg in sample.items() if msg)
        logger.info(f"全局 sample {global_sample_idx} -> {prompt_to_namespace(prompt)}/"
                    f"{episode_id} 的第 {sample_idx} 个 sample; {topic_name} "
                    f"log_time={message['log_time']} locations={len(message['locations'])} 条, "
                    f"首条={message['locations'][0]}")


if __name__ == "__main__":
    test()
