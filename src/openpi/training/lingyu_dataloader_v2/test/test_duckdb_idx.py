"""直接用 DuckDB 读 episode_index.parquet, 核对全局索引已构建且可反查。

链路里没有 duckdb 数据库文件: DuckDB 只是查询引擎, 产物是 episode_index.parquet,
catalog.db 则是 Iceberg 自己的 sqlite catalog。

核对::
    1. 列齐: prompt/episode_id/source_id/source_episode_seq/num_samples/data_file/row_group/
       episode_offset/sample_offset
    2. episode_offset 是 0..N-1 的连续编号, 无洞无重
    3. sample_offset == 前序 episode 的 num_samples 前缀和
    4. 按 episode_offset 取出后, (prompt, source_id, source_episode_seq) 单调递增
    5. 同一 prompt 内 episode_id 唯一(Iceberg 里的重试重复行已被折叠)
    6. 与 Iceberg 交叉校验: 索引的 episode 数/sample 数 == 各 prompt 表的实际总数
    7. 反查: 全局 sample 编号 -> (prompt, episode_id, 局部 sample_idx) -> Iceberg 里那条 message
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
    """读全局索引, 校验编号自洽、与 Iceberg 对得上, 并真的反查出一条 message。"""
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

    # --- 1. 列齐, 且两个编号都是整数(DuckDB 的 SUM 是 HUGEINT, 落 parquet 会退化成 DOUBLE) ---
    assert list(index_columns) == _INDEX_COLUMNS, f"索引列与设计不符: {index_columns}"
    assert index, "索引是空的, 采集或编号阶段没产出"
    assert all(isinstance(index[0][column], int) for column in ("episode_offset", "sample_offset")), \
        f"编号不是整数, 下游按编号取数会退化成浮点: {type(index[0]['sample_offset'])}"

    # --- 2. episode_offset 从 0 起连续 ---
    assert [entry["episode_offset"] for entry in index] == list(range(len(index))), \
        "episode_offset 不是从 0 起的连续编号"

    # --- 3. sample_offset 是前缀和 ---
    expected_sample_offset = 0
    for entry in index:
        assert entry["sample_offset"] == expected_sample_offset, \
            f"{entry['episode_id']} 的 sample_offset 与累加值不符"
        expected_sample_offset += entry["num_samples"]
    assert expected_sample_offset == total_samples, "sample_offset 累加后与 sample 总数不符"

    # --- 4. 排序键单调 ---
    identities = [(entry["prompt"], entry["source_id"], entry["source_episode_seq"])
                  for entry in index]
    assert identities == sorted(identities), \
        "索引未按 (prompt, source_id, source_episode_seq) 排序"

    # --- 5. episode_id 在 prompt 内唯一 ---
    assert len(set((entry["prompt"], entry["episode_id"]) for entry in index)) == len(index), \
        "同一 prompt 内出现重复 episode_id, 去重没生效"

    # --- 6. 与 Iceberg 交叉校验: 漏掉整张表在索引内部是看不出来的 ---
    catalog = load_episodes_catalog(WAREHOUSE_DIR)
    prompt_tables = {
        catalog.load_namespace_properties(namespace)["prompt"]:
            catalog.load_table(f"{namespace}.{EPISODES_TABLE_NAME}")
        for (namespace,) in catalog.list_namespaces()
    }
    for prompt, table in prompt_tables.items():
        # 只扫元数据列, samples 那根巨大的列完全不读
        table_meta = table.scan(selected_fields=("episode_id", "num_samples")).to_arrow()
        # Iceberg 里可能有重试写重的行, 故按 episode_id 去重后再比
        unique_meta = dict(zip(table_meta["episode_id"].to_pylist(),
                               table_meta["num_samples"].to_pylist()))
        indexed = {entry["episode_id"]: entry["num_samples"]
                   for entry in index if entry["prompt"] == prompt}
        assert indexed == unique_meta, \
            f"prompt={prompt!r} 的表与索引对不上: 表 {len(unique_meta)} 个 episode, " \
            f"索引 {len(indexed)} 个"
        logger.info(f"[{prompt_to_namespace(prompt)}] 表与索引一致: {len(indexed)} 个 episode, "
                    f"{sum(indexed.values())} 个 sample")

    # --- 7. 反查: 首、中、尾三个全局 sample 编号都要落回唯一 episode, 并取出真实 message ---
    # 这一步的目的写进日志: 索引是否可用, 只有真的从 Iceberg 里取出 message 才算数
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
