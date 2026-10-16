# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""把各 mcap 的 episode parquet(索引)登记进 Iceberg, 再用 DuckDB 生成全局索引。

索引由数据平台产出(构建脚本已迁至 platform-operator), 保证桶内每个选中的 mcap 同级目录下
都有 a/rec_0.episodes.parquet (见 pyiceberg_saver.mcap_index_key); 训练流程里只运行本脚本。
索引不含 mcap 位置信息, mcap 的位置由登记时的 data_file(s3://桶/key) 反推。

就绪检查阶段::
    主进程多线程只读各 parquet 的 footer: 缺失的记 ERROR 并跳过(提示索引由数据平台
    补产出), 不让单个坏文件阻断登记与编号。

登记阶段::
    每个 prompt 重建一张空表 -> add_files(该 prompt 选中 mcap 的 parquet) -> 一次 snapshot
    只登记不重写数据, 换一批 mcap 只需重跑本阶段; 选中的 mcap 都已有索引时, 整个准备过程
    只读 footer 与元数据列, 不回放任何 mcap。

最终编号阶段::
    各 prompt 表当前 snapshot 的数据文件 -> 逐文件只读元数据列, 并记下每行所在 (data_file, row_group)
        -> 拼成一张 Arrow 表 -> DuckDB
        -> ORDER BY (prompt, source_id, source_episode_seq) + ROW_NUMBER()
        -> episode_index.parquet: {prompt, episode_id, source_id, source_episode_seq,
                                   num_samples, data_file, row_group, episode_offset, sample_offset}

下载阶段(编号后自动执行, 构建流程的一环而非选项):: 全局索引登记的全部索引 parquet 从 minio
    拷到本地缓存(warehouse/local_parquet/); minio 对 GetObject 高并发限流(429)且 pyarrow
    读 parquet 遇 429 不重试直接崩, 本地化后取样只剩 mcap 字节读取打 minio(带重试)。

episode_offset 是全局连续的 episode 编号, sample_offset 是该 episode 首个 sample 的全局编号,
故全局 sample 编号 s 属于满足 sample_offset <= s < sample_offset + num_samples 的那个 episode,
其局部 sample_idx 为 s - sample_offset。
(data_file, row_group) 是该 episode 的物理位置(data_file 为 s3:// 路径): 写入端一个 episode 一行一个
row group, 故行号即 row group 号。取样时凭它直接 read_row_group, 不再经 Iceberg scan 逐个 manifest/footer 去找。
编号跨全部 prompt 只产出一份索引: Iceberg 的 namespace 不参与编号, prompt 列即定位表的坐标。
编号只读元数据列, 不碰 samples 这根巨大的列, 因此与数据量无关。

示例命令：
    cd ~/openpi/src/openpi/training/lingyu_dataloader_v2
    python build_sample_idx.py                # 登记 + 编号
    python build_sample_idx.py --index-only   # 只用 catalog 里已登记的表重跑编号阶段
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

from botocore.exceptions import ClientError, IncompleteReadError, ReadTimeoutError, ResponseStreamingError

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.fs import S3FileSystem

from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    S3_READ_ATTEMPTS, S3_THROTTLE_STATUS, s3_client)
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    EPISODES_TABLE_NAME, create_episodes_table, load_episodes_catalog, mcap_index_key)
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
    BUCKET_NAME, find_mcap_url_and_prompt_pairs, make_arrow_filesystem)
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics

# Constants
INDEX_CHECK_WORKERS = 32        # 主进程并行读 mcap 索引(footer/元数据列)的线程数, 纯网络 IO
WAREHOUSE_DIR = str(Path(__file__).parent / "iceberg_warehouse")
# 索引 parquet 的本地缓存目录名(在 warehouse 下), 取样时优先本地读取
PARQUET_CACHE_NAME = "local_parquet"
GLOBAL_INDEX_NAME = "episode_index.parquet"
LOG_PATH = str(Path(__file__).with_suffix(".log"))
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# 编号阶段从数据文件里读的元数据列, samples 那根巨大的列不读
EPISODE_META_COLUMNS = ["episode_id", "source_id", "source_episode_seq", "num_samples"]

# 全局编号: 先按 episode_id 去重(commit 重试可能写重, 任取一份的位置即可), 再按 source 稳定排序连续编号。
# 绝不用 parquet 行位置或 Iceberg commit 顺序当编号依据, 那两者都会随重跑而变;
# (data_file, row_group) 只是随编号一起存下的物理位置, 不参与排序。
_GLOBAL_INDEX_SQL = """
WITH unique_episodes AS (
    SELECT DISTINCT ON (prompt, episode_id)
           prompt, episode_id, source_id, source_episode_seq, num_samples, data_file, row_group
    FROM episodes_meta
)
SELECT prompt,
       episode_id,
       source_id,
       source_episode_seq,
       num_samples,
       data_file,
       row_group,
       ROW_NUMBER() OVER episode_order - 1                 AS episode_offset,
       -- SUM(BIGINT) 在 DuckDB 里是 HUGEINT, 落 parquet 会变成 DOUBLE, 故显式收回 BIGINT
       CAST(SUM(num_samples) OVER episode_order - num_samples AS BIGINT) AS sample_offset
FROM unique_episodes
WINDOW episode_order AS (
    ORDER BY prompt, source_id, source_episode_seq
    ROWS UNBOUNDED PRECEDING
)
ORDER BY episode_offset
"""

logger = logging.getLogger(__name__)


def check_mcap_indexes_ready(mcap_urls: list[str]) -> list[str]:
    """读各索引 footer, 只保留存在的 mcap; 缺失的记 ERROR 并跳过"""
    arrow_fs = make_arrow_filesystem()
    ready_urls = []
    for mcap_url in mcap_urls:
        try:
            index_schema = pq.read_schema(f"{BUCKET_NAME}/{mcap_index_key(mcap_url)}", filesystem=arrow_fs)
        except FileNotFoundError:
            logger.error(f"[索引缺失] {mcap_url}, 请检查数据平台(platform-operator)是否已产出索引")
            continue
        ready_urls.append(mcap_url)
    return ready_urls


def register_mcap_indexes(mcap_prompts: dict[str, str], topic_names: list,
                          warehouse_dir: str = WAREHOUSE_DIR) -> dict:
    """每个 prompt 重建一张空表, 把该 prompt 选中 mcap 的索引一次 add_files 登记为一个 snapshot。

    mcap_prompts 是 {mcap url: prompt}, 一个 prompt 一张表; 返回 {prompt: Table}。
    """
    tables = {}
    for prompt in dict.fromkeys(mcap_prompts.values()):
        index_urls = [f"s3://{BUCKET_NAME}/{mcap_index_key(mcap_url)}"
                      for mcap_url, mcap_prompt in mcap_prompts.items() if mcap_prompt == prompt]
        table = create_episodes_table(warehouse_dir, topic_names, prompt)
        # add_files 只登记文件不重写数据; 表刚重建, 不可能有重复文件, 故关掉那次全表扫描
        table.add_files(file_paths=index_urls,
                        snapshot_properties={"writer": "lingyu-mcap-index", "mcap-count": str(len(index_urls))},
                        check_duplicate_files=False)
        logger.info(f"[{prompt}] 登记 {len(index_urls)} 个 mcap 索引")
        tables[prompt] = table
    return tables


def build_global_index(tables: dict, warehouse_dir: str = WAREHOUSE_DIR) -> str:
    """Number every episode/sample of all prompts globally and write one episode_index.parquet."""
    # 各 prompt 表只取元数据列与物理位置(samples 那根巨大的列完全不读), 拼成一张 Arrow 表
    arrow_fs = make_arrow_filesystem()
    episodes_meta = pa.concat_tables(
        [load_episodes_meta(prompt, table, arrow_fs) for prompt, table in tables.items()])
    index_path = str(Path(warehouse_dir) / GLOBAL_INDEX_NAME)

    with duckdb.connect() as duck_conn:
        duck_conn.register("episodes_meta", episodes_meta)
        duck_conn.execute(
            f"COPY ({_GLOBAL_INDEX_SQL}) TO '{index_path}' (FORMAT PARQUET)")
        row = duck_conn.execute(
            f"SELECT count(*), sum(num_samples) FROM ({_GLOBAL_INDEX_SQL})").fetchone()
        assert row is not None, "global index query returned no rows"
        total_episodes, total_samples = row

    logger.info(f"global index: {total_episodes} episodes, {total_samples} samples "
                f"over {len(tables)} prompts -> {index_path}")
    # 索引构建的收尾步骤: 登记的索引 parquet 本地化(全量路径与 --index-only 路径都在此收敛)
    download_parquet_cache()
    return index_path


def load_file_meta(data_file: str, arrow_fs: S3FileSystem) -> pa.Table:
    """Read one mcap index's metadata columns, plus the row group each episode lives in."""
    assert data_file.startswith("s3://"), f"{data_file} 不在桶里, 是旧版本地数据文件, 请重跑 build_sample_idx.py"
    # pre_buffer 把各 row group 的小列块合并成少量 range 请求
    parquet_file = pq.ParquetFile(data_file.removeprefix("s3://"), filesystem=arrow_fs, pre_buffer=True)
    # 写入端一个 episode 一行一个 row group, 行号才等于 row group 号
    assert parquet_file.num_row_groups == parquet_file.metadata.num_rows, \
        f"{data_file} 不是一行一个 row group, 行号不能当 row group 号"
    file_meta = parquet_file.read(columns=EPISODE_META_COLUMNS)
    return (file_meta
            .append_column("data_file", pa.array([data_file] * file_meta.num_rows, pa.string()))
            .append_column("row_group", pa.array(range(file_meta.num_rows), pa.int64())))


def load_episodes_meta(prompt: str, table, arrow_fs: S3FileSystem) -> pa.Table:
    """Read one prompt table's metadata columns from all its data files in parallel.

    Returns columns EPISODE_META_COLUMNS + [data_file, row_group, prompt], one row per episode.
    """
    # plan_files 给出当前 snapshot 的全部数据文件(0 行的 mcap 索引已被跳过); add_files 写入的表没有 delete 文件
    data_files = [scan_task.file.file_path for scan_task in table.scan().plan_files()]
    assert data_files, f"{prompt!r} 的表里没有任何数据文件"
    # 每个文件都要读 footer 与几根小列, 纯网络 IO, 多线程并行; map 保持文件顺序
    with ThreadPoolExecutor(INDEX_CHECK_WORKERS) as executor:
        meta = pa.concat_tables(list(executor.map(load_file_meta, data_files, repeat(arrow_fs))))
    return meta.append_column("prompt", pa.array([prompt] * meta.num_rows, pa.string()))


def load_existing_tables(warehouse_dir: str = WAREHOUSE_DIR) -> dict:
    """Load every prompt's episodes table already in the catalog, as {prompt: Table}."""
    catalog = load_episodes_catalog(warehouse_dir)
    # 建表时 namespace 属性里存了 prompt 原文(namespace 名规范化后不可逆), 据此取回对应关系
    tables = {catalog.load_namespace_properties(namespace)["prompt"]:
              catalog.load_table((*namespace, EPISODES_TABLE_NAME))
              for namespace in catalog.list_namespaces()}
    assert tables, f"{warehouse_dir} 的 catalog 里没有任何表, 请先不带 --index-only 跑一次采集"
    return tables


def _download_index_parquet(data_file: str) -> int:
    """把一个桶内索引 parquet 下载到本地缓存, 返回字节数; 限流/超时按指数退避重试。

    先写 .part 临时文件再原子改名, 缓存里存在的文件必是完整的, 重跑即断点续传。
    """
    assert data_file.startswith("s3://"), f"{data_file} 不在桶里"
    bucket_name, object_key = data_file.removeprefix("s3://").split("/", 1)
    local_path = Path(WAREHOUSE_DIR, PARQUET_CACHE_NAME, object_key)
    if local_path.exists():
        return local_path.stat().st_size
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_suffix(".part")
    for attempt_idx in range(S3_READ_ATTEMPTS):
        try:
            response = s3_client().get_object(Bucket=bucket_name, Key=object_key)
            with open(temp_path, "wb") as temp_file:
                for data_chunk in response["Body"].iter_chunks(1024 * 1024):
                    temp_file.write(data_chunk)
            os.replace(temp_path, local_path)   # 原子改名, 半截文件不会留在缓存里
            return local_path.stat().st_size
        except (ReadTimeoutError, ResponseStreamingError, IncompleteReadError, ClientError) as error:
            # 与 fetch_mcap_bytes 相同的判定: 只重试限流/超时类, NoSuchKey 等其它错误照常抛出
            is_throttled = (not isinstance(error, ClientError) or
                            error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in S3_THROTTLE_STATUS)
            if not is_throttled or attempt_idx == S3_READ_ATTEMPTS - 1:
                raise
            # 指数退避 + 随机抖动, 避免多线程同时重试再次形成请求洪峰
            backoff_seconds = min(2 ** attempt_idx, 30) * random.uniform(0.5, 1.5)
            logger.warning(f"下载 {data_file} 失败({error}), {backoff_seconds:.1f}s 后第 {attempt_idx + 2} 次尝试")
            time.sleep(backoff_seconds)
    raise AssertionError("不可达: 重试循环必以 return 或 raise 结束")


def download_parquet_cache() -> None:
    """把全局索引登记的全部索引 parquet 多线程拷到本地缓存, 供训练取样时本地读取。

    minio 对 GetObject 高并发会限流(429), pyarrow 读 parquet 遇 429 又不重试直接崩,
    故取样前先把体量很小的索引 parquet 整体本地化, 训练只剩 mcap 字节读取打 minio。
    """
    index_path = Path(WAREHOUSE_DIR, GLOBAL_INDEX_NAME)
    assert index_path.exists(), f"没有 {index_path}, 请先运行 build_sample_idx.py"
    data_files = sorted({row[0] for row in duckdb.sql(
        f"SELECT DISTINCT data_file FROM '{index_path}'").fetchall()})
    download_start = time.time()
    with ThreadPoolExecutor(INDEX_CHECK_WORKERS) as executor:
        total_bytes = sum(executor.map(_download_index_parquet, data_files))
    logger.info(f"本地缓存 {len(data_files)} 个索引 parquet 共 {total_bytes / 1e9:.2f} GB "
                f"于 {time.time() - download_start:.1f} s 完成 -> {Path(WAREHOUSE_DIR, PARQUET_CACHE_NAME)}")


def build_all() -> str:
    """Check that per-mcap indexes are ready, register the selected ones into Iceberg, then number them."""
    start_time = time.time()
    mcap_prompts = find_mcap_url_and_prompt_pairs()
    topic_names = sorted(filter_topics())   # 排序: 同一套 topic 每次都得到同一个 schema
    ready_urls = check_mcap_indexes_ready(list(mcap_prompts))
    assert ready_urls, "没有任何 mcap 的索引就绪, 请检查数据平台(platform-operator)是否已产出索引"
    index_time = time.time()
    tables = register_mcap_indexes({mcap_url: mcap_prompts[mcap_url] for mcap_url in ready_urls}, topic_names)
    index_path = build_global_index(tables)
    logger.info(f"准备耗时: 查找 + 就绪检查 {index_time - start_time:.1f} s, "
                f"登记 + 编号 {time.time() - index_time:.1f} s")
    return index_path


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="把就绪的 mcap 索引登记进 Iceberg, 再生成全局索引并本地化索引 parquet")
    arg_parser.add_argument("--index-only", action="store_true",
                            help="跳过就绪检查与登记, 只用 catalog 里已有的表重建 episode_index.parquet")
    cli_args = arg_parser.parse_args()
    # 先删旧日志再以追加方式打开, 全部输出汇入同一个 .log 文件
    Path(LOG_PATH).unlink(missing_ok=True)
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_PATH, filemode="a",
                        encoding="utf-8")
    if cli_args.index_only:
        build_global_index(load_existing_tables())
    else:
        build_all()
