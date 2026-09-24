# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""并行抽取所有 mcap 的 episode 存进 Iceberg, 再用 DuckDB 生成全局索引。

采集阶段::
    mcap_queue: 主进程一次性填入全部 (mcap 路径, prompt), 末尾每个 worker 一个 None 哨兵
        -> MAX_MCAP_WORKERS 个常驻 worker 进程(spawn, 绕过 GIL)各自取一个 mcap
           worker 内 MCAPSampleExtractor.iter_episodes() 每产出一个 episode 就增量写进
           当前 parquet(一行一个 row group), 攒满 EPISODES_PER_PARQUET 行才换新文件
    parquet_queue: 里面只有写完文件的路径, 不含 episode 数据, 故无需限长
        -> 主进程单消费循环, 每收到一个文件就 saver.extend() 注册成一个 snapshot

worker 不抽完整个 mcap 再返回，逐个 episode 增量写之后, 每个
worker 的内存恒等于一个 episode, 与文件行数无关, 故 worker 数可以放心开大。
Arrow 转换放 worker 而非主进程: 它占保存开销的 95% 且是持 GIL 的纯 Python, 多线程实测
零加速(1.00x), 只有跨进程才真并行; 顺带队列里也不必再 pickle episode 数据。
主进程只做 add_files: 登记文件不重写数据, 单次 0.12 s, 不会成为瓶颈。

一个 prompt(任务)一张 Iceberg 表, 由 find_mcap_urls() 返回的 {mcap url: prompt} 决定,
故进程池是全局共享的, 队列元素带上 prompt, 单写线程再按 prompt 分派给对应的 saver。

最终编号阶段::
    各 prompt 表的元数据列 -> 拼成一张 Arrow 表 -> DuckDB
        -> ORDER BY (prompt, source_id, source_episode_seq) + ROW_NUMBER()
        -> episode_index.parquet: {prompt, episode_id, source_id, source_episode_seq,
                                   num_samples, episode_offset, sample_offset}
episode_offset 是全局连续的 episode 编号, sample_offset 是该 episode 首个 sample 的全局编号,
故全局 sample 编号 s 属于满足 sample_offset <= s < sample_offset + num_samples 的那个 episode,
其局部 sample_idx 为 s - sample_offset。
编号跨全部 prompt 只产出一份索引: Iceberg 的 namespace 不参与编号, prompt 列即定位表的坐标。
编号只读 Iceberg 的元数据列, 不碰 samples 这根巨大的列, 因此与数据量无关。

示例命令：
    cd ~/openpi/src/openpi/training/lingyu_dataloader_v2
    python build_sample_idx.py > build_sample_idx.log 2>&1 &
    tail -f build_sample_idx.log
"""
from __future__ import annotations

import logging
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

import duckdb
import pyarrow as pa

from openpi.training.lingyu_dataloader_v2.mcap_sample_extractor import MCAPSampleExtractor
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    EpisodeParquetWriter, EpisodeRecord, IcebergEpisodeSaver, load_episodes_table)
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import find_mcap_url_and_prompt_pairs
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics

# Constants
MAX_MCAP_WORKERS = 128          # 最多 MCAP 并行抽取数量
# 每个 parquet 攒多少个 episode(即一次 add_files/一个 snapshot)。只决定文件大小与 commit
# 次数, 不影响内存: worker 是逐个 episode 增量写的, 实测约 0.29 MB/episode。
EPISODES_PER_PARQUET = 100
WAREHOUSE_DIR = str(Path(__file__).parent / "iceberg_warehouse")
GLOBAL_INDEX_NAME = "episode_index.parquet"
# 哨兵要跨进程传, 必须用 pickle 后仍可按值比较的对象, 不能用 object() —— 它 pickle 后身份就变了
_WORKER_DONE = "__worker_done__"    # worker 领到 None 收工时送出, 主进程据此计数
# 主进程等 episode 的超时: 超时就查一次 worker 存活, 免得 worker 被 OOM killer 杀掉(送不出
# _WORKER_DONE)时主进程永远挂在 get() 上
QUEUE_POLL_SEC = 60

# 全局编号: 先按 episode_id 去重(commit 重试可能写重), 再按 source 稳定排序连续编号。
# 绝不用 parquet 行位置或 Iceberg commit 顺序当编号依据, 那两者都会随重跑而变。
_GLOBAL_INDEX_SQL = """
WITH unique_episodes AS (
    SELECT DISTINCT ON (prompt, episode_id)
           prompt, episode_id, source_id, source_episode_seq, num_samples
    FROM episodes_meta
)
SELECT prompt,
       episode_id,
       source_id,
       source_episode_seq,
       num_samples,
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


def extract_episodes_to_parquet(mcap_queue, parquet_queue, warehouse_dir: str,
                                topic_names: list) -> None:
    """Take mcaps off mcap_queue, append episodes to parquet, push each full file's path.

    Top-level (not nested) so the spawn worker process can import it.
    Runs until it draws the None sentinel; one bad mcap only loses that mcap.
    """
    child_logger = logging.getLogger(__name__)
    # prompt -> writer, 一个 prompt 一个在写的文件; 跨 mcap 续写, 故文件总能攒满
    writers: dict[str, EpisodeParquetWriter] = {}
    while True:
        task = mcap_queue.get()
        if task is None:
            for prompt, writer in writers.items():      # 收工: 交出最后那些不满的文件
                _report_closed_file(parquet_queue, prompt, writer.close())
            parquet_queue.put(_WORKER_DONE)
            return
        mcap_url, prompt = task
        if prompt not in writers:
            table = load_episodes_table(warehouse_dir, topic_names, prompt)
            writers[prompt] = EpisodeParquetWriter(table, topic_names, EPISODES_PER_PARQUET)
        try:
            extractor = MCAPSampleExtractor(mcap_url)
            for source_episode_seq, samples in extractor.iter_episodes():
                # 写一行就落一个 row group, 本进程内存恒等于一个 episode
                _report_closed_file(parquet_queue, prompt, writers[prompt].write(
                    EpisodeRecord(extractor.source_id, source_episode_seq, samples)))
            # log inside the child; the parent's logger will see it via StreamHandler
            child_logger.info(f"done: {mcap_url} ({extractor.num_episodes} episodes, "
                              f"{extractor.total_samples} samples)")
        except Exception as extract_error:
            child_logger.error(f"error: {mcap_url}: {extract_error}")


def _report_closed_file(parquet_queue, prompt: str, closed_file: tuple | None) -> None:
    """Hand a just-finished parquet file to the parent; do nothing while it is still open."""
    if closed_file is not None:
        parquet_path, num_episodes = closed_file
        parquet_queue.put((prompt, parquet_path, num_episodes))


def register_parquets_from_queue(parquet_queue, savers: dict[str, IcebergEpisodeSaver],
                                 workers: list) -> None:
    """Register every parquet file the workers report as one Iceberg snapshot."""
    num_done_workers = 0
    while num_done_workers < len(workers):
        try:
            item = parquet_queue.get(timeout=QUEUE_POLL_SEC)
        except Empty:
            # 只有 worker 全死光才收工: 被 OOM killer 杀掉的 worker 送不出 _WORKER_DONE
            if not any(worker.is_alive() for worker in workers):
                logger.error("worker 已全部退出但哨兵不齐, 可能有进程被杀, 提前收尾")
                break
            continue
        if item == _WORKER_DONE:
            num_done_workers += 1
            continue
        prompt, parquet_path, num_episodes = item
        # 逐个文件注册而非全部写完再一次性 add_files: 中途崩了已注册的部分仍在表里
        savers[prompt].extend([parquet_path], num_episodes)


def build_episodes_table(mcap_prompts: dict[str, str], warehouse_dir: str = WAREHOUSE_DIR) -> dict:
    """Extract all mcaps in parallel worker processes and write episodes to Iceberg.

    mcap_prompts is find_mcap_urls()'s {mcap url: prompt}; one prompt is one table.
    Architecture: resident spawn worker processes parse mcaps and write parquet files
    (both steps are GIL-bound, so they only parallelise across processes); the parent
    is the single consumer and only registers those files, one snapshot per file.
    Returns {prompt: Table}.
    """
    topic_names = sorted(filter_topics())   # 排序: 同一套 topic 每次都得到同一个 schema
    # 主进程先建好表, worker 里的 load_episodes_table() 就只是只读加载, 不会并发建表
    tables = {prompt: load_episodes_table(warehouse_dir, topic_names, prompt)
              for prompt in dict.fromkeys(mcap_prompts.values())}
    savers = {prompt: IcebergEpisodeSaver(table) for prompt, table in tables.items()}

    mp_ctx = get_context("spawn")
    mcap_queue = mp_ctx.Queue()
    parquet_queue = mp_ctx.Queue()
    num_workers = min(MAX_MCAP_WORKERS, len(mcap_prompts))
    for mcap_url, prompt in mcap_prompts.items():
        mcap_queue.put((mcap_url, prompt))
    for _ in range(num_workers):
        mcap_queue.put(None)        # 一个 worker 一个收工哨兵
    workers = [mp_ctx.Process(target=extract_episodes_to_parquet,
                              args=(mcap_queue, parquet_queue, warehouse_dir, topic_names),
                              daemon=True)
               for _ in range(num_workers)]
    for worker in workers:
        worker.start()

    register_parquets_from_queue(parquet_queue, savers, workers)

    for worker in workers:
        worker.join()
    killed_exitcodes = [worker.exitcode for worker in workers if worker.exitcode]
    if killed_exitcodes:
        logger.error(f"{len(killed_exitcodes)} 个 worker 非正常退出, "
                     f"exitcode={killed_exitcodes} (-9 即被 OOM killer 杀掉)")

    logger.info(f"saved {sum(saver.num_episodes for saver in savers.values())} episodes in "
                f"{sum(saver.num_commits for saver in savers.values())} snapshots "
                f"to {warehouse_dir} ({len(savers)} prompts)")
    return tables


def build_global_index(tables: dict, warehouse_dir: str = WAREHOUSE_DIR) -> str:
    """Number every episode/sample of all prompts globally and write one episode_index.parquet."""
    # 各 prompt 表只取元数据列(samples 那根巨大的列完全不读), 补一列 prompt 后拼成一张 Arrow 表
    prompt_metas = []
    for prompt, table in tables.items():
        meta = table.scan(selected_fields=(
            "episode_id", "source_id", "source_episode_seq", "num_samples")).to_arrow()
        prompt_metas.append(
            meta.append_column("prompt", pa.array([prompt] * meta.num_rows, pa.string())))
    episodes_meta = pa.concat_tables(prompt_metas)
    index_path = str(Path(warehouse_dir) / GLOBAL_INDEX_NAME)

    with duckdb.connect() as duck_conn:
        duck_conn.register("episodes_meta", episodes_meta)
        duck_conn.execute(
            f"COPY ({_GLOBAL_INDEX_SQL}) TO '{index_path}' (FORMAT PARQUET)")
        total_episodes, total_samples = duck_conn.execute(
            f"SELECT count(*), sum(num_samples) FROM ({_GLOBAL_INDEX_SQL})").fetchone()

    logger.info(f"global index: {total_episodes} episodes, {total_samples} samples "
                f"over {len(tables)} prompts -> {index_path}")
    return index_path


def build_all() -> str:
    """Extract the first num_mcaps mcaps into Iceberg, then build the global index."""
    # find_mcap_urls() 返回 {mcap url: prompt}, 截取时要连 prompt 一起留下
    mcap_prompts = dict(islice(find_mcap_url_and_prompt_pairs().items(), MAX_MCAP_WORKERS))
    tables = build_episodes_table(mcap_prompts, WAREHOUSE_DIR)
    return build_global_index(tables, WAREHOUSE_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_all()
