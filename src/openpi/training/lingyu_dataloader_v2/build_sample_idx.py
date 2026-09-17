# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""并行抽取所有 mcap 的 episode 存进 Iceberg, 再用 DuckDB 生成全局索引。

采集阶段::
    MAX_WORKERS 个进程, 一个进程一个 mcap (进程池 + spawn, 绕过 GIL)
        每个进程内 MCAPSampleExtractor.iter_episodes() 产出全部 episode
        -> 进程结束时把列表 pickle 回主进程
    主进程单写线程 IcebergEpisodeSaver
        -> 每 BATCH_EPISODES 个 episode 一次 snapshot

用进程而非线程: MCAP 反序列化是纯 Python CPU 密集型操作, 线程受 GIL 限制无法真正并行。
用单写线程而非每进程各写: sqlite catalog 不支持多进程并发 commit; 进程只传 episode 列表,
主进程持有唯一写者, 无锁竞争。
进程间 pickle 开销: 相对于每个 mcap 13 GB 的解析时间, 传输开销比例很小。

一个 prompt(任务)一张 Iceberg 表, 由 find_mcap_paths() 返回的 {mcap 路径: prompt} 决定,
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
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from queue import Queue
from threading import Thread

import duckdb
import pyarrow as pa

from openpi.training.lingyu_dataloader_v2.mcap_sample_extractor import MCAPSampleExtractor
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    BATCH_EPISODES, EpisodeRecord, IcebergEpisodeSaver, load_episodes_table)
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_paths import find_mcap_paths
from openpi.training.lingyu_dataloader_v2.utils.topics_filter import filter_topics

# Constants
MAX_WORKERS = 64                # 最多有 64 个 mcap 在并行抽取
# 待写入的 episode 上限, 满则采集线程阻塞。一个 episode 上千个 sample、内存里上百 MB,
# 故只留一个批次的余量: 一批在 commit 时, 最多再攒一批等着。
QUEUE_CAPACITY = BATCH_EPISODES
WAREHOUSE_DIR = str(Path(__file__).parent / "iceberg_warehouse")
GLOBAL_INDEX_NAME = "episode_index.parquet"
_QUEUE_END = object()           # 采集全部结束的哨兵, 通知写线程收工

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
       SUM(num_samples) OVER episode_order - num_samples   AS sample_offset
FROM unique_episodes
WINDOW episode_order AS (
    ORDER BY prompt, source_id, source_episode_seq
    ROWS UNBOUNDED PRECEDING
)
ORDER BY episode_offset
"""
_INDEX_META_FIELDS = ("episode_id", "source_id", "source_episode_seq", "num_samples")

logger = logging.getLogger(__name__)


def extract_episodes_for_mcap(mcap_url: str) -> list[EpisodeRecord]:
    """Replay one mcap in a child process and return all its complete episodes.

    Top-level (not nested) so it can be pickled by the spawn process pool.
    The whole episode list is pickled back to the parent once the mcap is done.
    """
    extractor = MCAPSampleExtractor(mcap_url)
    episodes = [
        EpisodeRecord(extractor.source_id, source_episode_seq, samples)
        for source_episode_seq, samples in extractor.iter_episodes()
    ]
    # log inside the child; the parent's logger will see it via StreamHandler
    logging.getLogger(__name__).info(
        f"done: {mcap_url} ({extractor.num_episodes} episodes, "
        f"{extractor.total_samples} samples)")
    return episodes


def save_episodes_from_queue(episode_queue: Queue, savers: dict[str, IcebergEpisodeSaver],
                             commit_errors: list) -> None:
    """Drain the queue into the saver of each episode's prompt until the end sentinel arrives."""
    while True:
        item = episode_queue.get()
        if item is _QUEUE_END:
            return
        prompt, episode = item
        try:
            savers[prompt].append(episode)
        except Exception as commit_error:
            # commit 失败仍要继续排空队列, 否则 feeder 线程会卡死在满队列上; 错误交主线程抛出
            commit_errors.append(commit_error)


def build_episodes_table(mcap_prompts: dict[str, str], warehouse_dir: str = WAREHOUSE_DIR) -> dict:
    """Extract all mcaps in parallel worker processes and write episodes to Iceberg.

    mcap_prompts is find_mcap_paths()'s {mcap path: prompt}; one prompt is one table.
    Architecture: spawn process pool (breaks GIL for CPU-bound MCAP parsing) feeds
    a bounded queue; a single writer thread drains it into the per-prompt
    IcebergEpisodeSaver so the sqlite catalog is never written from two threads at once.
    Returns {prompt: Table}.
    """
    topic_names = sorted(filter_topics())   # 排序: 同一套 topic 每次都得到同一个 schema
    tables = {prompt: load_episodes_table(warehouse_dir, topic_names, prompt)
              for prompt in dict.fromkeys(mcap_prompts.values())}
    episode_queue = Queue(maxsize=QUEUE_CAPACITY)
    commit_errors: list = []

    # ExitStack 同时持有各 prompt 的 saver: 出错时它们一律不 flush, 与单 saver 时语义一致
    with ExitStack() as saver_stack:
        savers = {prompt: saver_stack.enter_context(IcebergEpisodeSaver(table, topic_names))
                  for prompt, table in tables.items()}
        writer_thread = Thread(target=save_episodes_from_queue,
                               args=(episode_queue, savers, commit_errors), daemon=True)
        writer_thread.start()

        mp_ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp_ctx) as pool:
            future_to_url = {pool.submit(extract_episodes_for_mcap, url): url
                             for url in mcap_prompts}
            for future in as_completed(future_to_url):
                mcap_url = future_to_url[future]
                try:
                    for episode in future.result():
                        # blocks if queue is full -> back-pressure
                        episode_queue.put((mcap_prompts[mcap_url], episode))
                except Exception as extract_error:
                    logger.error(f"error: {mcap_url}: {extract_error}")

        episode_queue.put(_QUEUE_END)
        writer_thread.join()        # 等写线程把队列排空, 之后 __exit__ 再 flush 尾批
        # commit 失败就此中断: __exit__ 不再 flush, 半截 batch 不会被写成 snapshot
        if commit_errors:
            raise commit_errors[0]

    logger.info(f"saved {sum(saver.num_episodes for saver in savers.values())} episodes in "
                f"{sum(saver.num_commits for saver in savers.values())} snapshots "
                f"to {warehouse_dir} ({len(savers)} prompts)")
    return tables


def build_global_index(tables: dict, warehouse_dir: str = WAREHOUSE_DIR) -> str:
    """Number every episode/sample of all prompts globally and write one episode_index.parquet."""
    # 各 prompt 表只取元数据列(samples 那根巨大的列完全不读), 补一列 prompt 后拼成一张 Arrow 表
    prompt_metas = []
    for prompt, table in tables.items():
        meta = table.scan(selected_fields=_INDEX_META_FIELDS).to_arrow()
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


def build_all(num_mcaps: int = 64, warehouse_dir: str = WAREHOUSE_DIR) -> str:
    """Extract the first num_mcaps mcaps into Iceberg, then build the global index."""
    # find_mcap_paths() 返回 {mcap 路径: prompt}, 截取时要连 prompt 一起留下
    mcap_prompts = dict(islice(find_mcap_paths().items(), num_mcaps))
    tables = build_episodes_table(mcap_prompts, warehouse_dir)
    return build_global_index(tables, warehouse_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_all()
