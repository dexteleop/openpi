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

最终编号阶段::
    Iceberg 元数据列 -> DuckDB -> ORDER BY (source_id, source_episode_seq) + ROW_NUMBER()
        -> episode_index.parquet: {episode_id, source_id, source_episode_seq, num_samples,
                                   episode_offset, sample_offset}
episode_offset 是全局连续的 episode 编号, sample_offset 是该 episode 首个 sample 的全局编号,
故全局 sample 编号 s 属于满足 sample_offset <= s < sample_offset + num_samples 的那个 episode,
其局部 sample_idx 为 s - sample_offset。
编号只读 Iceberg 的元数据列, 不碰 samples 这根巨大的列, 因此与数据量无关。
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from queue import Queue
from threading import Thread

import duckdb

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
    SELECT DISTINCT ON (episode_id)
           episode_id, source_id, source_episode_seq, num_samples
    FROM episodes_meta
)
SELECT episode_id,
       source_id,
       source_episode_seq,
       num_samples,
       ROW_NUMBER() OVER episode_order - 1                 AS episode_offset,
       SUM(num_samples) OVER episode_order - num_samples   AS sample_offset
FROM unique_episodes
WINDOW episode_order AS (
    ORDER BY source_id, source_episode_seq
    ROWS UNBOUNDED PRECEDING
)
ORDER BY episode_offset
"""

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


def save_episodes_from_queue(episode_queue: Queue, saver: IcebergEpisodeSaver,
                             commit_errors: list) -> None:
    """Drain the queue into the saver until the end sentinel arrives."""
    while True:
        episode = episode_queue.get()
        if episode is _QUEUE_END:
            return
        try:
            saver.append(episode)
        except Exception as commit_error:
            # commit 失败仍要继续排空队列, 否则 feeder 线程会卡死在满队列上; 错误交主线程抛出
            commit_errors.append(commit_error)


def build_episodes_table(mcap_urls: list[str], warehouse_dir: str = WAREHOUSE_DIR):
    """Extract all mcaps in parallel worker processes and write episodes to Iceberg.

    Architecture: spawn process pool (breaks GIL for CPU-bound MCAP parsing) feeds
    a bounded queue; a single writer thread drains it into IcebergEpisodeSaver so
    the sqlite catalog is never written from two threads at once.
    """
    topic_names = sorted(filter_topics())   # 排序: 同一套 topic 每次都得到同一个 schema
    table = load_episodes_table(warehouse_dir, topic_names)
    episode_queue = Queue(maxsize=QUEUE_CAPACITY)
    commit_errors: list = []

    with IcebergEpisodeSaver(table, topic_names) as saver:
        writer_thread = Thread(target=save_episodes_from_queue,
                               args=(episode_queue, saver, commit_errors), daemon=True)
        writer_thread.start()

        mp_ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp_ctx) as pool:
            future_to_url = {pool.submit(extract_episodes_for_mcap, url): url
                             for url in mcap_urls}
            for future in as_completed(future_to_url):
                try:
                    for episode in future.result():
                        episode_queue.put(episode)  # blocks if queue is full -> back-pressure
                except Exception as extract_error:
                    logger.error(f"error: {future_to_url[future]}: {extract_error}")

        episode_queue.put(_QUEUE_END)
        writer_thread.join()        # 等写线程把队列排空, 之后 __exit__ 再 flush 尾批
        # commit 失败就此中断: __exit__ 不再 flush, 半截 batch 不会被写成 snapshot
        if commit_errors:
            raise commit_errors[0]

    logger.info(f"saved {saver.num_episodes} episodes in {saver.num_commits} snapshots "
                f"to {warehouse_dir}")
    return table


def build_global_index(table, warehouse_dir: str = WAREHOUSE_DIR) -> str:
    """Number every episode/sample globally with DuckDB and write episode_index.parquet."""
    # 只取元数据列, samples 那根巨大的列完全不读
    episodes_meta = table.scan(selected_fields=(
        "episode_id", "source_id", "source_episode_seq", "num_samples")).to_arrow()
    index_path = str(Path(warehouse_dir) / GLOBAL_INDEX_NAME)

    with duckdb.connect() as duck_conn:
        duck_conn.register("episodes_meta", episodes_meta)
        duck_conn.execute(
            f"COPY ({_GLOBAL_INDEX_SQL}) TO '{index_path}' (FORMAT PARQUET)")
        total_episodes, total_samples = duck_conn.execute(
            f"SELECT count(*), sum(num_samples) FROM ({_GLOBAL_INDEX_SQL})").fetchone()

    logger.info(f"global index: {total_episodes} episodes, {total_samples} samples "
                f"-> {index_path}")
    return index_path


def build_all(num_mcaps: int = 64, warehouse_dir: str = WAREHOUSE_DIR) -> str:
    """Extract the first num_mcaps mcaps into Iceberg, then build the global index."""
    table = build_episodes_table(find_mcap_paths()[:num_mcaps], warehouse_dir)
    return build_global_index(table, warehouse_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_all()
