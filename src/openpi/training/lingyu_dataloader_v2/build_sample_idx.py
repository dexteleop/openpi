# Copyright (c) 2026 Dexteleop Intelligence (Lingyu Intelligence)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""Extract episodes from all mcaps into Iceberg in parallel, then build the global index with DuckDB.

Collection stage::
    mcap_queue: the main process fills in all (mcap path, prompt) at once, followed by one None sentinel per worker
        -> MAX_MCAP_WORKERS resident worker processes (spawn, bypassing the GIL) each take one mcap
           inside a worker, MCAPSampleExtractor.iter_episodes() incrementally writes each produced episode
           into the current parquet (one row per row group), switching to a new file only after EPISODES_PER_PARQUET rows
    parquet_queue: contains only paths of finished files, no episode data, so no length limit is needed
        -> the main process's single consumer loop registers each received file as one snapshot via saver.extend()

A worker does not finish a whole mcap before returning; by writing each episode incrementally, every
worker's memory stays constant at one episode regardless of file row count, so the worker count can safely be large.
Arrow conversion runs in workers rather than the main process: it accounts for 95% of the save cost and is pure Python holding
the GIL -- measured multithreading speedup was zero (1.00x); only processes truly parallelize, and it also avoids pickling episode data through queues.
The main process only does add_files: registering files rewrites no data, 0.12 s each, never a bottleneck.

One Iceberg table per prompt (task), decided by the {mcap url: prompt} returned by find_mcap_urls(),
so the process pool is globally shared; queue elements carry the prompt, and the single writer thread dispatches to the matching saver by prompt.

Final numbering stage::
    data files of each prompt table's current snapshot -> read only the metadata columns file by file, recording each row's (data_file, row_group)
        -> concatenated into one Arrow table -> DuckDB
        -> ORDER BY (prompt, source_id, source_episode_seq) + ROW_NUMBER()
        -> episode_index.parquet: {prompt, episode_id, source_id, source_episode_seq,
                                   num_samples, data_file, row_group, episode_offset, sample_offset}
episode_offset is the globally contiguous episode number; sample_offset is the global number of the episode's first sample,
so a global sample number s belongs to the episode satisfying sample_offset <= s < sample_offset + num_samples,
and its local sample_idx is s - sample_offset.
(data_file, row_group) is the episode's physical location: the writer emits one episode per row group, so the row number
is the row group number. Sampling uses it to read_row_group directly instead of hunting through Iceberg manifests/footers.
Numbering across all prompts produces a single index: Iceberg namespaces take no part in numbering; the prompt column is the table's coordinates.
Numbering reads only metadata columns and never touches the huge samples column, so it is independent of data volume.

Example commands:
    cd ~/openpi/src/openpi/training/lingyu_dataloader_v2
    python build_sample_idx.py                # collect + number, no log file generated
    python build_sample_idx.py --log          # same, and write run logs to build_sample_idx.log
    python build_sample_idx.py --index-only   # data already in Iceberg, rerun only the numbering stage

Watch per-process traffic on NIC enp6s18 in real time
    sudo nethogs -v 4 enp6s18   # displayed directly in MB/s
"""
from __future__ import annotations

import argparse
import logging
import resource
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.lingyu_dataloader_v2.mcap_sample_extractor import MCAPSampleExtractor
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    EPISODES_TABLE_NAME, EpisodeParquetWriter, EpisodeRecord, IcebergEpisodeSaver,
    load_episodes_catalog, load_episodes_table)
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import find_mcap_url_and_prompt_pairs
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics

# Constants
MAX_MCAP_WORKERS = 128          # Maximum number of mcaps extracted in parallel
# How many episodes each parquet accumulates (i.e. one add_files / one snapshot). Only affects file size and commit
# count, not memory: workers write episode by episode, measured at about 0.29 MB/episode.
EPISODES_PER_PARQUET = 100
WAREHOUSE_DIR = str(Path(__file__).parent / "iceberg_warehouse")
GLOBAL_INDEX_NAME = "episode_index.parquet"
# Main process and spawn workers share the same log file and level: spawn children do not inherit the main process's logging config
LOG_PATH = str(Path(__file__).with_suffix(".log"))
LOG_LEVEL = logging.INFO
# With timestamp and process number: 128 workers write one file; pid stitches start/episode/done into one timeline
LOG_FORMAT = "%(asctime)s %(processName)s[%(process)d] %(levelname)s %(name)s: %(message)s"
# Sentinels cross process boundaries, so they must be objects that remain equal by value after pickling; object() does not work -- its identity changes after pickling
_WORKER_DONE = "__worker_done__"    # Sent when a worker receives None and finishes; the main process counts these
# Timeout for the main process waiting for episodes: on timeout it checks worker liveness once, so the main process does not hang on get() forever
# when a worker is killed by the OOM killer (and can never send _WORKER_DONE)
QUEUE_POLL_SEC = 60
# Metadata columns read from data files during the numbering stage; the huge samples column is not read
EPISODE_META_COLUMNS = ["episode_id", "source_id", "source_episode_seq", "num_samples"]

# Global numbering: first deduplicate by episode_id (commit retries may write duplicates; any copy's location works), then stable-sort by source for contiguous numbering.
# Never use parquet row positions or Iceberg commit order as numbering bases -- both change across reruns;
# (data_file, row_group) is just the physical location stored along with the number and takes no part in sorting.
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


def extract_episodes_to_parquet(mcap_queue, parquet_queue, warehouse_dir: str,
                                topic_names: list, log_enabled: bool) -> None:
    """Take mcaps off mcap_queue, append episodes to parquet, push each full file's path.

    Top-level (not nested) so the spawn worker process can import it.
    Runs until it draws the None sentinel; one bad mcap only loses that mcap.
    """
    # spawn workers are separate execution entries and must configure logging themselves; with --log only, append to the same file the main process truncated
    # force=True: third-party libraries may already have attached handlers to the root logger at import time; without force, basicConfig silently does nothing
    if log_enabled:
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_PATH, filemode="a",
                            encoding="utf-8", force=True)
        # Keep WARNING and above; silence the extractor's Start/Stop INFO
        logging.getLogger("openpi.training.lingyu_dataloader_v2.mcap_sample_extractor").setLevel(logging.WARNING)
    # prompt -> writer, one in-progress file per prompt; writes continue across mcaps so files always fill up
    writers: dict[str, EpisodeParquetWriter] = {}
    while True:
        task = mcap_queue.get()
        if task is None:
            for prompt, writer in writers.items():      # Wrapping up: hand over the last not-yet-full files
                _report_closed_file(parquet_queue, prompt, writer.close())
            parquet_queue.put(_WORKER_DONE)
            return
        mcap_url, prompt = task
        # A process killed by the OOM killer writes no logs at all; only a "start without done" can identify which mcap it was
        logger.info(f"start: {mcap_url}")
        try:
            # Table/writer creation is also inside the try: otherwise its exception only goes to stderr, invisible in the log file
            if prompt not in writers:
                table = load_episodes_table(warehouse_dir, topic_names, prompt)
                writers[prompt] = EpisodeParquetWriter(table, topic_names, EPISODES_PER_PARQUET)
            extractor = MCAPSampleExtractor(mcap_url)
            for source_episode_seq, samples in extractor.iter_episodes():
                # Flush one row group per row written; this process's memory stays constant at one episode
                record = EpisodeRecord(extractor.source_id, source_episode_seq, samples)
                _report_closed_file(parquet_queue, prompt, writers[prompt].write(record))
                # One heartbeat line per episode: proves the process is alive and exposes whether memory keeps growing
                logger.info(f"episode {record.episode_id} ({len(samples)} samples), "
                            f"peak_rss={_peak_rss_gb():.2f}GB")
            logger.info(f"done: {mcap_url} ({extractor.num_episodes} episodes, "
                        f"{extractor.total_samples} samples), peak_rss={_peak_rss_gb():.2f}GB")
        except Exception:
            logger.exception(f"error: {mcap_url}")      # exception() includes the full traceback


def _peak_rss_gb() -> float:
    """Peak resident memory of the current process in GB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


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
            # Only finish when all workers are dead: an OOM-killed worker can never send _WORKER_DONE
            if not any(worker.is_alive() for worker in workers):
                logger.error("worker 已全部退出但哨兵不齐, 可能有进程被杀, 提前收尾")
                break
            continue
        if item == _WORKER_DONE:
            num_done_workers += 1
            continue
        prompt, parquet_path, num_episodes = item
        # Register file by file instead of one add_files after everything is written: if we crash midway, the registered files remain in the table
        savers[prompt].extend([parquet_path], num_episodes)


def build_episodes_table(mcap_prompts: dict[str, str], warehouse_dir: str = WAREHOUSE_DIR,
                         log_enabled: bool = False) -> dict:
    """Extract all mcaps in parallel worker processes and write episodes to Iceberg.

    mcap_prompts is find_mcap_urls()'s {mcap url: prompt}; one prompt is one table.
    Architecture: resident spawn worker processes parse mcaps and write parquet files
    (both steps are GIL-bound, so they only parallelise across processes); the parent
    is the single consumer and only registers those files, one snapshot per file.
    Returns {prompt: Table}.
    """
    topic_names = sorted(filter_topics())   # Sorted: the same topic set yields the same schema every time
    # The main process creates the tables first, so load_episodes_table() in workers is a read-only load and tables are not created concurrently
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
        mcap_queue.put(None)        # One finish sentinel per worker
    workers = [mp_ctx.Process(target=extract_episodes_to_parquet,
                              args=(mcap_queue, parquet_queue, warehouse_dir, topic_names, log_enabled),
                              daemon=True)
               for _ in range(num_workers)]
    for worker in workers:
        worker.start()

    register_parquets_from_queue(parquet_queue, savers, workers)

    for worker in workers:
        worker.join()
    # pid -> exitcode: use the pid to find that worker's last start in the log, which identifies the failing mcap
    killed_exitcodes = {worker.pid: worker.exitcode for worker in workers if worker.exitcode}
    if killed_exitcodes:
        logger.error(f"{len(killed_exitcodes)} 个 worker 非正常退出, "
                     f"{{pid: exitcode}}={killed_exitcodes} (-9 即被 OOM killer 杀掉)")

    logger.info(f"saved {sum(saver.num_episodes for saver in savers.values())} episodes in "
                f"{sum(saver.num_commits for saver in savers.values())} snapshots "
                f"to {warehouse_dir} ({len(savers)} prompts)")
    return tables


def build_global_index(tables: dict, warehouse_dir: str = WAREHOUSE_DIR) -> str:
    """Number every episode/sample of all prompts globally and write one episode_index.parquet."""
    # Take only metadata columns and physical locations from each prompt table (the huge samples column is never read), concatenated into one Arrow table
    episodes_meta = pa.concat_tables(
        [load_episodes_meta(prompt, table) for prompt, table in tables.items()])
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
    return index_path


def load_episodes_meta(prompt: str, table) -> pa.Table:
    """Read one prompt table's metadata columns file by file, plus where each episode lives.

    Returns columns EPISODE_META_COLUMNS + [data_file, row_group, prompt], one row per episode.
    """
    file_metas = []
    # plan_files gives all data files of the current snapshot; tables written by add_files have no delete files, so reading file by file reads the whole table
    for scan_task in table.scan().plan_files():
        data_file = scan_task.file.file_path
        parquet_file = pq.ParquetFile(data_file)
        # The writer emits one episode per row group, so the row number equals the row group number
        assert parquet_file.num_row_groups == parquet_file.metadata.num_rows, \
            f"{data_file} 不是一行一个 row group, 行号不能当 row group 号"
        file_meta = parquet_file.read(columns=EPISODE_META_COLUMNS)
        file_metas.append(file_meta
                          .append_column("data_file", pa.array([data_file] * file_meta.num_rows, pa.string()))
                          .append_column("row_group", pa.array(range(file_meta.num_rows), pa.int64())))
    assert file_metas, f"{prompt!r} 的表里没有任何数据文件"
    meta = pa.concat_tables(file_metas)
    return meta.append_column("prompt", pa.array([prompt] * meta.num_rows, pa.string()))


def load_existing_tables(warehouse_dir: str = WAREHOUSE_DIR) -> dict:
    """Load every prompt's episodes table already in the catalog, as {prompt: Table}."""
    catalog = load_episodes_catalog(warehouse_dir)
    # Table creation stores the prompt text in namespace attributes (namespace names are normalized and irreversible); use it to recover the mapping
    tables = {catalog.load_namespace_properties(namespace)["prompt"]:
              catalog.load_table((*namespace, EPISODES_TABLE_NAME))
              for namespace in catalog.list_namespaces()}
    assert tables, f"{warehouse_dir} 的 catalog 里没有任何表, 请先不带 --index-only 跑一次采集"
    return tables


def build_all(log_enabled: bool = False) -> str:
    """Extract all mcaps into Iceberg, then build the global index."""
    # All mcaps are enqueued and picked up by MAX_MCAP_WORKERS resident workers; the process count does not grow with mcap count
    tables = build_episodes_table(find_mcap_url_and_prompt_pairs(), WAREHOUSE_DIR, log_enabled)
    return build_global_index(tables, WAREHOUSE_DIR)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="并行抽取 mcap 存进 Iceberg, 再生成全局索引")
    arg_parser.add_argument("--index-only", action="store_true",
                            help="跳过采集, 只用 catalog 里已有的表重建 episode_index.parquet")
    arg_parser.add_argument("--log", action="store_true",
                            help="把运行日志写入 build_sample_idx.log; 不带此参数则不生成日志文件")
    cli_args = arg_parser.parse_args()
    # Only write the log file with --log: delete the old log first, then open in append mode; every process uses O_APPEND so they do not overwrite each other.
    # "w" mode would overwrite worker content from offset 0
    if cli_args.log:
        Path(LOG_PATH).unlink(missing_ok=True)
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_PATH, filemode="a",
                            encoding="utf-8", force=True)
    if cli_args.index_only:
        # Do not re-collect: rerunning collection would add_files the same data again, leaving two copies of every episode in the table
        build_global_index(load_existing_tables())
    else:
        build_all(cli_args.log)
