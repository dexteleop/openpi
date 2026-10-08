"""用真实 mcap 的 episode 校验 EpisodeParquetWriter 的增量写与 IcebergEpisodeSaver 的注册。

核对::
    1. 逐个 episode 写入时 RSS 不随行数增长: worker 的内存恒等于一个 episode
    2. 按 roll_episodes 滚动换文件: 满的文件返回 (路径, 行数), 未满时返回 None
    3. 一行一个 row group, 且行数 == 写入的 episode 数
    4. add_files 能登记这种多 row group 文件, 读回的 samples 结构原样可用

之所以要盯住第 1 条: 上一版把整批 episode 攒在内存里再转 Arrow, 64 个 worker 一起跑会
把内存撑爆(曾被 OOM killer 杀掉), 增量写是否真的释放内存必须实测, 不能只看代码。
"""
import os
import shutil

import pyarrow.parquet as pq

from openpi.training.lingyu_dataloader_v2.build_sample_idx import EPISODES_PER_PARQUET
from openpi.training.lingyu_dataloader_v2.mcap_sample_extractor import MCAPSampleExtractor
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics
from openpi.training.lingyu_dataloader_v2.utils.pyiceberg_saver import (
    LOCATION_TYPE, EpisodeParquetWriter, EpisodeRecord, IcebergEpisodeSaver,
    load_episodes_table)
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import find_mcap_url_and_prompt_pairs
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_WAREHOUSE_DIR = "/tmp/test_pyiceberg_saver_wh"
_PROMPT = "Test incremental writer."
_ROLL_EPISODES = 3              # 缩小滚动阈值, 少抽几个 episode 就能看到换文件
_NUM_EPISODES = 7               # 2 个满文件(3+3) + 1 个尾批(1)
_RSS_TOLERANCE_GB = 0.30        # 一个 episode 的 Arrow 表约 40 MB, 留足余量


def _rss_gb() -> float:
    """当前进程的 RSS, 直接读 /proc 避免引入依赖。"""
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / 1024**3


def test_pyiceberg_saver():
    """逐个 episode 增量写 parquet, 内存不涨、按阈值滚动, 再 add_files 注册后读回。"""
    shutil.rmtree(_WAREHOUSE_DIR, ignore_errors=True)
    topic_names = sorted(filter_topics())
    table = load_episodes_table(_WAREHOUSE_DIR, topic_names, _PROMPT)
    writer = EpisodeParquetWriter(table, topic_names, _ROLL_EPISODES)

    mcap_path = next(iter(find_mcap_url_and_prompt_pairs()))
    extractor = MCAPSampleExtractor(mcap_path)
    logger.info("抽 %s 的前 %d 个 episode, 每 %d 个滚动一个 parquet",
                mcap_path, _NUM_EPISODES, _ROLL_EPISODES)

    base_rss = _rss_gb()
    closed_files = []           # [(路径, episode 数), ...]
    num_written = 0
    for source_episode_seq, samples in extractor.iter_episodes():
        episode = EpisodeRecord(extractor.source_id, source_episode_seq, samples)
        closed_file = writer.write(episode)
        num_written += 1
        # --- 1. 内存不随已写行数增长 ---
        rss_growth = _rss_gb() - base_rss
        assert rss_growth < _RSS_TOLERANCE_GB, \
            f"写到第 {num_written} 个 episode 时 RSS 已涨 {rss_growth:.2f} GB, 增量写没有释放内存"
        # --- 2. 只有写满 _ROLL_EPISODES 行才交出文件, 未满时返回 None ---
        if num_written % _ROLL_EPISODES == 0:
            assert closed_file is not None, f"第 {num_written} 个 episode 应当刚好写满一个文件"
            closed_files.append(closed_file)
        else:
            assert closed_file is None, f"第 {num_written} 个 episode 不该触发换文件"
        logger.info("episode#%d 已写入, 累计 %d 个, RSS 增量 %+.2f GB%s",
                    source_episode_seq, num_written, rss_growth,
                    f", 滚出文件 {closed_file[1]} 行" if closed_file else "")
        if num_written >= _NUM_EPISODES:
            break

    tail_file = writer.close()  # 尾批: 不满一个文件也要交出来
    assert tail_file is not None and tail_file[1] == _NUM_EPISODES % _ROLL_EPISODES, \
        f"尾批行数与预期不符: {tail_file}"
    closed_files.append(tail_file)
    assert writer.close() is None, "重复 close() 不应再交出文件"

    # --- 3. 一行一个 row group, 行数与写入的 episode 数一致 ---
    for parquet_path, num_episodes in closed_files:
        parquet_file = pq.ParquetFile(parquet_path.removeprefix("file://"))
        assert parquet_file.metadata.num_rows == num_episodes, \
            f"{parquet_path} 的行数与交出的 episode 数不符"
        assert parquet_file.num_row_groups == num_episodes, \
            f"{parquet_path} 不是一行一个 row group"
        size_mb = os.path.getsize(parquet_path.removeprefix("file://")) / 1024**2
        logger.info("文件 %d 行 / %d 个 row group / %.2f MB", num_episodes,
                    parquet_file.num_row_groups, size_mb)

    # --- 4. add_files 登记这些文件, 一个文件一个 snapshot, 读回结构原样可用 ---
    saver = IcebergEpisodeSaver(table)
    for parquet_path, num_episodes in closed_files:
        saver.extend([parquet_path], num_episodes)
    assert saver.num_commits == len(closed_files), "snapshot 数与文件数不符"
    assert saver.num_episodes == _NUM_EPISODES, "注册的 episode 总数不符"

    episode_ids = table.scan(selected_fields=("episode_id",)).to_arrow()
    assert episode_ids.num_rows == _NUM_EPISODES, "读回的行数与写入的 episode 数不符"
    assert len(set(episode_ids.column("episode_id").to_pylist())) == _NUM_EPISODES, \
        "episode_id 出现重复"
    first_row = table.scan(limit=1).to_arrow().to_pylist()[0]
    assert first_row["num_samples"] == len(first_row["samples"]), \
        "num_samples 与 samples 实际长度不符"
    first_message = first_row["samples"][0][topic_names[0]]
    assert list(first_message["locations"][0]) == list(LOCATION_TYPE.names), \
        "locations 四元组没有带上字段名"

    logger.info("%d 个 episode 分 %d 个文件写入并注册, 首行 %s 有 %d 个 sample",
                _NUM_EPISODES, len(closed_files), first_row["episode_id"],
                first_row["num_samples"])
    logger.info("采集时实际用的滚动阈值 EPISODES_PER_PARQUET=%d", EPISODES_PER_PARQUET)
    shutil.rmtree(_WAREHOUSE_DIR, ignore_errors=True)


if __name__ == "__main__":
    test_pyiceberg_saver()
