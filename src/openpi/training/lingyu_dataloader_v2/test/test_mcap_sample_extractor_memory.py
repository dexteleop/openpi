"""
Watch how worker memory grows while MCAPSampleExtractor extracts one mcap, locating build_sample_idx's OOM.

A background thread records every _MONITOR_SEC seconds: process RSS, the current episode's accumulated sample count,
and the GOP length (len(locations)) of each video topic's latest frame. It exits directly when RSS exceeds _RSS_LIMIT_GB.

Usage::
    python test/test_mcap_sample_extractor_memory.py <mcap_url>
"""
import os
import sys
import threading
import time

import psutil

from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_video_topics_gop
from openpi.training.lingyu_dataloader_v2.mcap_sample_extractor import MCAPSampleExtractor
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_MONITOR_SEC = 10
_RSS_LIMIT_GB = 30      # Exceeding this exits proactively, so the machine never reaches OOM


def _monitor_memory(extractor: MCAPSampleExtractor, video_topics: list, stop_event):
    """Periodically record RSS / the current episode's sample count / each video topic's GOP length"""
    process = psutil.Process()
    start_time = time.time()
    while not stop_event.wait(_MONITOR_SEC):
        rss_gb = process.memory_info().rss / 1024 ** 3
        # Read-only reference, extractor state untouched; reading a half-updated dict only affects this one log line
        gop_lens = {topic.split('/')[1]: len(extractor._cur_msg_record[topic][3])
                    for topic in video_topics if topic in extractor._cur_msg_record}
        logger.info("t=%4ds rss=%.2fGB episodes=%d cur_episode_samples=%d gop_lens=%s",
                    time.time() - start_time, rss_gb, extractor.num_episodes,
                    len(extractor._episode_samples), gop_lens)
        if rss_gb > _RSS_LIMIT_GB:
            logger.error("rss %.2fGB > %dGB, 主动退出", rss_gb, _RSS_LIMIT_GB)
            os._exit(1)


def run(mcap_url: str):
    """Extract the whole mcap, recording each episode's sample count and RSS at its end"""
    assert mcap_url.endswith(".mcap"), f"不是 mcap: {mcap_url}"
    video_topics = [topic for topic, gop in load_mcap_video_topics_gop().items() if gop > 1]
    extractor = MCAPSampleExtractor(mcap_url)
    stop_event = threading.Event()
    threading.Thread(target=_monitor_memory, args=(extractor, video_topics, stop_event),
                     daemon=True).start()
    for source_episode_seq, samples in extractor.iter_episodes():
        logger.info("episode %d: %d samples, rss=%.2fGB", source_episode_seq, len(samples),
                    psutil.Process().memory_info().rss / 1024 ** 3)
    stop_event.set()
    logger.info("done: %s (%d episodes, %d samples)",
                mcap_url, extractor.num_episodes, extractor.total_samples)


if __name__ == "__main__":
    run(sys.argv[1])
