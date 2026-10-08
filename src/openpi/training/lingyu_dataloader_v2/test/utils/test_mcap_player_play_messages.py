"""
Play messages from an mcap, count per-topic message totals, and verify
that every message carries a non-empty msg_def string.
"""
from collections import Counter
from time import perf_counter
from openpi.training.lingyu_dataloader_v2.utils.mcap_player import _play_messages
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_SAMPLE_COUNT = 2000  # enough to cover all topics without scanning the whole file


def test():
    """Play the first _SAMPLE_COUNT messages and verify msg_def is present for every topic."""
    from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import find_mcap_url_and_prompt_pairs

    t0 = perf_counter()
    path = next(iter(find_mcap_url_and_prompt_pairs()))
    t1 = perf_counter()
    logger.info(f"find_mcap elapsed: {t1 - t0:.3f}s")
    logger.info(f"playing {path}")
    counts_per_topic = Counter()          # topic_name -> message count
    topic_msg_types: dict[str, str] = {}  # topic_name -> msg_type
    topic_msg_defs: dict[str, str] = {}   # topic_name -> msg_def (first seen)
    log_time_first = log_time_last = 0

    play_start = perf_counter()  # 播放耗时: 从取第一条消息开始计时
    msg_gen = _play_messages(path)
    for i, message in enumerate(msg_gen):
        logger.info("  %s", message)
        topic = message["topic_name"]
        counts_per_topic[topic] += 1
        topic_msg_types[topic] = message["msg_type"]
        if topic not in topic_msg_defs and message["msg_def"]:
            topic_msg_defs[topic] = message["msg_def"]
        log_time_first = log_time_first or message["log_time"]
        log_time_last = message["log_time"]
        if i >= _SAMPLE_COUNT:
            break
    play_elapsed = perf_counter() - play_start

    close_start = perf_counter()
    msg_gen.close()   # 显式关闭 generator
    close_elapsed = perf_counter() - close_start

    assert counts_per_topic, "no messages found in mcap"
    logger.info(
        "_play_messages: sampled %d messages, topics %d, "
        "duration %.3fs, play_elapsed %.3fs, pool_shutdown %.3fs, total %.3fs",
        sum(counts_per_topic.values()),
        len(counts_per_topic),
        (log_time_last - log_time_first) / 1e9,
        play_elapsed,
        close_elapsed,
        play_elapsed + close_elapsed,
    )
    for topic_name, count in sorted(counts_per_topic.items()):
        logger.info(
            "%s\t%s\t%d\tmsg_def[:40]=%r",
            topic_name,
            topic_msg_types[topic_name],
            count,
            topic_msg_defs.get(topic_name, "")[:40],
        )

    missing_def = [t for t in counts_per_topic if not topic_msg_defs.get(t)]
    assert not missing_def, f"topics missing msg_def: {missing_def}"


if __name__ == "__main__":
    test()
