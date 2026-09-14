"""
逐条播放mcap，完成后
统计mcap中每个topic的message数量
然后更新 lingyu_dataloader_v2/topics_table_mapping.py 中的话题映射表格
"""
from openpi.training.lingyu_dataloader_v2.utils.mcap_player import _play_messages
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
logger = get_logger(__file__)
from collections import Counter


def test():
    """ 运行一个mcap来测试 MCAP播放器 功能 """
    from openpi.training.lingyu_dataloader_v2.utils.search_mcap_paths import (
        find_mcap_paths)
    path = find_mcap_paths()[0]

    logger.info(f"播放 {path}")
    counts_per_topic = Counter()  # topic_name -> 消息条数
    topic_msg_types: dict[str, str] = {}  # topic_name -> msg_type
    log_time_first = log_time_last = 0

    for idx, message in enumerate(_play_messages(path)):
        logger.info("  %s", message)
        counts_per_topic[message["topic_name"]] += 1
        topic_msg_types[message["topic_name"]] = message["msg_type"]
        log_time_first = log_time_first or message["log_time"]
        log_time_last = message["log_time"]

    logger.info(f"  消息总数 {sum(counts_per_topic.values())}, "
                f"topic 数 {len(counts_per_topic)}, "
                f"时长 {(log_time_last - log_time_first) / 1e9:.3f}s")

    for topic_name, count in sorted(counts_per_topic.items()):
        logger.info("%s\t%s\t%d", topic_name, topic_msg_types[topic_name], count)


if __name__ == "__main__":
    test()