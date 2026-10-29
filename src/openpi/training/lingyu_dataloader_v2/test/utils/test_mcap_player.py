"""
Play an mcap message by message with MCAP_Player (yielding only topics in USER_SELECTED_TOPIC), then
count each topic's messages and the locations accumulated from the IDR keyframe for ffmpeg topics
"""
from openpi.training.lingyu_dataloader_v2.utils.mcap_player import MCAP_Player
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
logger = get_logger(__file__)
from collections import Counter

_SAMPLE_COUNT = 2000

def test():
    """ Run one mcap to test the MCAP_Player class's play_messages() functionality """
    from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
        find_mcap_url_and_prompt_pairs)
    path = next(iter(find_mcap_url_and_prompt_pairs()))

    logger.info(f"播放 {path}")
    counts_per_topic = Counter()  # topic_name -> message count
    topic_msg_types: dict[str, str] = {}  # topic_name -> msg_type
    locations_max: dict[str, int] = {}  # topic_name -> max locations length (longest GOP frame count)
    log_time_first = log_time_last = 0

    mcap_player = MCAP_Player(path)

    for i, message in enumerate(mcap_player.play_messages()):
        # A message was read
        logger.info("  %s", message)
        topic_name, msg_type, msg_def, log_time, locations = message

        counts_per_topic[topic_name] += 1
        topic_msg_types[topic_name] = msg_type
        locations_max[topic_name] = max(locations_max.get(topic_name, 0), len(locations))
        log_time_first = log_time_first or int(log_time)
        log_time_last = int(log_time)

        if i + 1 >= _SAMPLE_COUNT:
            break

    logger.info(f"  消息总数 {sum(counts_per_topic.values())}, "
                f"topic 数 {len(counts_per_topic)}, "
                f"时长 {(log_time_last - log_time_first) / 1e9:.3f}s")

    for topic_name, count in sorted(counts_per_topic.items()):
        logger.info("%s\t%s\t%d\t%d", topic_name, topic_msg_types[topic_name],
                    count, locations_max[topic_name])


if __name__ == "__main__":
    test()
