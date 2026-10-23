"""
Play one real mcap with MCAPSampleExtractor and check the yielded episodes are self-consistent:
every sample has all topics, each topic is [msg_type, msg_def, log_time, locations],
and an action topic's locations are the locating sequence of [current action, the next chunk-1 actions].

Only the first _MAX_EPISODES episodes are taken; the whole mcap need not be played.
The first _DUMP_SAMPLES samples of each episode have all their topics logged line by line, for eyeball-checking sample appearance.
"""
from collections import Counter

from openpi.training.lingyu_dataloader_v2.mcap_sample_extractor import (
    MCAPSampleExtractor,
    MIN_EPISODE_LENGTH,
    ACTION_CHUNK_LENGTH)
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_MAX_EPISODES = 2       # One mcap is easily a dozen GB; verifying the first few episodes suffices
_DUMP_SAMPLES = 2       # How many of each episode's first samples get all topics printed


def _log_sample(source_episode_seq: int, sample_idx: int, topics: dict, topics_role: dict):
    """Log each topic of one sample on one line: role / parsed format / timestamp / locations."""
    logger.info("ep=%s sample=%s 共 %d 个 topic",
                source_episode_seq, sample_idx, len(topics))
    # Sort by role then topic name, so obs/state/action appear in blocks and every episode's output order is consistent
    for topic in sorted(topics, key=lambda name: (topics_role[name], name)):
        msg_type, msg_def, log_time, locations = topics[topic]
        # Each location is (mcap_url, chunk_file_offset, uncompressed_byte_offset, record_length);
        # mcap_url is constant for the same file and very long, so it is dropped when printing, keeping only the three offsets
        offsets = [loc[1:] for loc in locations]
        logger.info("    [%-6s] %-52s msg_type=%-42s log_time=%s def=%dB locations=%d %s",
                    topics_role[topic], topic, msg_type, log_time,
                    len(msg_def), len(offsets), offsets)


def test():
    """ Run one mcap to test the MCAPSampleExtractor class's iter_episodes() functionality """
    from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
        find_mcap_url_and_prompt_pairs)
    path = next(iter(find_mcap_url_and_prompt_pairs()))

    logger.info(f"播放 {path}")
    extractor = MCAPSampleExtractor(path)
    topics_role = filter_topics()   # {topic: obs/state/action}, only for annotating roles in logs

    topic_counts = Counter()        # Topics per sample -> sample count
    record_lengths = Counter()      # Fields per message record -> record count
    chunk_lengths = Counter()       # Locations per action topic -> record count
    for source_episode_seq, samples in extractor.iter_episodes():
        logger.info("ep=%s\tsamples=%d", source_episode_seq, len(samples))
        # Persisted episodes must all reach the minimum length
        assert len(samples) >= MIN_EPISODE_LENGTH, \
            f"ep={source_episode_seq} 只有 {len(samples)} 个 sample"
        last_sample_idx = len(samples) - 1
        for sample_idx, topics in enumerate(samples):
            if sample_idx < _DUMP_SAMPLES:
                _log_sample(source_episode_seq, sample_idx, topics, topics_role)
            topic_counts[len(topics)] += 1
            for topic, record in topics.items():
                record_lengths[len(record)] += 1
                if topic in extractor.action_topics_set:
                    # action's locations are [current action, the next chunk-1 actions]; where the tail runs short, the last sample repeats
                    chunk_lengths[len(record[-1])] += 1
                    for step, location in enumerate(record[-1]):
                        future_idx = min(sample_idx + step, last_sample_idx)
                        # A future sample's locations[0] is its own current action
                        assert location == samples[future_idx][topic][-1][0], \
                            f"ep={source_episode_seq} sample={sample_idx} 的 action 第 {step} 步不是未来动作"

        if extractor.num_episodes >= _MAX_EPISODES:
            break

    logger.info(f"  X/Y 对 {extractor.num_xy_pairs}, 产出 episode {extractor.num_episodes}, "
                f"sample {extractor.total_samples}")

    assert extractor.num_episodes, "没有 yield 出任何 episode"
    # Every sample must cover all selected topics, and every record must be [msg_type, msg_def, log_time, locations]
    assert set(topic_counts) == {len(extractor.all_topics_set)}, \
        f"存在 topic 不齐的 sample: {dict(topic_counts)}"
    assert set(record_lengths) == {4}, f"message 记录字段数异常: {dict(record_lengths)}"
    # Each action topic's location count must equal the model config's chunk length
    assert set(chunk_lengths) == {ACTION_CHUNK_LENGTH}, \
        f"action locations 条数异常: {dict(chunk_lengths)}"


if __name__ == "__main__":
    test()
