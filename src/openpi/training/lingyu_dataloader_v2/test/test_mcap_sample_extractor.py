"""
用 MCAPSampleExtractor 播一个真实 mcap, 检查 yield 出来的 episode 是否自洽：
每个 sample 的 topic 齐全, 各 topic 都是 [msg_type, msg_def, log_time, locations],
且 action topic 的 locations 是 [当前动作, 后续 chunk-1 个动作] 的定位序列。

只取前 _MAX_EPISODES 个 episode, 不必播完整个 mcap。
每个 episode 的前 _DUMP_SAMPLES 个 sample 会把全部 topic 逐行打进日志, 便于肉眼核对样本长相。
"""
from collections import Counter

from openpi.training.lingyu_dataloader_v2.mcap_sample_extractor import (
    MCAPSampleExtractor,
    MIN_EPISODE_LENGTH,
    ACTION_CHUNK_LENGTH)
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_MAX_EPISODES = 2       # 一个 mcap 动辄十几 GB, 验完前几个 episode 即可
_DUMP_SAMPLES = 2       # 每个 episode 打印前几个 sample 的全部 topic


def _log_sample(source_episode_seq: int, sample_idx: int, topics: dict, topics_role: dict):
    """把一个 sample 的每个 topic 打成一行: 角色 / 解析格式 / 时间戳 / locations。"""
    logger.info("ep=%s sample=%s 共 %d 个 topic",
                source_episode_seq, sample_idx, len(topics))
    # 按角色再按 topic 名排序, 让 obs/state/action 成块出现, 每个 episode 的输出顺序一致
    for topic in sorted(topics, key=lambda name: (topics_role[name], name)):
        msg_type, msg_def, log_time, locations = topics[topic]
        # locations 每条是 (mcap_url, chunk_file_offset, uncompressed_byte_offset, record_length),
        # mcap_url 对同一个文件恒定且很长, 打印时丢掉, 只留三个偏移量
        offsets = [loc[1:] for loc in locations]
        logger.info("    [%-6s] %-52s msg_type=%-42s log_time=%s def=%dB locations=%d %s",
                    topics_role[topic], topic, msg_type, log_time,
                    len(msg_def), len(offsets), offsets)


def test():
    """ 运行一个mcap来测试 MCAPSampleExtractor 类的 iter_episodes() 功能 """
    from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
        find_mcap_url_and_prompt_pairs)
    path = next(iter(find_mcap_url_and_prompt_pairs()))

    logger.info(f"播放 {path}")
    extractor = MCAPSampleExtractor(path)
    topics_role = filter_topics()   # {topic: obs/state/action}, 仅供日志标注角色

    topic_counts = Counter()        # 每个 sample 的 topic 数 -> sample 数
    record_lengths = Counter()      # 每条 message 记录的字段数 -> 记录数
    chunk_lengths = Counter()       # 每个 action topic 的 locations 条数 -> 记录数
    for source_episode_seq, samples in extractor.iter_episodes():
        logger.info("ep=%s\tsamples=%d", source_episode_seq, len(samples))
        # 落盘的 episode 必须都达到最小长度
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
                    # action 的 locations 是 [当前动作, 后续 chunk-1 个动作], 末尾不足处重复最后一个 sample
                    chunk_lengths[len(record[-1])] += 1
                    for step, location in enumerate(record[-1]):
                        future_idx = min(sample_idx + step, last_sample_idx)
                        # 未来 sample 的 locations 第 0 条即它自己的当前动作
                        assert location == samples[future_idx][topic][-1][0], \
                            f"ep={source_episode_seq} sample={sample_idx} 的 action 第 {step} 步不是未来动作"

        if extractor.num_episodes >= _MAX_EPISODES:
            break

    logger.info(f"  X/Y 对 {extractor.num_xy_pairs}, 产出 episode {extractor.num_episodes}, "
                f"sample {extractor.total_samples}")

    assert extractor.num_episodes, "没有 yield 出任何 episode"
    # 每个 sample 必须覆盖全部选定 topic, 每条记录必须是 [msg_type, msg_def, log_time, locations]
    assert set(topic_counts) == {len(extractor.all_topics_set)}, \
        f"存在 topic 不齐的 sample: {dict(topic_counts)}"
    assert set(record_lengths) == {4}, f"message 记录字段数异常: {dict(record_lengths)}"
    # 每个 action topic 的 locations 条数必须等于模型配置的 chunk 长度
    assert set(chunk_lengths) == {ACTION_CHUNK_LENGTH}, \
        f"action locations 条数异常: {dict(chunk_lengths)}"


if __name__ == "__main__":
    test()
