"""把用户选定的 TeleAvatarV2 标准 topic 展开成 mcap 文件里实际使用的 topic 名称。

USER_SELECTED_TOPICS 里写的是 TeleAvatarV2 的标准 topic 名, 不同机器人录出的 mcap
里对应的 topic 名可能不同, 故统一经该机器人的 TELEAVATAV2_MCAP_TOPICS_MAPPING 映射。
配置从 robots_config.config.load_topics_config() 读取, 本模块不关心当前是哪台机器人。

用法::
    from openpi.training.lingyu_dataloader_v2.utils.topics_filter import filter_topics

    if topic_name in filter_topics():             # 只判断是否选中
        ...
    if filter_topics()[topic_name] == 'action':   # 还需知道该 topic 的角色
        ...
"""
from __future__ import annotations
from openpi.training.lingyu_dataloader_v2.robots_config.config import load_topics_config


def filter_topics() -> dict[str, str]:
    """
    把用户选定的 TeleAvatarV2 topic 展开成 {本 mcap 中实际使用的 topic 名: obs/state/action}。

    一个标准 topic 可映射到多个 mcap topic, 它们共享该标准 topic 的角色;
    用户选定了映射表里没登记的标准 topic 时直接 assert, 不继续运行。
    """
    mcap_topics_mapping, user_selected_topics = load_topics_config()
    unregistered_topics = sorted(user_selected_topics.keys() - mcap_topics_mapping.keys())
    assert not unregistered_topics, \
        f"USER_SELECTED_TOPICS 中这些 Standard topic 未登记在映射表中: {unregistered_topics}"

    selected_mcap_topics = {}
    for user_topic, topic_role in user_selected_topics.items():
        for mcap_topic in mcap_topics_mapping[user_topic]:
            selected_mcap_topics[mcap_topic] = topic_role
    return selected_mcap_topics
