"""Expand the user-selected TeleAvatarV2 standard topics into the topic names actually used in mcap files.

USER_SELECTED_TOPICS holds TeleAvatarV2's standard topic names; mcaps recorded by different robots
may use different topic names, so everything goes through that robot's TELEAVATAV2_MCAP_TOPICS_MAPPING.
The config is read from mcap_config.config.load_topics_config(); this module does not care which robot is current.

Usage::
    from openpi.training.lingyu_dataloader_v2.utils.topics_filter import filter_topics

    if topic_name in filter_topics():             # Only check selection
        ...
    if filter_topics()[topic_name] == 'action':   # Also need the topic's role
        ...
"""
from __future__ import annotations
from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_topics_config


def filter_topics() -> dict[str, str]:
    """
    Expand the user-selected TeleAvatarV2 topics into {topic name actually used in this mcap: obs/state/action}.

    One standard topic may map to multiple mcap topics, all sharing the standard topic's role;
    selecting a standard topic absent from the mapping table asserts immediately instead of running on.
    """
    mcap_topics_mapping, user_selected_topics = load_mcap_topics_config()
    unregistered_topics = sorted(user_selected_topics.keys() - mcap_topics_mapping.keys())
    assert not unregistered_topics, \
        f"USER_SELECTED_TOPICS 中这些 Standard topic 未登记在映射表中: {unregistered_topics}"

    selected_mcap_topics = {}
    for user_topic, topic_role in user_selected_topics.items():
        for mcap_topic in mcap_topics_mapping[user_topic]:
            selected_mcap_topics[mcap_topic] = topic_role
    return selected_mcap_topics
