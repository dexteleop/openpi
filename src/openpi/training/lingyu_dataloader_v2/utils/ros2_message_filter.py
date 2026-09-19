"""按配置从 ros2 message 中取出该 topic 真正需要的字段。

STATE_and_ACTION_TOPICS_FIELDS 的键已是 mcap topic 名, 与 MCAP_Player 播出来的
topic_name 同名, 故无需经 MCAP_TOPICS_MAPPING 映射, 直接查表。
配置从 mcap_config.config 读取, 本模块不关心当前是哪台机器人。

用法::
    from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import ros2_message_filter

    data = ros2_message_filter('/left_arm/joint_states', message)
    # -> {'position': array([...])}
"""
from __future__ import annotations

import numpy as np

from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_state_and_action_topics_fields


def ros2_message_filter(mcap_topic_name: str, ros2_message) -> dict[str, np.ndarray]:
    """
    取出 topic_name 在配置中选定的字段, 返回 {字段名: 一维数组}。

    各字段分开返回而不在此拼接: 拼接顺序由模型侧的 state/action 结构决定, 这里只负责取值;
    topic 未登记在配置中时直接 assert, 不静默返回空数据。
    """
    # 1. 查配置: 这个 topic 的 message 要取哪几个字段
    state_and_action_fields = load_mcap_state_and_action_topics_fields()
    assert mcap_topic_name in state_and_action_fields, \
        f"{mcap_topic_name} 未登记在 STATE_and_ACTION_TOPICS_FIELDS 中, 无法确定取哪个字段"
    selected_fields = state_and_action_fields[mcap_topic_name]

    # 2. 逐个字段取值, 并统一成一维数组
    fields_data = {}
    for field_name in selected_fields:
        field_value = getattr(ros2_message, field_name)   # 如 JointState.position
        fields_data[field_name] = np.atleast_1d(np.asarray(field_value))  # 标量字段也变成长度 1 的数组

    return fields_data
