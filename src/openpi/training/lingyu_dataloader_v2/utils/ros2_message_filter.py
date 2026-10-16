"""按配置从 ros2 message 中取出该 topic 真正需要的字段。

要取的字段直接来自 model_config 的 STATE_CONCAT/ACTION_CONCAT: 拼接配置里的
(mcap topic, 字段) 已完整登记各 topic 要取的字段, 不再需要 mcap_config 侧的字段
登记表。键已是 mcap topic 名, 与 MCAP_Player 播出来的 topic_name 同名, 故无需经
MCAP_TOPICS_MAPPING 映射, 直接查表。

用法::
    from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import ros2_message_filter

    data = ros2_message_filter('/left_arm/joint_states', message)
    # -> {'position': array([...])}
"""
from __future__ import annotations

import numpy as np

from openpi.training.lingyu_dataloader_v2.model_config.config import load_state_and_action_concat


def ros2_message_filter(mcap_topic_name: str, ros2_message) -> dict[str, np.ndarray]:
    """
    取出 topic_name 在配置中选定的字段, 返回 {字段名: 一维数组}。

    各字段分开返回而不在此拼接: 拼接顺序由模型侧的 state/action 结构决定, 这里只负责取值;
    topic 未登记在配置中时直接 assert, 不静默返回空数据。
    """
    # 1. 查拼接配置: 这个 topic 的 message 要取哪几个字段
    state_concat, action_concat = load_state_and_action_concat()
    state_and_action_fields = {}
    for mcap_topic, field_name in (*state_concat, *action_concat):
        state_and_action_fields.setdefault(mcap_topic, []).append(field_name)
    assert mcap_topic_name in state_and_action_fields, \
        f"{mcap_topic_name} 未登记在 STATE_CONCAT/ACTION_CONCAT 中, 无法确定取哪个字段"
    selected_fields = state_and_action_fields[mcap_topic_name]

    # 2. 逐个字段取值, 并统一成一维数组
    fields_data = {}
    for field_name in selected_fields:
        field_value = getattr(ros2_message, field_name)   # 如 JointState.position
        # 嵌套 message(如 Pose.position 的 Point)不是数组: 按字段声明顺序取数值拼成数组, Point -> [x, y, z]
        if hasattr(field_value, "__msgtype__"):
            field_value = [getattr(field_value, nested_name) for nested_name
                           in type(field_value).__annotations__ if not nested_name.startswith("_")]
        fields_data[field_name] = np.atleast_1d(np.asarray(field_value))  # 标量字段也变成长度 1 的数组

    return fields_data
