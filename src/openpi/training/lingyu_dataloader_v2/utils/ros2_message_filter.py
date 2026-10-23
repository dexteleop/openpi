"""Extract from ros2 messages the fields each topic really needs, per config.

STATE_and_ACTION_TOPICS_FIELDS's keys are already mcap topic names and match the topic_name
played by MCAP_Player, so no MCAP_TOPICS_MAPPING lookup is needed; query the table directly.
The config is read from mcap_config.config; this module does not care which robot is current.

Usage::
    from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import ros2_message_filter

    data = ros2_message_filter('/left_arm/joint_states', message)
    # -> {'position': array([...])}
"""
from __future__ import annotations

import numpy as np

from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_state_and_action_topics_fields


def ros2_message_filter(mcap_topic_name: str, ros2_message) -> dict[str, np.ndarray]:
    """
    Extract the fields selected in the config for topic_name, returning {field name: 1-D array}.

    Fields are returned separately, not concatenated here: concat order is decided by the model side's state/action structure; this only extracts values;
    a topic absent from the config asserts immediately instead of silently returning empty data.
    """
    # 1. Look up the config: which fields of this topic's message to take
    state_and_action_fields = load_mcap_state_and_action_topics_fields()
    assert mcap_topic_name in state_and_action_fields, \
        f"{mcap_topic_name} 未登记在 STATE_and_ACTION_TOPICS_FIELDS 中, 无法确定取哪个字段"
    selected_fields = state_and_action_fields[mcap_topic_name]

    # 2. Extract each field and normalize it into a 1-D array
    fields_data = {}
    for field_name in selected_fields:
        field_value = getattr(ros2_message, field_name)   # e.g. JointState.position
        fields_data[field_name] = np.atleast_1d(np.asarray(field_value))  # Scalar fields also become length-1 arrays

    return fields_data
