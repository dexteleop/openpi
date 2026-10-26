"""Unified config entry: specify which robot is in use and expose functions to read its topic configs.

Other programs always get configs via load_topics_config() and never import a robot's config file directly,
so switching robots only requires changing that single import line at the top of this file.

Every robot config file must provide the same-named constants; note which topic naming each constant uses:
    TELEAVATAV2_MCAP_TOPICS_MAPPING  {standard topic: (mcap topic, ...)}  keys standard, values mcap
    USER_SELECTED_TOPICS             {standard topic: obs/state/action}; keys must go through the mapping table before use
    TELEAVATAV2_VIDEO_TOPICS_GOP     mcap topic names, usable directly
    EPISODE_SIGNAL                   mcap topic name, usable directly; {topic_name, start, end}
"""
from __future__ import annotations

from openpi.training.lingyu_dataloader_v2.mcap_config import teleavatar_v2  # Switching robots only changes this line
from openpi.training.lingyu_dataloader_v2.model_config.config import load_state_and_action_concat


ROBOT = "teleavatar_v2" # TODO: pick which robot's mcap to use


#####
# Index used by MCAP_Player to play mcap topics
#####

def load_mcap_topics_config() -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Return the current robot's (mcap topic mapping table, {user-selected standard topic: role})."""
    if ROBOT == "teleavatar_v2":
        return teleavatar_v2.MCAP_TOPICS_MAPPING, teleavatar_v2.MODEL_SELECTED_TOPICS
    # TODO: elif other robots' constant configs
    else:
        raise ValueError(
            f"Cannot find the relevant robot configuration of {ROBOT!r} "
            f"in lingyu_dataloader_v2/robots_config."
        )


def load_mcap_video_topics_gop() -> dict[str, int]:
    """Return the current robot's {video topic: GOP length}; GOP>1 means inter-frame coding, requiring keyframe tracing."""
    if ROBOT == "teleavatar_v2":
        return teleavatar_v2.MCAP_VIDEO_TOPICS_GOP
    # TODO: elif other robots' constant configs
    else:
        raise ValueError(
            f"Cannot find the relevant robot configuration of {ROBOT!r} "
            f"in lingyu_dataloader_v2/robots_config."
        )


#####
# Used to determine the start and end of episodes
#####

def load_mcap_episode_topic_signal() -> dict:
    """Return the current robot's key signal {topic_name, start, end} marking episode start/end."""
    if ROBOT == "teleavatar_v2":
        return teleavatar_v2.MCAP_EPISODE_TOPIC_SIGNAL
    # TODO: elif other robots' constant configs
    else:
        raise ValueError(
            f"Cannot find the relevant robot configuration of {ROBOT!r} "
            f"in lingyu_dataloader_v2/robots_config."
        )


#####
# Used to determine the fields inside state and action topic messages
#####

def load_mcap_state_and_action_topics_fields() -> dict[str, tuple[str, ...]]:
    """Return {mcap topic: (required fields, ...)}; keys are already mcap topic names, no mapping needed."""
    # topic 与字段的唯一来源是 model_config 的 STATE_CONCAT/ACTION_CONCAT，机器人侧不再单独登记
    state_concat, action_concat = load_state_and_action_concat()
    state_and_action_fields = {}  # 同一 topic 需要多个字段时在此聚合成一个 tuple
    for mcap_topic, field_name in state_concat + action_concat:
        state_and_action_fields[mcap_topic] = state_and_action_fields.get(mcap_topic, ()) + (field_name,)
    return state_and_action_fields
