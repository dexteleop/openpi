"""读取标记 episode 起止的按键信号配置。

EPISODE_SIGNAL 里写的已是 mcap 中实际存在的 topic 名, 无需经映射表换算。
配置从 mcap_config.config 读取, 本模块不关心当前是哪台机器人。

用法::
    from openpi.training.lingyu_dataloader_v2.utils.episode_signals import set_episode_signal

    signal_topic, start_button_idx, end_button_idx = set_episode_signal()
"""
from __future__ import annotations
from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_episode_signal


def set_episode_signal() -> tuple[str, int, int]:
    """把 EPISODE_SIGNAL 拆成 (signal_topic, start_button_idx, end_button_idx)。"""
    episode_signal = load_episode_signal()
    return episode_signal["topic_name"], episode_signal["start"], episode_signal["end"]
