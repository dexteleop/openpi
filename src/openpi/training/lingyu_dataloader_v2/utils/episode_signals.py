"""Read the key-signal config marking episode start/end.

EPISODE_SIGNAL already holds topic names that actually exist in the mcap, no mapping-table conversion needed.
The config is read from mcap_config.config; this module does not care which robot is current.

Usage::
    from openpi.training.lingyu_dataloader_v2.utils.episode_signals import set_episode_signal

    signal_topic, start_button_idx, end_button_idx = set_episode_signal()
"""
from __future__ import annotations
from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_episode_topic_signal


def set_episode_signal() -> tuple[str, int, int]:
    """Split EPISODE_SIGNAL into (signal_topic, start_button_idx, end_button_idx)."""
    episode_signal = load_mcap_episode_topic_signal()
    return episode_signal["topic_name"], episode_signal["start"], episode_signal["end"]
