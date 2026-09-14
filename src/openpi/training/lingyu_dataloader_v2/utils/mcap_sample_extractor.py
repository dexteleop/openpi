# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
把一个 mcap 按 X/Y 按键切成 episode, 并为每个 sample 选出各 topic 的 message。

只保留两套状态机:
    1. X/Y 按键 -> episode 起止
    2. 时间窗口 -> 每个 sample 的 state / action 采集时刻
每到一个采集时刻, 就把该时刻各 topic 的"最新一条 message"记下来。

数据 topic 与其 obs/state/action 角色均来自 USER_SELECTED_TOPICS, 由 MCAP_Player 完成筛选;
唯一需要反序列化的是标记 episode 起止的 Joy 按键消息, 经 MCAP_Message_Fetcher 回读。

一个 episode 的全部 sample 先攒在内存, 按 Y 键正常收尾且样本数不少于 MIN_EPISODE_LENGTH
才整段落盘, 因此结果里不会有重录作废的、末尾没按 Y 的、以及过短 episode 的样本。
单个 sample 同样要求完整: obs/state 先挂起, 配齐 action 且 topic 齐全才收录。

输出 <output_dir>/sample_idx_<mcap 名>.json, 结构::
    {episode_idx: {sample_idx: {topic: [msg_type, msg_def, log_time, locations]}}}
episode_idx 为该 mcap 内的 X/Y 对序号, 丢弃的 episode 不出现, 故序号可能不连续。
每个 episode 单独成行写出, 使 build_sample_idx.py 能逐行流式合并, 不必整份读入内存。

用法::
    extractor = MCAPSampleExtractor("/path/to/rec_xxx_0.mcap", "/path/to/output")
    try:
        extractor.convert_single_mcap()
    finally:
        extractor.close()
"""

import json
from pathlib import Path
import logging
import traceback

from openpi.training.lingyu_dataloader_v2.utils.mcap_player import MCAP_Player
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher
from openpi.training.lingyu_dataloader_v2.utils.topics_filter import filter_topics
from openpi.training.lingyu_dataloader_v2.utils.episode_signals import set_episode_signal

# Constants
MIN_EPISODE_LENGTH = 30
ACTION_OFFSET_RATIO = 1.0 / 3.0
# episode 起止按键信号: 与 build_sample_idx.py 分属两套独立来源, 故此处仍硬编码, 不读 robots_config
MCAP_SIGNAL_TOPIC, X_BUTTON, Y_BUTTON = set_episode_signal()
OUTPUT_JSON_PREFIX = "sample_idx_"


class MCAPSampleExtractor:
    """Replay one mcap and record which message each sample selects per topic."""

    def __init__(self, mcap_path: str, output_dir: str, fps: int = 30):
        self.mcap_path = mcap_path
        self.mcap_name = Path(mcap_path).name
        self.fps = fps
        self.frame_duration = 1.0 / self.fps

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # User Selected Topics: {mcap topic: 角色 obs/state/action}
        mcap_topics_role = filter_topics()
        self.all_topics_set = frozenset(mcap_topics_role)
        # state 采集窗口覆盖观测与状态, action 采集窗口覆盖动作, 两者互斥且合起来即全部选定 topic
        self.state_phase_topics = frozenset(
            topic for topic, role in mcap_topics_role.items() if role in ('obs', 'state'))
        self.action_topics_set = frozenset(
            topic for topic, role in mcap_topics_role.items() if role == 'action')

        # Episode tracking
        self.num_xy_pairs = 0
        self.total_samples = 0          # 已落盘的 sample 总数
        self.num_episodes = 0           # 已落盘的 episode 总数
        # message tracking: topic -> [msg_type, msg_def, log_time, locations], 即该 topic 的最新一条 message
        self._cur_msg_record = {}
        # 当前 episode 已攒下的 sample, 按 Y 键正常收尾才整段落盘; 重录或未收尾则整段丢弃
        self._episode_samples = {}

        self.logger.info(f"\n=== Processing {self.mcap_name} ===")

        # 播放器已完成 topic 筛选与视频 GOP 累积, 文件在首次迭代 play_messages() 时才打开
        self.player = MCAP_Player(mcap_path)
        # 按键消息需反序列化, fetcher 自带 typestore 与自定义类型注册
        self.fetcher = MCAP_Message_Fetcher()

        # 结果 json, 每收尾一个 episode 就追加一行, 避免整个 mcap 的样本全压在内存里
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mcap_stem = Path(mcap_path).stem
        self.output_json_path = output_dir / f"{OUTPUT_JSON_PREFIX}{mcap_stem}.json"
        self._json_file = open(self.output_json_path, "w", encoding="utf-8")
        self._json_file.write("{\n")

    def close(self):
        """Close the output json."""
        self._json_file.write("\n}\n")
        self._json_file.close()

    def update_messages(self, topic_name: str, msg_type: str, msg_def: str,
                        log_time: int, locations: list):
        """A interface to record the latest message of each tracked topic."""
        # 不反序列化, 只记下这条 message 的解析格式与定位, 作为该 topic 的"当前最新值"
        if topic_name in self.all_topics_set:
            self._cur_msg_record[topic_name] = [msg_type, msg_def, log_time, locations]

    def _snapshot_topics(self, phase_topics: frozenset) -> dict:
        """Snapshot the message chosen for each topic at one sampling instant."""
        # 只取已出现过的 topic, 没出现过说明该时刻无可选 message
        return {topic: self._cur_msg_record[topic]
                for topic in phase_topics if topic in self._cur_msg_record}

    def _commit_episode(self):
        """Append the finished episode to the output json as one line."""
        separator = ",\n" if self.num_episodes else ""
        self._json_file.write(f'{separator}"{self.num_xy_pairs}": ')
        json.dump(self._episode_samples, self._json_file)
        self.num_episodes += 1
        self.total_samples += len(self._episode_samples)

    def convert_single_mcap(self):
        """Pick the state/action message of every sample while the player replays."""
        is_recording = False
        previous_buttons = None
        episode_state_target_t = None
        episode_action_target_t = None
        is_in_adding_phase = False
        sample_idx = 0
        pending_state_snapshot = None   # 已命中 state 窗口但 action 还没配齐的那份快照

        # 播放器只额外要一路 episode 起止信号, 它不属于训练数据, 故不在 USER_SELECTED_TOPICS 里
        for topic, msg_type, msg_def, log_time, locations in \
                self.player.play_messages(extra_topics=(MCAP_SIGNAL_TOPIC,)):
            timestamp = log_time / 1e9      # log_time(ns) 是该 message 的唯一身份

            # 复刻原 non_recording_filter: 非录制期间只有按键信号可见
            if not is_recording and topic != MCAP_SIGNAL_TOPIC:
                continue

            self.update_messages(topic, msg_type, msg_def, log_time, locations)

            if topic == MCAP_SIGNAL_TOPIC:
                try:
                    buttons = self.fetcher.fetch_message(msg_type, msg_def, locations)[0].buttons

                    if len(buttons) > max(X_BUTTON, Y_BUTTON):
                        if previous_buttons is not None:
                            if (previous_buttons[X_BUTTON] == 0 and buttons[X_BUTTON] == 1 and not is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"🔴 Start #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                sample_idx = 0
                                pending_state_snapshot = None
                                self._episode_samples = {}

                            elif (previous_buttons[X_BUTTON] == 0 and buttons[X_BUTTON] == 1 and is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"🔴 Re-record #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                sample_idx = 0
                                pending_state_snapshot = None
                                # 重录: 之前攒下的 sample 整段作废
                                self._episode_samples = {}

                            if (previous_buttons[Y_BUTTON] == 0 and buttons[Y_BUTTON] == 1 and is_recording):
                                is_recording = False
                                end_time = timestamp
                                duration = end_time - start_time
                                self.logger.info(
                                    f"⏹️ Stop TimeRange: {start_time:.3f} to {end_time:.3f} seconds. Duration: {duration:.3f} seconds")

                                if sample_idx < MIN_EPISODE_LENGTH:
                                    # 样本数不足, 整段丢弃
                                    self.logger.info(
                                        f"⏹️ Discard #Episode {self.num_xy_pairs}: "
                                        f"only {sample_idx} samples (< {MIN_EPISODE_LENGTH})")
                                else:
                                    self._commit_episode()
                                self._episode_samples = {}

                                self.num_xy_pairs += 1

                                episode_state_target_t = None
                                episode_action_target_t = None

                        previous_buttons = list(buttons)

                except Exception as e:
                    error_msg = f"Error processing Joy message: {e}\n"
                    error_msg += f"Traceback (most recent call last):\n"
                    error_msg += traceback.format_exc()
                    self.logger.error(error_msg)

            if is_recording:
                if timestamp < episode_state_target_t:
                    continue

                elif episode_state_target_t <= timestamp <= episode_action_target_t and not is_in_adding_phase:
                    # obs/state 先挂起, 配齐 action 才算一个完整 sample
                    pending_state_snapshot = self._snapshot_topics(self.state_phase_topics)
                    is_in_adding_phase = True

                elif timestamp > episode_action_target_t and is_in_adding_phase:
                    sample_topics = pending_state_snapshot | self._snapshot_topics(self.action_topics_set)
                    # topic 不齐的 sample 无法成为训练样本, 直接跳过且不占用 sample_idx
                    if len(sample_topics) == len(self.all_topics_set):
                        self._episode_samples[sample_idx] = sample_topics
                        sample_idx += 1
                    pending_state_snapshot = None

                    time_gap = timestamp - episode_action_target_t
                    skipped_frames = time_gap // self.frame_duration

                    episode_state_target_t += (skipped_frames + 1) * self.frame_duration
                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration
                    is_in_adding_phase = False

                elif timestamp > episode_action_target_t and not is_in_adding_phase:
                    time_gap = timestamp - episode_action_target_t
                    skipped_frames = time_gap // self.frame_duration + 1

                    episode_state_target_t += skipped_frames * self.frame_duration
                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                    pending_state_snapshot = None

        # 末尾未按 Y 的 episode 可能不完整, 整段丢弃
        if is_recording:
            self.logger.info(f"⏹️ Discard #Episode {self.num_xy_pairs}: "
                             f"{sample_idx} samples, no end button")
            self._episode_samples = {}
