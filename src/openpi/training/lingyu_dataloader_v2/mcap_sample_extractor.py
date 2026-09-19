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

本模块只产出内存对象, 不负责任何落盘: 每收尾一个 episode 就 yield 一次
(source_episode_seq, samples), 由 build_sample_idx.py 交给 Iceberg 保存。
samples 是 [sample0, sample1, ...], 每个 sample 为 {topic: [msg_type, msg_def, log_time, locations]},
三种角色只在 locations 的条数上不同::
    {obs topic:    [msg_type, msg_def, log_time, [loca0, loca1, ...]],
     state topic:  [msg_type, msg_def, log_time, [loca0]],
     action topic: [msg_type, msg_def, log_time, [loca0, ..., loca29]]}
obs 是视频帧, 帧间编码时要回溯到关键帧, 故 locations 为一整段 GOP; state 每条消息独立, 只有一条。
action 的 locations 是 [当前动作, 后续 ACTION_CHUNK_LENGTH-1 个动作] 各自的定位,
episode 末尾未来动作不足时重复该 episode 最后一个动作(hold-last), 与 lingyu_dataloader 的补齐方式一致。
source_episode_seq 为该 mcap 内的 X/Y 对序号, 丢弃的 episode 不出现, 故序号可能不连续,
但对同一个 mcap 恒定可复现, 故可作为 episode 的稳定身份与排序依据。

用法::
    for source_episode_seq, samples in MCAPSampleExtractor(mcap_path).iter_episodes():
        ...
"""

from pathlib import Path
import logging
import traceback

from openpi.training.lingyu_dataloader_v2.utils.mcap_player import MCAP_Player
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher
from openpi.training.lingyu_dataloader_v2.utils.mcap_topics_filter import filter_topics
from openpi.training.lingyu_dataloader_v2.utils.episode_signals import set_episode_signal
from openpi.training.lingyu_dataloader_v2.model_config.config import load_action_chunk_length

# Constants
MIN_EPISODE_LENGTH = 30
ACTION_OFFSET_RATIO = 1.0 / 3.0
# episode 起止按键信号: 与 build_sample_idx.py 分属两套独立来源, 故此处仍硬编码, 不读 mcap_config
MCAP_SIGNAL_TOPIC, X_BUTTON, Y_BUTTON = set_episode_signal()
# 一个 sample 的动作序列长度: [当前动作, 后续 ACTION_CHUNK_LENGTH-1 个动作]
ACTION_CHUNK_LENGTH = load_action_chunk_length()


class MCAPSampleExtractor:
    """Replay one mcap and record which message each sample selects per topic."""

    def __init__(self, mcap_path: str, fps: int = 30):
        self.mcap_path = mcap_path
        self.mcap_name = Path(mcap_path).name
        # source_id: 一个 mcap 即一个 source, 文件名去掉后缀已能唯一标识一次录制
        self.source_id = Path(mcap_path).stem
        self.fps = fps
        self.frame_duration = 1.0 / self.fps

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # User Selected Topics: {mcap topic: 角色 obs/state/action}
        mcap_topics_role = filter_topics()
        self.all_topics_set = frozenset(mcap_topics_role)
        # state 采集窗口覆盖观测与状态, action 采集窗口覆盖动作, 两者互斥且合起来即全部选定 topic
        self.state_topics_set = frozenset(
            topic for topic, role in mcap_topics_role.items() if role in ('obs', 'state'))
        self.action_topics_set = frozenset(
            topic for topic, role in mcap_topics_role.items() if role == 'action')

        # Episode tracking
        self.num_xy_pairs = 0
        self.total_samples = 0          # 已 yield 的 sample 总数
        self.num_episodes = 0           # 已 yield 的 episode 总数
        # message tracking: topic -> [msg_type, msg_def, log_time, locations], 即该 topic 的最新一条 message
        self._cur_msg_record = {}
        # 当前 episode 已攒下的 sample, 按 Y 键正常收尾才整段产出; 重录或未收尾则整段丢弃
        self._episode_samples = {}

        self.logger.info(f"\n=== Processing {self.mcap_name} ===")

        # 播放器已完成 topic 筛选与视频 GOP 累积, 文件在首次迭代 play_messages() 时才打开
        self.player = MCAP_Player(mcap_path)
        # 按键消息需反序列化, fetcher 自带 typestore 与自定义类型注册
        self.fetcher = MCAP_Message_Fetcher()

    def update_messages(self, topic_name: str, msg_type: str, msg_def: str,
                        log_time: int, locations: list):
        """A interface to record the latest message of each tracked topic."""
        # 不反序列化, 只记下这条 message 的解析格式与定位, 作为该 topic 的"当前最新值"
        if topic_name in self.all_topics_set:
            self._cur_msg_record[topic_name] = [msg_type, msg_def, log_time, locations]

    def _snapshot_topics(self, phase_topics: frozenset) -> dict:
        """Snapshot the message chosen for each topic at one sampling instant."""
        # 只取已出现过的 topic, 没出现过说明该时刻无可选 message
        snapshot = {}
        for topic in phase_topics:  # 只看本阶段关心的topic
            if topic in self._cur_msg_record:  # 这个topic出现过消息才抄
                snapshot[topic] = self._cur_msg_record[topic]  # 抄下它的最新消息
        return snapshot

    def _expand_action_chunks(self):
        """Extend each action's locations to [current action, next ACTION_CHUNK_LENGTH-1 actions]."""
        # 未来动作即后续 sample 的 action, 故整段收尾后才能展开; 末尾不足时重复最后一个 sample(hold-last)
        # 按 sample_idx 升序改写, 读到的未来 sample 一定还没展开; 动作非视频, 每条消息只有一个 location
        last_sample_idx = len(self._episode_samples) - 1  # default: len(self._episode_samples)>30
        for sample_idx, sample_topics in self._episode_samples.items():
            for topic in self.action_topics_set:
                # 将未来 action_chunk 个动作拼接到当前样本索引中
                chunk_locations = [
                    self._episode_samples[min(sample_idx + step, last_sample_idx)][topic][-1][0]
                    for step in range(ACTION_CHUNK_LENGTH)
                ]
                # 整条记录换成新 list, 不就地改 locations: 相邻 sample 可能共用同一条 message 记录
                msg_type, msg_def, log_time, _ = sample_topics[topic]
                sample_topics[topic] = [msg_type, msg_def, log_time, chunk_locations]

    def _take_episode_samples(self) -> list[dict]:
        """Expand action chunks and hand the finished episode out as a sample list."""
        self._expand_action_chunks()
        # sample_idx 本就是从 0 起连续填入的, 转成 list 后下标即 sample_idx, 不再需要这层 dict
        samples = [self._episode_samples[sample_idx]
                   for sample_idx in range(len(self._episode_samples))]
        self._episode_samples = {}
        self.num_episodes += 1
        self.total_samples += len(samples)
        return samples

    def iter_episodes(self):
        """Yield (source_episode_seq, samples) for every complete episode while replaying."""
        is_recording = False
        previous_buttons = None
        episode_state_target_t = None
        episode_action_target_t = None
        is_in_adding_phase = False
        sample_idx = 0
        sample_state_snapshot = None  # 在构建一个 sample 时，存放 state 快照

        # 播放器只额外要一路 episode 起止信号, 它不属于训练数据, 故不在 USER_SELECTED_TOPICS 里
        for topic, msg_type, msg_def, log_time, locations in \
                self.player.play_messages(extra_topics=(MCAP_SIGNAL_TOPIC,)):
            timestamp = int(log_time) / 1e9  # log_time 为字符串, 转 int 后再换算秒

            # 复刻原 non_recording_filter: 非录制期间只有按键信号可见
            if not is_recording and topic != MCAP_SIGNAL_TOPIC:
                continue

            self.update_messages(topic, msg_type, msg_def, log_time, locations)

            if topic == MCAP_SIGNAL_TOPIC:
                finished_episode = None  # 按 Y 键收尾的 episode, 在 try 外 yield
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
                                sample_state_snapshot = None
                                self._episode_samples = {}

                            elif (previous_buttons[X_BUTTON] == 0 and buttons[X_BUTTON] == 1 and is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"🔴 Re-record #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                sample_idx = 0
                                sample_state_snapshot = None
                                self._episode_samples = {} # 重录: 之前的 sample 整段作废

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
                                    self._episode_samples = {}
                                else:
                                    finished_episode = (self.num_xy_pairs,
                                                        self._take_episode_samples())

                                self.num_xy_pairs += 1

                                episode_state_target_t = None
                                episode_action_target_t = None

                        previous_buttons = list(buttons)

                except Exception as e:
                    error_msg = f"Error processing Joy message: {e}\n"
                    error_msg += f"Traceback (most recent call last):\n"
                    error_msg += traceback.format_exc()
                    self.logger.error(error_msg)

                # 在 try 外 yield: 别让本类的 Joy 异常处理吞掉消费者抛出的异常
                if finished_episode is not None:
                    yield finished_episode

            if is_recording:
                if timestamp < episode_state_target_t:
                    continue

                elif episode_state_target_t <= timestamp <= episode_action_target_t and not is_in_adding_phase:
                    # sample 中的 obs/state 完成抓拍
                    sample_state_snapshot = self._snapshot_topics(self.state_topics_set)
                    is_in_adding_phase = True

                elif timestamp > episode_action_target_t and is_in_adding_phase:
                    # sample 中的 action 完成抓拍，并与 状态抓拍 构建 sample
                    sample_action_snapshot = self._snapshot_topics(self.action_topics_set)
                    sample_topics = sample_state_snapshot | sample_action_snapshot
                    # topic 不齐的 sample 无法成为训练样本, 直接跳过且不占用 sample_idx
                    if len(sample_topics) == len(self.all_topics_set):
                        self._episode_samples[sample_idx] = sample_topics
                        sample_idx += 1
                    sample_state_snapshot = None

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

                    sample_state_snapshot = None

        # 末尾未按 Y 的 episode 可能不完整, 整段丢弃
        if is_recording:
            self.logger.info(f"⏹️ Discard #Episode {self.num_xy_pairs}: "
                             f"{sample_idx} samples, no end button")
            self._episode_samples = {}
