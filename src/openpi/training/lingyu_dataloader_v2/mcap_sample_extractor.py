# Copyright (c) 2026 Dexteleop Intelligence (Lingyu Intelligence)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Split one mcap into episodes by the X/Y keys, and pick each topic's message for every sample.

Only two state machines are kept:
    1. X/Y keys -> episode start/end
    2. time window -> the collection moments of each sample's state / action
At each collection moment, the "latest message" of every topic at that moment is recorded.

Data topics and their obs/state/action roles all come from USER_SELECTED_TOPICS, filtered by MCAP_Player;
the only thing that must be deserialized is the Joy key message marking episode start/end, read back via MCAP_Message_Fetcher.

All samples of one episode accumulate in memory first, and are written out in one piece only if it ends normally with the Y key
and has at least MIN_EPISODE_LENGTH samples; therefore the result never contains samples from redone/aborted episodes,
episodes without a trailing Y, or episodes that are too short.

This module only produces in-memory objects and does no persistence: each finished episode is yielded once as
(source_episode_seq, samples), handed by build_sample_idx.py to Iceberg for saving.
samples is [sample0, sample1, ...], each sample being {topic: [msg_type, msg_def, log_time, locations]};
the three roles differ only in the number of locations::
    {obs topic:    [msg_type, msg_def, log_time, [loca0, loca1, ...]],
     state topic:  [msg_type, msg_def, log_time, [loca0]],
     action topic: [msg_type, msg_def, log_time, [loca0, ..., loca29]]}
obs is a video frame; with inter-frame coding it must trace back to a keyframe, so locations are one whole GOP; each state message is independent, so just one.
action's locations are [current action, the next ACTION_CHUNK_LENGTH-1 actions], each's locating info;
when future actions run out near the episode end, the episode's last action repeats (hold-last), matching lingyu_dataloader's padding.
source_episode_seq is the X/Y pair sequence number within one mcap; discarded episodes do not appear, so numbers may be non-contiguous,
but they are constant and reproducible for the same mcap, serving as the episode's stable identity and sort key.

Usage::
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
MIN_EPISODE_LENGTH = 180 # An episode needs at least 4 seconds
ACTION_OFFSET_RATIO = 1.0 / 3.0
# Episode start/end key signals: kept hardcoded here and not read from mcap_config, as this and build_sample_idx.py are two independent sources
MCAP_SIGNAL_TOPIC, X_BUTTON, Y_BUTTON = set_episode_signal()
# One sample's action sequence length: [current action, the next ACTION_CHUNK_LENGTH-1 actions]
ACTION_CHUNK_LENGTH = load_action_chunk_length()

logger = logging.getLogger(__name__)


class MCAPSampleExtractor:
    """Replay one mcap and record which message each sample selects per topic."""

    def __init__(self, mcap_path: str, fps: int = 45):
        self.mcap_path = mcap_path
        self.mcap_name = Path(mcap_path).name
        # source_id: one mcap is one source; the file name without its extension already identifies one recording uniquely
        self.source_id = Path(mcap_path).stem
        self.fps = fps
        self.frame_duration = 1.0 / self.fps

        # User Selected Topics: {mcap topic: role obs/state/action}
        mcap_topics_role = filter_topics()
        self.all_topics_set = frozenset(mcap_topics_role)
        # The state collection window covers observation and state, the action window covers actions; the two are disjoint and together cover all selected topics
        self.state_topics_set = frozenset(
            topic for topic, role in mcap_topics_role.items() if role in ('obs', 'state'))
        self.action_topics_set = frozenset(
            topic for topic, role in mcap_topics_role.items() if role == 'action')

        # Episode tracking
        self.num_xy_pairs = 0
        self.total_samples = 0          # Total samples yielded so far
        self.num_episodes = 0           # Total episodes yielded so far
        # message tracking: topic -> [msg_type, msg_def, log_time, locations], i.e. the latest message of that topic
        self._cur_msg_record = {}
        # Samples accumulated for the current episode, yielded in one piece only when it ends normally with the Y key; redone or unfinished episodes are dropped entirely
        self._episode_samples = {}

        logger.info(f"\n=== Processing {self.mcap_name} ===")

        # The player has already done topic filtering and video GOP accumulation; the file is opened only on the first iteration of play_messages()
        self.player = MCAP_Player(mcap_path)
        # Key messages need deserialization; the fetcher carries its own typestore and custom type registrations
        self.fetcher = MCAP_Message_Fetcher()

    def update_messages(self, topic_name: str, msg_type: str, msg_def: str,
                        log_time: int, locations: list):
        """A interface to record the latest message of each tracked topic."""
        # No deserialization: just record this message's parsed format and locating info as the topic's "current latest value"
        if topic_name in self.all_topics_set:
            self._cur_msg_record[topic_name] = [msg_type, msg_def, log_time, locations]

    def _snapshot_topics(self, phase_topics: frozenset) -> dict:
        """Snapshot the message chosen for each topic at one sampling instant."""
        # Take only topics that have appeared; an absent topic means no selectable message at that moment
        snapshot = {}
        for topic in phase_topics:  # Only look at topics this phase cares about
            if topic in self._cur_msg_record:  # Only copy topics that have seen messages
                snapshot[topic] = self._cur_msg_record[topic]  # Copy its latest message
        return snapshot

    def _expand_action_chunks(self):
        """Extend each action's locations to [current action, next ACTION_CHUNK_LENGTH-1 actions]."""
        # Future actions are later samples' actions, so expansion is only possible after the episode ends; when the tail runs short the last sample repeats (hold-last)
        # Rewriting in sample_idx ascending order guarantees a referenced future sample is not yet expanded; actions are not video, one location per message
        last_sample_idx = len(self._episode_samples) - 1  # default: len(self._episode_samples)>30
        for sample_idx, sample_topics in self._episode_samples.items():
            for topic in self.action_topics_set:
                # Splice the next action_chunk actions into the current sample's index
                chunk_locations = [
                    self._episode_samples[min(sample_idx + step, last_sample_idx)][topic][-1][0]
                    for step in range(ACTION_CHUNK_LENGTH)
                ]
                # Replace the whole record with a new list, never modifying locations in place: adjacent samples may share the same message record
                msg_type, msg_def, log_time, _ = sample_topics[topic]
                sample_topics[topic] = [msg_type, msg_def, log_time, chunk_locations]

    def _take_episode_samples(self) -> list[dict]:
        """Expand action chunks and hand the finished episode out as a sample list."""
        self._expand_action_chunks()
        # sample_idx was already filled in contiguously from 0; after converting to a list, the list index is sample_idx, so this dict layer is unnecessary
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
        sample_state_snapshot = None  # Holds the state snapshot while building one sample

        # The player needs only one extra stream of episode start/end signals; they are not training data, hence absent from USER_SELECTED_TOPICS
        for topic, msg_type, msg_def, log_time, locations in \
                self.player.play_messages(extra_topics=(MCAP_SIGNAL_TOPIC,)):
            timestamp = int(log_time) / 1e9  # log_time is a string; convert to int then to seconds

            # Replicate the original non_recording_filter: outside recording only the key signals are visible
            if not is_recording and topic != MCAP_SIGNAL_TOPIC:
                continue

            self.update_messages(topic, msg_type, msg_def, log_time, locations)

            if topic == MCAP_SIGNAL_TOPIC:
                finished_episode = None  # The episode ended by the Y key, yielded outside the try
                try:
                    buttons = self.fetcher.fetch_message(msg_type, msg_def, locations)[0].buttons

                    if len(buttons) > max(X_BUTTON, Y_BUTTON):
                        if previous_buttons is not None:
                            if (previous_buttons[X_BUTTON] == 0 and buttons[X_BUTTON] == 1 and not is_recording):
                                is_recording = True
                                start_time = timestamp
                                logger.info(f"🔴 Start #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                sample_idx = 0
                                sample_state_snapshot = None
                                self._episode_samples = {}

                            elif (previous_buttons[X_BUTTON] == 0 and buttons[X_BUTTON] == 1 and is_recording):
                                is_recording = True
                                start_time = timestamp
                                logger.info(f"🔴 Re-record #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                sample_idx = 0
                                sample_state_snapshot = None
                                self._episode_samples = {} # Redo: all previous samples of this episode are void

                            if (previous_buttons[Y_BUTTON] == 0 and buttons[Y_BUTTON] == 1 and is_recording):
                                is_recording = False
                                end_time = timestamp
                                duration = end_time - start_time
                                logger.info(
                                    f"⏹️ Stop TimeRange: {start_time:.3f} to {end_time:.3f} seconds. Duration: {duration:.3f} seconds")

                                if sample_idx < MIN_EPISODE_LENGTH:
                                    # Not enough samples, drop the whole episode
                                    logger.info(
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
                    logger.error(error_msg)

                # Yield outside the try: do not let this class's Joy exception handling swallow exceptions raised by the consumer
                if finished_episode is not None:
                    yield finished_episode

            if is_recording:
                if timestamp < episode_state_target_t:
                    continue

                elif episode_state_target_t <= timestamp <= episode_action_target_t and not is_in_adding_phase:
                    # obs/state of the sample captured
                    sample_state_snapshot = self._snapshot_topics(self.state_topics_set)
                    is_in_adding_phase = True

                elif timestamp > episode_action_target_t and is_in_adding_phase:
                    # action of the sample captured, combined with the state snapshot into a sample
                    sample_action_snapshot = self._snapshot_topics(self.action_topics_set)
                    sample_topics = sample_state_snapshot | sample_action_snapshot
                    # A sample missing topics cannot become a training sample; skip it without consuming a sample_idx
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

        # An episode without a trailing Y may be incomplete; drop the whole episode
        if is_recording:
            logger.info(f"⏹️ Discard #Episode {self.num_xy_pairs}: "
                        f"{sample_idx} samples, no end button")
            self._episode_samples = {}
