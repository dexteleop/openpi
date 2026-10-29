"""
Play the first 2000 messages of an mcap with MCAP_Player, hand the state/action ones to
MCAP_Message_Fetcher to get ros2 messages, then extract fields per config with ros2_message_filter,
verifying the extracted {field name: array} matches the message's own field values element-wise.

The key point: topics that are both sensor_msgs/JointState take different fields:
arms take position, grippers take effort, so the config must be queried per topic rather than taking one uniform field.
The config's keys are already mcap topic names, so they compare directly against MCAP_Player's topic_name without the mapping table.
"""
from collections import defaultdict
import numpy as np

from openpi.training.lingyu_dataloader_v2.utils.mcap_player import MCAP_Player
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import find_mcap_url_and_prompt_pairs
from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import ros2_message_filter
from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_state_and_action_topics_fields
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_MAX_PLAYED_MESSAGES = 2000  # Read only the first few messages; enough to cover all state/action topics


def test_ros2_message_filter():
    """Every state/action message among the first 2000 has its fields extracted correctly per config."""
    path = next(iter(find_mcap_url_and_prompt_pairs()))
    state_and_action_fields = load_mcap_state_and_action_topics_fields()
    player = MCAP_Player(path)
    fetcher = MCAP_Message_Fetcher()
    logger.info("播放 %s 前 %d 条 message, 待测 topic 数: %d",
                path, _MAX_PLAYED_MESSAGES, len(state_and_action_fields))

    filtered_counts = defaultdict(int)  # topic -> messages verified
    for played_idx, message in enumerate(player.play_messages()):
        if played_idx >= _MAX_PLAYED_MESSAGES:
            break
        topic_name, msg_type, msg_def, log_time, locations = message
        if topic_name not in state_and_action_fields:
            continue

        # player locating -> fetcher fetching the message -> filter extracting fields per config
        ros2_message = fetcher.fetch_message(msg_type, msg_def, locations)[-1]
        fields_data = ros2_message_filter(topic_name, ros2_message)

        # Keys must correspond one to one to the config's registered fields, and values must equal the message's own field values element-wise
        fields = state_and_action_fields[topic_name]
        assert fields_data.keys() == set(fields), \
            f"{topic_name} 返回的字段 {sorted(fields_data)} 与配置 {fields} 不符"
        for field_name, field_array in fields_data.items():
            expected = np.atleast_1d(np.asarray(getattr(ros2_message, field_name)))
            assert np.array_equal(field_array, expected), f"{topic_name}.{field_name} 取值不符"
            assert field_array.ndim == 1, f"{topic_name}.{field_name} 应为一维: {field_array.shape}"

        filtered_counts[topic_name] += 1
        logger.info("%-28s -> %s", topic_name,
                    {k: v.tolist() for k, v in fields_data.items()})

    for topic_name, count in sorted(filtered_counts.items()):
        logger.info("%-28s 共校验 %d 条", topic_name, count)
    assert filtered_counts.keys() == state_and_action_fields.keys(), \
        f"以下 topic 在前 {_MAX_PLAYED_MESSAGES} 条中未播放到: " \
        f"{sorted(state_and_action_fields.keys() - filtered_counts.keys())}"


if __name__ == "__main__":
    test_ros2_message_filter()
