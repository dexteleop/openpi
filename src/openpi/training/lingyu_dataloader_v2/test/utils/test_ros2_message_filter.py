"""
用 MCAP_Player 播放 mcap 的前 2000 条 message, 其中属于 state/action 的交给
MCAP_Message_Fetcher 取出 ros2 message, 再用 ros2_message_filter 按配置提取字段,
校验取出的 {字段名: 数组} 与 message 上该字段的值逐元素一致。

关键点是同为 sensor_msgs/JointState 的 topic 取的字段并不相同:
手臂取 position, 夹爪取 effort, 所以必须逐 topic 查配置, 不能统一取某个字段。
配置的键已是 mcap topic 名, 故与 MCAP_Player 播出的 topic_name 直接比对, 不经映射表。
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

_MAX_PLAYED_MESSAGES = 2000  # 只读前若干条, 够覆盖全部 state/action topic 即可


def test_ros2_message_filter():
    """前 2000 条 message 中的每条 state/action message 都能按配置取出正确的字段。"""
    path = next(iter(find_mcap_url_and_prompt_pairs()))
    state_and_action_fields = load_mcap_state_and_action_topics_fields()
    player = MCAP_Player(path)
    fetcher = MCAP_Message_Fetcher()
    logger.info("播放 %s 前 %d 条 message, 待测 topic 数: %d",
                path, _MAX_PLAYED_MESSAGES, len(state_and_action_fields))

    filtered_counts = defaultdict(int)  # topic -> 已校验的 message 条数
    for played_idx, message in enumerate(player.play_messages()):
        if played_idx >= _MAX_PLAYED_MESSAGES:
            break
        topic_name, msg_type, msg_def, log_time, locations = message
        if topic_name not in state_and_action_fields:
            continue

        # player 定位 -> fetcher 取 message -> filter 按配置提字段
        ros2_message = fetcher.fetch_message(msg_type, msg_def, locations)[-1]
        fields_data = ros2_message_filter(topic_name, ros2_message)

        # 键必须与配置登记的字段一一对应, 值必须逐元素等于 message 上该字段的值
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
