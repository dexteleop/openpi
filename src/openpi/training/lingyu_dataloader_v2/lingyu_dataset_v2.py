# 将一个sample中对应的所有的topic的message都找到，

# 根据 model_config 中的
# State Concat and Action Concat Config
# 完成模型需要的拼接结构，从而生成模型架构需要的 state 和 action

# 然后组建成一个dict({obs:..., state:..., action:...})
# dataset的格式以及输出接口参考
# /home/ubuntu/openpi/src/openpi/training/lingyu_dataloader/webdataset_load_tar.py
from __future__ import annotations

import numpy as np

from openpi.training.lingyu_dataloader_v2.utils.ros2_message_filter import ros2_message_filter
from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_topics_fields


def filter_sample_messages(topic_messages: dict[str, list]) -> dict[str, dict[str, np.ndarray]]:
    """
    输入 {topic_name: MCAP_Message_Fetcher.fetch_message() 返回的 message 列表},
    输出 {topic_name: {字段名: 一维数组}}。

    只处理登记在 STATE_and_ACTION_TOPICS_FIELDS 中的 state/action topic; 视频 topic
    不在配置内, 由 video_frame_decoder 解码, 在此跳过。
    message 列表取最后一条: state/action 的 locations 长度恒为 1, 视频 GOP 的末条才是当前帧。
    """
    state_and_action_fields = load_topics_fields()
    return {topic: ros2_message_filter(topic, messages[-1])
            for topic, messages in topic_messages.items()
            if topic in state_and_action_fields}
