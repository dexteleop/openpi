"""Verify the model's action_horizon cooperates with the index's ACTION_CHUNK_LENGTH: horizon <= chunk is enough for normal training.

Checks::
    1. The training config's action_horizon <= the ACTION_CHUNK_LENGTH used to build the index
    2. The actual location count of the action topic in the index (reflecting whether the index was rebuilt with the current ACTION_CHUNK_LENGTH)
    3. End-to-end sampling: with action_horizon the output action shape is (action_horizon, action_dim);
       without it the index's full length, proving the actions step count the model receives in training always equals action_horizon
"""
import openpi.training.config as _config
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import (
    LingyuDatasetV2, load_sample_topics)
from openpi.training.lingyu_dataloader_v2.model_config.config import load_action_chunk_length
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"


def test():
    # --- 1. Parameter consistency: horizon must not exceed the index's action sequence length ---
    chunk_length = load_action_chunk_length()
    action_horizon = _config.get_config(_CONFIG_NAME).model.action_horizon
    logger.info(f"训练 action_horizon={action_horizon}, 索引 ACTION_CHUNK_LENGTH={chunk_length}")
    assert action_horizon <= chunk_length, \
        f"action_horizon={action_horizon} 超过 ACTION_CHUNK_LENGTH={chunk_length}, 须重建索引或调小 horizon"

    # --- 2. Actual index steps: read the index only, no S3 access ---
    dataset = LingyuDatasetV2(load_images=False, action_horizon=action_horizon)
    episode = dataset.locate_episode(0)
    sample_topics = load_sample_topics(episode["data_file"], episode["row_group"], -episode["sample_offset"])
    index_steps = {topic: len(message["locations"]) for topic, message in sample_topics.items()
                   if topic in dataset.action_topics}
    logger.info(f"索引 sample 0 ({episode['episode_id']}) 各 action topic 的步数: {index_steps}")

    # --- 3. End-to-end sampling: after truncation the action step count == action_horizon ---
    item = dataset[0]
    logger.info(f"传 action_horizon={action_horizon}: action 形状 {tuple(item['action'].shape)}, "
                f"state 形状 {tuple(item['observation']['state'].shape)}")
    assert item["action"].shape[0] == action_horizon, "action 步数不等于 action_horizon"

    # Comparison: without truncation the output is the index's full-length action sequence
    full_item = LingyuDatasetV2(load_images=False)[0]
    logger.info(f"不传 action_horizon: action 形状 {tuple(full_item['action'].shape)}")
    assert full_item["action"].shape[0] == min(index_steps.values()), \
        "不截断时 action 步数应等于索引里的步数"
    logger.info("全部通过: 训练收到的 actions 步数恒等于 action_horizon, 与 ACTION_CHUNK_LENGTH 解耦")


if __name__ == "__main__":
    test()
