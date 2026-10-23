"""验证模型 action_horizon 与索引 ACTION_CHUNK_LENGTH 的配合: horizon <= chunk 即可正常训练。

核对::
    1. 训练 config 的 action_horizon <= 索引构建用的 ACTION_CHUNK_LENGTH
    2. 索引里 action topic 的 locations 实际条数(反映索引是否已按当前 ACTION_CHUNK_LENGTH 重建)
    3. 端到端取样: 传 action_horizon 时输出 action 形状 (action_horizon, action_dim),
       不传时为索引全长, 证明训练时模型收到的 actions 步数恒等于 action_horizon
"""
import openpi.training.config_lingyu as _config
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import (
    LingyuDatasetV2, load_sample_topics)
from openpi.training.lingyu_dataloader_v2.model_config.config import load_action_chunk_length
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"


def test():
    # --- 1. 参数一致性: horizon 不得超过索引的动作序列长度 ---
    chunk_length = load_action_chunk_length()
    action_horizon = _config.get_config(_CONFIG_NAME).model.action_horizon
    logger.info(f"训练 action_horizon={action_horizon}, 索引 ACTION_CHUNK_LENGTH={chunk_length}")
    assert action_horizon <= chunk_length, \
        f"action_horizon={action_horizon} 超过 ACTION_CHUNK_LENGTH={chunk_length}, 须重建索引或调小 horizon"

    # --- 2. 索引实际步数: 只读索引, 不碰 S3 ---
    dataset = LingyuDatasetV2(load_images=False, action_horizon=action_horizon)
    episode = dataset.locate_episode(0)
    sample_topics = load_sample_topics(episode["data_file"], episode["row_group"], -episode["sample_offset"])
    index_steps = {topic: len(message["locations"]) for topic, message in sample_topics.items()
                   if topic in dataset.action_topics}
    logger.info(f"索引 sample 0 ({episode['episode_id']}) 各 action topic 的步数: {index_steps}")

    # --- 3. 端到端取样: 截断后 action 步数 == action_horizon ---
    item = dataset[0]
    logger.info(f"传 action_horizon={action_horizon}: action 形状 {tuple(item['action'].shape)}, "
                f"state 形状 {tuple(item['observation']['state'].shape)}")
    assert item["action"].shape[0] == action_horizon, "action 步数不等于 action_horizon"

    # 对照: 不截断时输出索引全长的动作序列
    full_item = LingyuDatasetV2(load_images=False)[0]
    logger.info(f"不传 action_horizon: action 形状 {tuple(full_item['action'].shape)}")
    assert full_item["action"].shape[0] == min(index_steps.values()), \
        "不截断时 action 步数应等于索引里的步数"
    logger.info("全部通过: 训练收到的 actions 步数恒等于 action_horizon, 与 ACTION_CHUNK_LENGTH 解耦")


if __name__ == "__main__":
    test()
