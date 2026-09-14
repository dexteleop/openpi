"""统一配置入口: 指定当前使用哪台机器人, 并对外提供读取其 topic 配置的函数。

其他程序一律通过 load_topics_config() 拿配置, 不直接 import 某台机器人的配置文件,
这样换机器人时只需改本文件顶部那一行 import。

每个机器人配置文件必须提供同名的常量, 注意各常量用的是哪套 topic 命名:
    TELEAVATAV2_MCAP_TOPICS_MAPPING  {标准 topic: (mcap topic, ...)}  键标准名, 值 mcap 名
    USER_SELECTED_TOPICS             {标准 topic: obs/state/action}, 键用前需经映射表换成 mcap 名
    TELEAVATAV2_VIDEO_TOPICS_GOP     mcap topic 名, 直接可用
    EPISODE_SIGNAL                   mcap topic 名, 直接可用; {topic_name, start, end}
"""
from __future__ import annotations

from openpi.training.lingyu_dataloader_v2.robots_config import teleavatar_v2  # 换机器人只改这一行
# TODO:导入机器人配置文件


ROBOT = "teleavatar_v2" # TODO：选定使用的哪个机器人的mcap


#####
# 用于 MCAP_Player 播放 topic 的索引
#####

def load_topics_config() -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """返回当前机器人的 (mcap topic 映射表, {用户选定的标准 topic: 角色})。"""
    if ROBOT == "teleavatar_v2":
        return teleavatar_v2.TELEAVATAR_V2_MCAP_TOPICS_MAPPING, teleavatar_v2.USER_SELECTED_TOPICS
    # TODO： elif 其他机器人的常量配置
    else:
        raise ValueError(
            f"Cannot find the relevant robot configuration of {ROBOT!r} "
            f"in lingyu_dataloader_v2/robots_config."
        )


def load_video_topics_gop() -> dict[str, int]:
    """返回当前机器人的 {视频 topic: GOP 长度}, GOP>1 表示帧间编码, 需回溯到关键帧。"""
    if ROBOT == "teleavatar_v2":
        return teleavatar_v2.TELEAVATAR_V2_VIDEO_TOPICS_GOP
    # TODO： elif 其他机器人的常量配置
    else:
        raise ValueError(
            f"Cannot find the relevant robot configuration of {ROBOT!r} "
            f"in lingyu_dataloader_v2/robots_config."
        )


#####
# 用于 episode 确定起始和结束
#####

def load_episode_signal() -> dict:
    """返回当前机器人标记 episode 起止的按键信号 {topic_name, start, end}。"""
    if ROBOT == "teleavatar_v2":
        return teleavatar_v2.EPISODE_SIGNAL
    # TODO： elif 其他机器人的常量配置
    else:
        raise ValueError(
            f"Cannot find the relevant robot configuration of {ROBOT!r} "
            f"in lingyu_dataloader_v2/robots_config."
        )
