用于 其他机器人topics 对齐 TeleAvatar_v2 topics
{TeleAvatarV2 topics : 其他机器人topics}
其他机器人topics只能对应一个TeleAvatarV2 topics
一个 TeleAvatarV2 topics 能够对应多个 其他机器人topics

每个机器人对应一个配置文件，必须提供同名的四个常量。注意各常量用的是哪套 topic 命名，
只有 USER_SELECTED_TOPICS 写标准名、需要经映射表换算，其余两个直接写 mcap 中的实际名：
- TELEAVATAV2_MCAP_TOPICS_MAPPING：{标准 topic: (mcap topic, ...)}，键标准名、值 mcap 名
- USER_SELECTED_TOPICS：{用户选定要读取的**标准** topic: 角色}，角色取 obs/state/action，
  供下游按角色区分观测、状态与动作；键经映射表换成 mcap 名后仍可直接做成员判断
- TELEAVATAV2_VIDEO_TOPICS_GOP：{**mcap** 视频 topic: GOP 长度}，GOP>1 表示帧间编码，
  播放时需回溯到关键帧；未登记的 topic 按 GOP=1（每条消息可独立解码）处理
- EPISODE_SIGNAL：{topic_name, start, end}，topic_name 是**mcap** topic 名，
  start/end 为标记 episode 起止的按键索引

config.py 是统一配置入口：指定当前使用哪台机器人（顶部一行 import），并提供
load_topics_config() / load_video_topics_gop() / load_episode_signal() 返回上述常量。
其他程序一律调这些函数取配置，不直接 import 某台机器人的配置文件。
