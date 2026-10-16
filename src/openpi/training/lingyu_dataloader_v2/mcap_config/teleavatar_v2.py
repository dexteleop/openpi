"""
用于 TeleAvatarV2 topics 对齐 Standard topics
{Standard topics： TeleAvatarV2 topics}
TeleAvatarV2 topics 只能对应一个 Standard topics
一个 Standard topics 能够对应多个 TeleAvatarV2 topics
"""
######
# Standard Topics Config
######

MCAP_TOPICS_MAPPING = { # 标准 topic: MCAP topic
    ## 常用Topic
    # 摄像头
    '/ffmpeg': (
        '/xr_video_topic/ffmpeg',
        '/right/color/image_raw/ffmpeg',
        '/left/color/image_raw/ffmpeg',
    ),

    # 状态
    "/left_arm/states": (
        "/left_arm/joint_states",
        "/left_gripper/joint_states",
        "/left_arm/current_ee_pose",
    ),
    "/right_arm/states": (
        "/right_arm/joint_states",
        "/right_gripper/joint_states",
        "/right_arm/current_ee_pose",
    ),
    "/other/states": (
        "/chassis/joint_states",
        "/kinco/actual_velocity",
    ),

    # 指令
    "/left_arm/cmd": (
        "/left_arm/joint_cmd",
        "/left_gripper/joint_cmd",
        "/left_arm/target_ee_pose",
        "/left_target_ee_pose",
    ),
    "/right_arm/cmd": (
        "/right_arm/joint_cmd",
        "/right_gripper/joint_cmd",
        "/right_arm/target_ee_pose",
        "/right_target_ee_pose",
    ),
    "/other/cmd": (
        "/chassis/joint_cmd",
        "/kinco/cmd_velocity",
    ),
}


MODEL_SELECTED_TOPICS = { # 标准topic: 该topic在训练样本中的角色 obs/state/action
    "/ffmpeg": "obs",

    "/left_arm/states": "state",
    "/right_arm/states": "state",
    "/other/states": "state",

    "/left_arm/cmd": "action",
    "/right_arm/cmd": "action",
    "/other/cmd": "action",
}


######
# MCAP Topics Config
######

# Video Decoder Config
MCAP_VIDEO_TOPICS_GOP = { # MCAP topics
    '/xr_video_topic/ffmpeg': 45,
    '/right/color/image_raw/ffmpeg': 45,
    '/left/color/image_raw/ffmpeg': 45,
}
