"""
Map TeleAvatarV2 topics to Standard topics
{Standard topics： TeleAvatarV2 topics}
A TeleAvatarV2 topic maps to exactly one Standard topic
while one Standard topic can correspond to multiple TeleAvatarV2 topics
"""
######
# Standard Topics Config
######

MCAP_TOPICS_MAPPING = { # standard topic: MCAP topic
    # Cameras
    '/ffmpeg': (
        '/xr_video_topic/ffmpeg',
        '/right/color/image_raw/ffmpeg',
        '/left/color/image_raw/ffmpeg',
    ),

    # Dual-arm state
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

    # Dual-arm commands
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

# standard topic: the topic's role in training samples obs/state/action
MODEL_SELECTED_TOPICS = { 
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


# Sample Extractor Config
MCAP_EPISODE_TOPIC_SIGNAL = { # MCAP topic
    'topic_name': '/xr/left_hand_inputs',
    'start': 2,   # whether the value at array idx 2 shows an edge transition
    'end': 3,     # whether the value at array idx 3 shows an edge transition
}


# State and Action Message Filter Config
MCAP_STATE_and_ACTION_TOPICS_FIELDS = { # MCAP topic: required message fields (nested fields use flatten_msg's dotted paths)
    # Dual-arm state — sensor_msgs/JointState
    '/left_arm/joint_states':        ('position',),
    '/right_arm/joint_states':       ('position',),

    # Dual-arm action — sensor_msgs/JointState
    '/left_arm/joint_cmd':           ('position',),
    '/left_gripper/joint_cmd':       ('effort',),
    '/right_arm/joint_cmd':          ('position',),
    '/right_gripper/joint_cmd':      ('effort',),
}