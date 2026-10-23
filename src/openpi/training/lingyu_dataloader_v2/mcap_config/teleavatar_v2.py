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
    ## Commonly used topics
    # Three cameras
    '/xr_video_topic/ffmpeg': ('/xr_video_topic/ffmpeg',),
    '/right/color/image_raw/ffmpeg': ('/right/color/image_raw/ffmpeg',),
    '/left/color/image_raw/ffmpeg': ('/left/color/image_raw/ffmpeg',),

    # Dual-arm state
    '/left_arm/joint_states': ('/left_arm/joint_states',),
    '/left_arm/current_ee_pose': ('/left_arm/current_ee_pose',),
    '/left_gripper/joint_states': ('/left_gripper/joint_states',),
    '/right_arm/joint_states': ('/right_arm/joint_states',),
    '/right_arm/current_ee_pose': ('/right_arm/current_ee_pose',),
    '/right_gripper/joint_states': ('/right_gripper/joint_states',),

    # Dual-arm commands
    '/left_arm/joint_cmd': ('/left_arm/joint_cmd',),
    '/left_arm/target_ee_pose': ('/left_arm/target_ee_pose',),
    '/left_gripper/joint_cmd': ('/left_gripper/joint_cmd',),
    '/right_arm/joint_cmd': ('/right_arm/joint_cmd',),
    '/right_arm/target_ee_pose': ('/right_arm/target_ee_pose',),
    '/right_gripper/joint_cmd': ('/right_gripper/joint_cmd',),

    ## Rarely used topics
    # Chassis commands
    '/chassis/joint_cmd': ('/chassis/joint_cmd',),
    '/chassis/joint_states': ('/chassis/joint_states',),
    '/chassis_target_vel': ('/chassis_target_vel',),
    '/kinco/cmd_velocity': ('/kinco/cmd_velocity',),
    '/kinco/actual_velocity': ('/kinco/actual_velocity',),
    '/kinco/motor_state': ('/kinco/motor_state',),

    # The following topics are rarely used
    '/left_hand_position': ('/left_hand_position',),
    '/left_shoulder_position': ('/left_shoulder_position',),
    '/left_target_ee_pose': ('/left_target_ee_pose',),
    '/left_wrist_position': ('/left_wrist_position',),
    '/right_hand_position': ('/right_hand_position',),
    '/right_shoulder_position': ('/right_shoulder_position',),
    '/right_target_ee_pose': ('/right_target_ee_pose',),
    '/right_wrist_position': ('/right_wrist_position',),

    '/xr/hmd_pose': ('/xr/hmd_pose',),
    '/xr/left_aim_pose': ('/xr/left_aim_pose',),
    '/xr/left_hand_inputs': ('/xr/left_hand_inputs',),
    '/xr/right_aim_pose': ('/xr/right_aim_pose',),
    '/xr/right_hand_inputs': ('/xr/right_hand_inputs',),

    '/tf': ('/tf',),
    '/fsm_state': ('/fsm_state',),

    '/can0/motor_cmd': ('/can0/motor_cmd',),
    '/can0/motor_states': ('/can0/motor_states',),
    '/can1/motor_cmd': ('/can1/motor_cmd',),
    '/can1/motor_states': ('/can1/motor_states',),
    '/can2/motor_cmd': ('/can2/motor_cmd',),
    '/can3/motor_cmd': ('/can3/motor_cmd',),
    '/can3/motor_states': ('/can3/motor_states',),
    '/can4/motor_cmd': ('/can4/motor_cmd',),
    '/can4/motor_states': ('/can4/motor_states',),

    '/can0/error_code': ('/can0/error_code',),
    '/can1/error_code': ('/can1/error_code',),
    '/can3/error_code': ('/can3/error_code',),
    '/can4/error_code': ('/can4/error_code',),
    '/left_arm/error_code': ('/left_arm/error_code',),
    '/left_gripper/error_code': ('/left_gripper/error_code',),
    '/right_arm/error_code': ('/right_arm/error_code',),
    '/right_gripper/error_code': ('/right_gripper/error_code',),
}


MODEL_SELECTED_TOPICS = { # standard topic: the topic's role in training samples obs/state/action
    '/xr_video_topic/ffmpeg': 'obs',
    '/right/color/image_raw/ffmpeg': 'obs',
    '/left/color/image_raw/ffmpeg': 'obs',
    '/left_arm/joint_states': 'state',
    '/right_arm/joint_states': 'state',
    '/left_arm/joint_cmd': 'action',
    '/left_gripper/joint_cmd': 'action',
    '/right_arm/joint_cmd': 'action',
    '/right_gripper/joint_cmd': 'action',
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