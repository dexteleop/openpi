"""
用于 TeleAvatarV2 topics 对齐 Standard topics
{Standard topics： TeleAvatarV2 topics}
TeleAvatarV2 topics 只能对应一个 Standard topics
一个 Standard topics 能够对应多个 TeleAvatarV2 topics
"""
TELEAVATAR_V2_MCAP_TOPICS_MAPPING = { # 标准 topic: MCAP topic
    ## 常用Topic
    # 三路摄像头
    '/xr_video_topic/ffmpeg': ('/xr_video_topic/ffmpeg',),
    '/right/color/image_raw/ffmpeg': ('/right/color/image_raw/ffmpeg',),
    '/left/color/image_raw/ffmpeg': ('/left/color/image_raw/ffmpeg',),

    # 双臂状态
    '/left_arm/joint_states': ('/left_arm/joint_states',),
    '/left_arm/current_ee_pose': ('/left_arm/current_ee_pose',),
    '/left_gripper/joint_states': ('/left_gripper/joint_states',),
    '/right_arm/joint_states': ('/right_arm/joint_states',),
    '/right_arm/current_ee_pose': ('/right_arm/current_ee_pose',),
    '/right_gripper/joint_states': ('/right_gripper/joint_states',),

    # 双臂指令
    '/left_arm/joint_cmd': ('/left_arm/joint_cmd',),
    '/left_arm/target_ee_pose': ('/left_arm/target_ee_pose',),
    '/left_gripper/joint_cmd': ('/left_gripper/joint_cmd',),
    '/right_arm/joint_cmd': ('/right_arm/joint_cmd',),
    '/right_arm/target_ee_pose': ('/right_arm/target_ee_pose',),
    '/right_gripper/joint_cmd': ('/right_gripper/joint_cmd',),

    ## 不常用Topic
    # 底盘指令
    '/chassis/joint_cmd': ('/chassis/joint_cmd',),
    '/chassis/joint_states': ('/chassis/joint_states',),
    '/chassis_target_vel': ('/chassis_target_vel',),
    '/kinco/cmd_velocity': ('/kinco/cmd_velocity',),
    '/kinco/actual_velocity': ('/kinco/actual_velocity',),
    '/kinco/motor_state': ('/kinco/motor_state',),

    # 以下Topic经常用不到
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


TELEAVATAR_V2_VIDEO_TOPICS_GOP = { # MCAP中存在的topics
    '/xr_video_topic/ffmpeg': 45,
    '/right/color/image_raw/ffmpeg': 45,
    '/left/color/image_raw/ffmpeg': 45,
}


USER_SELECTED_TOPICS = { # 标准topic: 该topic在训练样本中的角色 obs/state/action
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


EPISODE_SIGNAL = { # MCAP中存在的topics
    'topic_name': '/xr/left_hand_inputs',
    'start': 2,
    'end': 3,
}