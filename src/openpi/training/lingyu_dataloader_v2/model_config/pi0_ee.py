######
# Action Chunk Length
######
ACTION_CHUNK_LENGTH = 30


######
# State Concat and Action Concat Config
######
# 拼接顺序即模型 state/action 向量的维度顺序: (mcap topic, 该 topic 取的字段)
# 字段名即 ros2 message 上的属性名, 由 ros2_message_filter 直接取值, 不再需要其他登记表
#
# 与 teleavatar_v2_ee_policy.TeleavatarEEInputs 的 62 维输入布局逐位对齐(LeRobot v2 约定,
# 同 generate_tars.py 的 62 维布局):
#   [ 0:16] positions   左臂7 左夹爪1 右臂7 右夹爪1
#   [16:32] velocities  同上
#   [32:48] efforts     同上(夹爪 effort 落在 idx 39/47)
#   [48:55] 左 EE pose  xyz(3) + 四元数 xyzw(4)
#   [55:62] 右 EE pose  同上
# EE policy 只读 state 的夹爪开度(idx 7/15)与 EE 位姿(48-61)、action 的夹爪 effort(idx 39/47)
# 与目标 EE 位姿(48-61), 其余维度为布局对齐的死维度, 按录制原样拼入。
# state 取 measured topic(joint_states + current_ee_pose), action 取 commanded topic
# (joint_cmd + target_ee_pose), 两侧同布局故下标互相兼容。

# state: measured topic, EE 反馈位姿由 current_ee_pose 提供(state 角色, 每 sample 1 条)
STATE_CONCAT = (
    # [0:16] position
    ('/left_arm/joint_states',     'position'),
    ('/left_gripper/joint_states', 'position'),   # 左夹爪开度, policy 读 idx 7 换算 trigger
    ('/right_arm/joint_states',    'position'),
    ('/right_gripper/joint_states', 'position'),  # 右夹爪开度, policy 读 idx 15
    # [16:32] velocity (死维度)
    ('/left_arm/joint_states',     'velocity'),
    ('/left_gripper/joint_states', 'velocity'),
    ('/right_arm/joint_states',    'velocity'),
    ('/right_gripper/joint_states', 'velocity'),
    # [32:48] effort (死维度)
    ('/left_arm/joint_states',     'effort'),
    ('/left_gripper/joint_states', 'effort'),
    ('/right_arm/joint_states',    'effort'),
    ('/right_gripper/joint_states', 'effort'),
    # [48:55] 左 EE 位姿: Point(3) + Quaternion xyzw(4), policy 由它构造绝对 EE 帧
    ('/left_arm/current_ee_pose', 'position'),
    ('/left_arm/current_ee_pose', 'orientation'),
    # [55:62] 右 EE 位姿
    ('/right_arm/current_ee_pose', 'position'),
    ('/right_arm/current_ee_pose', 'orientation'),
)

# action: commanded topic, 目标 EE 位姿由 target_ee_pose 提供(action 角色, 每 sample 100 步)
ACTION_CONCAT = (
    # [0:16] position (死维度; 夹爪指令在 effort 块, 录制里 position 恒 0)
    ('/left_arm/joint_cmd',     'position'),
    ('/left_gripper/joint_cmd', 'position'),
    ('/right_arm/joint_cmd',    'position'),
    ('/right_gripper/joint_cmd', 'position'),
    # [16:32] velocity (死维度)
    ('/left_arm/joint_cmd',     'velocity'),
    ('/left_gripper/joint_cmd', 'velocity'),
    ('/right_arm/joint_cmd',    'velocity'),
    ('/right_gripper/joint_cmd', 'velocity'),
    # [32:48] effort: 夹爪 effort 落在 idx 39/47, policy 读它换算 trigger
    ('/left_arm/joint_cmd',     'effort'),
    ('/left_gripper/joint_cmd', 'effort'),
    ('/right_arm/joint_cmd',    'effort'),
    ('/right_gripper/joint_cmd', 'effort'),
    # [48:55] 左 EE 目标位姿, policy 相对当前 EE 帧求相对位姿
    ('/left_arm/target_ee_pose', 'position'),
    ('/left_arm/target_ee_pose', 'orientation'),
    # [55:62] 右 EE 目标位姿
    ('/right_arm/target_ee_pose', 'position'),
    ('/right_arm/target_ee_pose', 'orientation'),
)
