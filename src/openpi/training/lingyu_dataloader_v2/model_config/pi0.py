######
# Action Chunk Length
######
ACTION_CHUNK_LENGTH = 100


######
# State Concat and Action Concat Config
######
# The concat order is the dimension order of the model's state/action vectors: (mcap topic, the field taken from that topic)
# 这里的 (topic, 字段) 也是数据侧字段提取的唯一来源，ros2_message_filter 据此决定从消息中取哪些字段
STATE_CONCAT = (
    ('/left_arm/joint_states',   'position'),
    ('/right_arm/joint_states',  'position'),
)

ACTION_CONCAT = (
    ('/left_arm/joint_cmd',      'position'),
    ('/left_gripper/joint_cmd',  'effort'),
    ('/right_arm/joint_cmd',     'position'),
    ('/right_gripper/joint_cmd', 'effort'),
)
