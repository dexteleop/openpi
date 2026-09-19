######
# Action Chunk Length
######
ACTION_CHUNK_LENGTH = 30


######
# State Concat and Action Concat Config
######
# 拼接顺序即模型 state/action 向量的维度顺序: (mcap topic, 该 topic 取的字段)
# 字段须已登记在 mcap_config 的 MCAP_STATE_and_ACTION_TOPICS_FIELDS 中
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
