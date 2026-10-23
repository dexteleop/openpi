######
# Action Chunk Length
######
ACTION_CHUNK_LENGTH = 100


######
# State Concat and Action Concat Config
######
# The concat order is the dimension order of the model's state/action vectors: (mcap topic, the field taken from that topic)
# Fields must already be registered in mcap_config's MCAP_STATE_and_ACTION_TOPICS_FIELDS
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
