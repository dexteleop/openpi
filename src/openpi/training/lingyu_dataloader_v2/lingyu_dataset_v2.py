# 将一个sample中对应的所有的topic的message都找到，

# 根据 model_config 中的
# State Concat and Action Concat Config
# 完成模型需要的拼接结构，从而生成模型架构需要的 state 和 action

# 然后组建成一个dict({obs:..., state:..., action:...})
# dataset的格式以及输出接口参考
# /home/ubuntu/openpi/src/openpi/training/lingyu_dataloader/webdataset_load_tar.py
