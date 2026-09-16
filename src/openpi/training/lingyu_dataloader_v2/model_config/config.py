"""统一配置入口: 指定当前使用哪个具身智能模型, 并对外提供读取其配置的函数。

其他程序一律通过 load_action_chunk_length() 拿配置, 不直接 import 某个模型的配置文件,
这样换模型时只需改本文件顶部那一行 import。

每个模型配置文件必须提供同名的常量:
    ACTION_CHUNK_LENGTH  一个 sample 的动作序列长度, 即 [当前动作, 后续 n-1 个动作] 的长度
"""
from __future__ import annotations

from openpi.training.lingyu_dataloader_v2.model_config import pi0  # 换模型只改这一行


MODEL = "pi0"  # TODO：选定使用哪个具身智能模型


#####
# 加载一个 sample 的动作序列长度
#####

def load_action_chunk_length() -> int:
    """返回当前模型一个 sample 的动作序列长度(含当前动作)。"""
    if MODEL == "pi0":
        return pi0.ACTION_CHUNK_LENGTH
    # TODO： elif 其他模型的常量配置
    else:
        raise ValueError(
            f"Cannot find the relevant model configuration of {MODEL!r} "
            f"in lingyu_dataloader_v2/model_config."
        )
