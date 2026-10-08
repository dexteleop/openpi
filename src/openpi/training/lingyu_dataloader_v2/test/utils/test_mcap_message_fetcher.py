"""
Verify the (msg_type, msg_def) pair produced by MCAP_Player is enough to
resolve a full ROS2 type structure through MCAP_Message_Fetcher's typestore.

Test plan:
  Walk every message of mcap_paths[0] with _play_messages (the unfiltered
  scanner — MCAP_Player.play_messages would only yield the 9 topics selected by
  topics_filter), collect every unique msg_type, register it from its msg_def,
  then log the expanded field tree plus the field values of one really
  deserialized message per type.
"""
from rosbags.interfaces import Nodetype

from openpi.training.lingyu_dataloader_v2.utils.mcap_player import _play_messages
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import find_mcap_url_and_prompt_pairs
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_SAMPLE_COUNT = 10000
_STRUCTURE_MAX_DEPTH = 3  # nested msg_type levels expanded when printing
_VALUE_PREVIEW_COUNT = 6  # array / sequence elements kept in the log preview


def _describe_type(typestore, msg_type: str, depth: int = 0) -> list[str]:
    """递归展开 msg_type 的字段树, 返回缩进文本行列表。深度超过上限时折叠。"""
    indent = " " * (depth * 2)
    if depth > _STRUCTURE_MAX_DEPTH:
        return [indent + "..."]
    _, fields = typestore.fielddefs[msg_type]
    lines = []
    for fname, ftype in fields:
        kind = ftype[0]
        if kind == Nodetype.BASE:                     # 基础类型: ('float64', 0)
            lines.append(f"{indent}{fname}: {ftype[1][0]}")
        elif kind == Nodetype.NAME:                   # 嵌套 message, 递归展开
            lines.append(f"{indent}{fname}: {ftype[1]}")
            lines.extend(_describe_type(typestore, ftype[1], depth + 1))
        else:                                         # SEQUENCE / ARRAY, 取内层类型
            inner_kind, inner_args = ftype[1][0]
            container = "sequence" if kind == Nodetype.SEQUENCE else "array"
            if inner_kind == Nodetype.BASE:
                lines.append(f"{indent}{fname}: {container}<{inner_args[0]}>")
            else:
                lines.append(f"{indent}{fname}: {container}<{inner_args}>")
                lines.extend(_describe_type(typestore, inner_args, depth + 1))
    return lines


def _sample_value(msg) -> list[str]:
    """把一条 ROS2 message 对象的字段值转成可读文本, 长数组只留前几项。"""
    lines = []
    for fname, val in vars(msg).items():
        if fname.startswith("_"):
            continue
        if hasattr(val, "__len__") and len(val) > _VALUE_PREVIEW_COUNT:
            preview = ", ".join(str(v) for v in list(val)[:_VALUE_PREVIEW_COUNT])
            lines.append(f"  {fname}: [{preview}, ...] (len={len(val)})")
        else:
            lines.append(f"  {fname}: {val}")
    return lines


def test():
    """每个 msg_type 都能由 msg_def 解析出类型结构, 并反序列化出实际字段值。"""
    path = next(iter(find_mcap_url_and_prompt_pairs()))
    logger.info("mcap file: %s", path)

    fetcher = MCAP_Message_Fetcher()
    seen_types: dict[str, tuple[str, tuple]] = {}  # msg_type -> (msg_def, 首次出现的 location)
    seen_topics: dict[str, str] = {}               # topic_name -> msg_type
    played_count = 0

    for i, msg in enumerate(_play_messages(path)):
        location = (path,
                    msg["chunk_file_offset"], msg["uncompressed_byte_offset"], msg["record_length"])
        seen_types.setdefault(msg["msg_type"], (msg["msg_def"], location))
        seen_topics.setdefault(msg["topic_name"], msg["msg_type"])
        played_count += 1
        if played_count >= _SAMPLE_COUNT:
            break
    logger.info("scanned %d messages, found %d topics / %d unique msg_types",
                played_count, len(seen_topics), len(seen_types))
    logger.info("topic -> msg_type:\n%s",
                "\n".join(f"  {t}: {mt}" for t, mt in sorted(seen_topics.items())))
    assert seen_types, "mcap file must contain at least one msg_type"

    for msg_type, (msg_def, location) in sorted(seen_types.items()):
        # 读取ros2原始的结构化消息
        sample_msg = fetcher.fetch_message(msg_type, msg_def, [location])[0]

        # 确保所有的 ros2 msg 都可以被解析
        assert msg_type in fetcher._typestore.fielddefs, (
            f"{msg_type} must be resolvable from its msg_def"
        )
        logger.info("\n--- type structure: %s ---\n%s",
                    msg_type, "\n".join(_describe_type(fetcher._typestore, msg_type)))

        # 展示 ros2 msg 反序列化后的情况
        assert type(sample_msg).__msgtype__ == msg_type, "反序列化结果类型必须与 msg_type 一致"
        logger.info("--- sample values: %s ---\n%s",
                    msg_type, "\n".join(_sample_value(sample_msg)))


if __name__ == "__main__":
    test()
