"""用 fetch_SummaryRecords 定位 Summary Section, 再分别解析其中的
Schema / Channel / ChunkIndex Record, 并用 ChunkIndex 读出一个 chunk 内的
全部 MessageIdx, 打印解析结果。
"""
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
    find_mcap_url_and_prompt_pairs
)
from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    OP_SCHEMA, OP_CHANNEL, OP_CHUNK_IDX,
    fetch_SummaryRecords,
    parse_SchemaRecord_in_SummarySection,
    parse_ChannelRecord_in_SummarySection,
    parse_ChunkIndexRecord_in_SummarySection,
    parse_MessageIdx_from_ChunkIndexRecord,
)

from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
logger = get_logger(__file__)


def test():
    mcap_url = next(iter(find_mcap_url_and_prompt_pairs()))
    logger.info(f"mcap_url: {mcap_url}")

    # Summary Section 中各类 Record 的位置: opcode -> (start, length)
    summary_records_loca = fetch_SummaryRecords(mcap_url)
    logger.info(f"{ {f'0x{k.hex()}': v for k, v in summary_records_loca.items()} }")

    schemas = parse_SchemaRecord_in_SummarySection(
        mcap_url, *summary_records_loca[bytes([OP_SCHEMA])])
    logger.info(f"===== Schema x{len(schemas)} =====")
    for schema in schemas:
        logger.info(f"{schema}")

    channels = parse_ChannelRecord_in_SummarySection(
        mcap_url, *summary_records_loca[bytes([OP_CHANNEL])])
    logger.info(f"===== Channel x{len(channels)} =====")
    for channel in channels:
        logger.info(f"{channel}")

    chunks = parse_ChunkIndexRecord_in_SummarySection(
        mcap_url, *summary_records_loca[bytes([OP_CHUNK_IDX])])
    logger.info(f"===== ChunkIndex x{len(chunks)} =====")
    for chunk in chunks[:3]:
        # 只打印前三条
        logger.info(f"{chunk}")

    # 第一个 chunk 内的全部 MessageIdx: channel_id -> [(log_time, offset), ...]
    msg_idxes = parse_MessageIdx_from_ChunkIndexRecord(mcap_url, chunks[0])
    logger.info(f"===== ChunkIndex 0 MessageIdx x{len(msg_idxes)} channels, "
                f"{sum(map(len, msg_idxes.values()))} messages =====")
    for channel_id, msg_idx_ls in msg_idxes.items():
        logger.info(f"channel_id={channel_id} x{len(msg_idx_ls)}: {msg_idx_ls[:7]}")


if __name__ == "__main__":
    test()
