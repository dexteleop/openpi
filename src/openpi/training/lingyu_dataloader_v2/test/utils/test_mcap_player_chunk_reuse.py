"""
验证顺序播放复用已下载 chunk(预读缓冲 + fetch_data_bytes 命中刚播完的 chunk)后:
1. 产出与逐条回头请求 S3 的旧行为逐条一致(含 GOP locations, 即关键帧判断结果一致)
2. 按键 message 反序列化结果一致
3. S3 请求数降到约每个 chunk 一次
"""
import sys
import time
from collections import Counter

from openpi.training.lingyu_dataloader_v2.mcap_sample_extractor import MCAP_SIGNAL_TOPIC
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
from openpi.training.lingyu_dataloader_v2.utils import mcap_player
from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import fetch_mcap_bytes, s3_client
from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher

logger = get_logger(__file__)

_MCAP_URL = ("raw/20251103-TEST/2026-07-29/15/20251103-TEST_20260729-151253-930136_"
             "20260729-152547-124514/rec_20260729_151253_0.mcap")
_MAX_MESSAGES = 8000     # 旧行为每帧两次 S3 往返, 很慢, 只比前一段
_BIG_READ_BYTES = 64 * 1024


def _fetch_unbuffered(mcap_path, buffer, length, offset):
    """旧行为: 每次都直接请求 S3, 不用预读缓冲"""
    return fetch_mcap_bytes(mcap_path, length, offset), buffer


def _play(max_messages: int, request_counter: Counter) -> tuple[list, list]:
    """播前 max_messages 条, 返回 (产出元组列表, 按键列表); 请求数计入 request_counter"""
    fetcher = MCAP_Message_Fetcher()
    played, buttons = [], []
    player = mcap_player.MCAP_Player(_MCAP_URL)
    for message in player.play_messages(extra_topics=(MCAP_SIGNAL_TOPIC,)):
        topic_name, msg_type, msg_def, log_time, locations = message
        played.append(message)
        if topic_name == MCAP_SIGNAL_TOPIC:
            buttons.append(tuple(fetcher.fetch_message(msg_type, msg_def, locations)[0].buttons))
        if len(played) >= max_messages:
            break
    logger.info("requests: %s", dict(request_counter))
    return played, buttons


def _count_requests(request_counter: Counter):
    """包一层 get_object, 按读取大小分类计数"""
    client = s3_client()
    original_get = client.get_object

    def counted_get(**kwargs):
        range_start, range_end = map(int, kwargs["Range"].split("=")[1].split("-"))
        is_big = range_end - range_start + 1 > _BIG_READ_BYTES
        request_counter["big" if is_big else "small"] += 1
        return original_get(**kwargs)

    client.get_object = counted_get


def test():
    """新旧行为逐条对比, 并检查请求数"""
    request_counter = Counter()
    _count_requests(request_counter)

    start_time = time.time()
    new_played, new_buttons = _play(_MAX_MESSAGES, request_counter)
    new_sec, new_requests = time.time() - start_time, dict(request_counter)
    logger.info("new: %d messages in %.1fs, requests=%s", len(new_played), new_sec, new_requests)

    # 关掉两处复用, 还原成逐条回头请求 S3
    request_counter.clear()
    remember_original, fetch_original = mcap_player.remember_chunk_records, mcap_player._fetch_buffered
    mcap_player.remember_chunk_records = lambda *args: None
    mcap_player._fetch_buffered = _fetch_unbuffered
    try:
        start_time = time.time()
        old_played, old_buttons = _play(_MAX_MESSAGES, request_counter)
        old_sec, old_requests = time.time() - start_time, dict(request_counter)
    finally:
        mcap_player.remember_chunk_records, mcap_player._fetch_buffered = remember_original, fetch_original
    logger.info("old: %d messages in %.1fs, requests=%s", len(old_played), old_sec, old_requests)

    assert new_played == old_played, "复用 chunk 后产出与旧行为不一致"
    assert new_buttons == old_buttons, "复用 chunk 后按键解析与旧行为不一致"
    gop_max = max(len(message[4]) for message in new_played)
    logger.info("一致: %d messages, %d signals, 最长 GOP locations=%d, 提速 %.1fx",
                len(new_played), len(new_buttons), gop_max, old_sec / new_sec)
    # 每个 chunk 一次大请求, 小请求只剩开头的少量(文件头之后的首个缓冲)
    assert new_requests.get("small", 0) <= 2, f"仍有逐条小请求: {new_requests}"


if __name__ == "__main__":
    test()
    sys.exit(0)
