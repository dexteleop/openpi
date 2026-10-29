"""
Verify that sequential playback reusing downloaded chunks (prefetch buffer + fetch_data_bytes hitting the just-played chunk):
1. Output matches the old behavior of re-requesting S3 per message, item by item (including GOP locations, i.e. identical keyframe decisions)
2. Deserialized key messages match
3. S3 request count drops to about one per chunk
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
_MAX_MESSAGES = 8000     # The old behavior costs two S3 round trips per frame, very slow; compare only a prefix
_BIG_READ_BYTES = 64 * 1024


def _fetch_unbuffered(mcap_path, buffer, length, offset):
    """Old behavior: request S3 directly every time, no prefetch buffer"""
    return fetch_mcap_bytes(mcap_path, length, offset), buffer


def _play(max_messages: int, request_counter: Counter) -> tuple[list, list]:
    """Play the first max_messages messages, returning (yielded tuple list, key list); requests are counted into request_counter"""
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
    """Wrap get_object, counting reads by size class"""
    client = s3_client()
    original_get = client.get_object

    def counted_get(**kwargs):
        range_start, range_end = map(int, kwargs["Range"].split("=")[1].split("-"))
        is_big = range_end - range_start + 1 > _BIG_READ_BYTES
        request_counter["big" if is_big else "small"] += 1
        return original_get(**kwargs)

    client.get_object = counted_get


def test():
    """Compare old and new behaviors item by item, and check request counts"""
    request_counter = Counter()
    _count_requests(request_counter)

    start_time = time.time()
    new_played, new_buttons = _play(_MAX_MESSAGES, request_counter)
    new_sec, new_requests = time.time() - start_time, dict(request_counter)
    logger.info("new: %d messages in %.1fs, requests=%s", len(new_played), new_sec, new_requests)

    # Disable both reuse points to restore per-message S3 re-requesting
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
    # One large request per chunk; small requests reduce to the few at the start (first buffer after the file header)
    assert new_requests.get("small", 0) <= 2, f"仍有逐条小请求: {new_requests}"


if __name__ == "__main__":
    test()
    sys.exit(0)
