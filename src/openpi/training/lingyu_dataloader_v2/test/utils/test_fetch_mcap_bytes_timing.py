"""Verify fetch_mcap_bytes's timing log and pool_stats connection-pool monitoring:
serial reads build one connection and keep reusing it; concurrent reads build no more connections than threads,
and after the requests all connections return to the pool idle. Timing logs go into this test's own .log file.
"""
import logging
from concurrent.futures import ThreadPoolExecutor

from openpi.training.lingyu_dataloader_v2.utils import _mcap_utils
from openpi.training.lingyu_dataloader_v2.utils._mcap_utils import (
    MAGIC_SIZE, S3_MAX_POOL_CONNECTIONS,
    fetch_mcap_bytes, fetch_SummaryRecords, pool_stats,
)
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
    find_mcap_url_and_prompt_pairs
)

from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
logger = get_logger(__file__)

# Thread count and request count for concurrent reads
CONCURRENT_WORKERS = 32
CONCURRENT_REQUESTS = 128


def _route_mcap_utils_log() -> None:
    """Route _mcap_utils's DEBUG timing log into this test's .log file"""
    mcap_utils_logger = logging.getLogger(_mcap_utils.__name__)
    mcap_utils_logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if handler not in mcap_utils_logger.handlers:
            mcap_utils_logger.addHandler(handler)


def test():
    _route_mcap_utils_log()
    mcap_url = next(iter(find_mcap_url_and_prompt_pairs()))
    logger.info(f"mcap_url: {mcap_url}")

    # --- Serial: read magic + Summary; only 1 new connection expected, every other request reuses it ---
    magic = fetch_mcap_bytes(mcap_url, MAGIC_SIZE, 0)
    assert magic == b"\x89MCAP0\r\n", f"magic 不符: {magic!r}"
    fetch_SummaryRecords(mcap_url)
    serial_stats = pool_stats()
    logger.info(f"串行读取后: {serial_stats}")
    assert len(serial_stats) == 1
    assert serial_stats[0]["maxsize"] == S3_MAX_POOL_CONNECTIONS
    assert serial_stats[0]["created"] == 1, "串行请求应只新建一条连接"
    assert serial_stats[0]["idle"] == 1 and serial_stats[0]["in_use"] == 0

    # --- Concurrent: new connections <= thread count, and all connections idle afterwards ---
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as pool:
        results = list(pool.map(lambda req_idx: fetch_mcap_bytes(mcap_url, 4096, req_idx * 4096),
                                range(CONCURRENT_REQUESTS)))
    assert all(len(data) == 4096 for data in results)
    concurrent_stats = pool_stats()
    logger.info(f"并发读取后: {concurrent_stats}")
    assert concurrent_stats[0]["created"] <= CONCURRENT_WORKERS + 1
    assert concurrent_stats[0]["idle"] == concurrent_stats[0]["created"]
    assert concurrent_stats[0]["in_use"] == 0
    assert concurrent_stats[0]["requests"] == serial_stats[0]["requests"] + CONCURRENT_REQUESTS


if __name__ == "__main__":
    test()
