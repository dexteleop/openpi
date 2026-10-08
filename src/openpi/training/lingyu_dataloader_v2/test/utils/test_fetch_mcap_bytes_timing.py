"""验证 fetch_mcap_bytes 的计时日志与 pool_stats 连接池监控:
串行读取只建一条连接并持续复用; 并发读取的新建连接数不超过并发线程数,
请求结束后连接全部回到池中空闲。计时日志写入同一个 .log 文件。
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

# 并发读取的线程数与请求数
CONCURRENT_WORKERS = 32
CONCURRENT_REQUESTS = 128


def _route_mcap_utils_log() -> None:
    """把 _mcap_utils 的 DEBUG 计时日志导入本测试的 .log 文件"""
    mcap_utils_logger = logging.getLogger(_mcap_utils.__name__)
    mcap_utils_logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if handler not in mcap_utils_logger.handlers:
            mcap_utils_logger.addHandler(handler)


def test():
    _route_mcap_utils_log()
    mcap_url = next(iter(find_mcap_url_and_prompt_pairs()))
    logger.info(f"mcap_url: {mcap_url}")

    # --- 串行: 读 magic + Summary, 只应新建 1 条连接, 其余请求全部复用 ---
    magic = fetch_mcap_bytes(mcap_url, MAGIC_SIZE, 0)
    assert magic == b"\x89MCAP0\r\n", f"magic 不符: {magic!r}"
    fetch_SummaryRecords(mcap_url)
    serial_stats = pool_stats()
    logger.info(f"串行读取后: {serial_stats}")
    assert len(serial_stats) == 1
    assert serial_stats[0]["maxsize"] == S3_MAX_POOL_CONNECTIONS
    assert serial_stats[0]["created"] == 1, "串行请求应只新建一条连接"
    assert serial_stats[0]["idle"] == 1 and serial_stats[0]["in_use"] == 0

    # --- 并发: 新建连接数不超过线程数, 结束后连接全部空闲 ---
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
