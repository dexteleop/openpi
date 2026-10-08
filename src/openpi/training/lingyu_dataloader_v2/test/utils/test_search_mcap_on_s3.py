from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import (
    BUCKET_NAME,
    find_mcap_url_and_prompt_pairs,
    list_buckets,
    make_client)

from time import time

from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
logger = get_logger(__file__)


def test():
    buckets = list_buckets()
    logger.info(f"找到桶数: {len(buckets)}, 分别为: {buckets}")

    mcap_urls = find_mcap_url_and_prompt_pairs()
    logger.info(f"找到 mcap 文件数: {len(mcap_urls)}")
    for mcap_url, prompt in mcap_urls.items():
        logger.info("%s\t%s", prompt, mcap_url)

    # 取首个 mcap 读 8 字节, 验证确实能读到内容(mcap 魔数为 \x89MCAP0\r\n)
    first_mcap_url = next(iter(mcap_urls))
    start_time = time()
    magic = make_client().get_object(
        Bucket=BUCKET_NAME,
        Key=first_mcap_url,
        Range="bytes=0-7")["Body"].read()
    stop_time = time()

    logger.info("首个 mcap %s 魔数: %r", first_mcap_url, magic)
    logger.info(f"读取花费时间{stop_time-start_time}s")
    assert magic == b"\x89MCAP0\r\n"


if __name__ == "__main__":
    test()
