"""
_mcap_utils 本地读取分派的正确性测试:
  1. mcap_url 为绝对路径时 fetch_mcap_bytes / fetch_mcap_length 走本地 pread, 结果与直接读文件一致
  2. 绝对路径分派不触碰 s3_client (monkeypatch 为直接报错的函数, 防止分派条件回归)
  3. 同一文件本地读与 S3 读字节级一致 (真实挂载 + 真实桶各读几段对比)
  4. 瞬态 errno (ossfs2 连接抖动) 时丢弃坏 fd 退避重试; 非瞬态 errno 不重试; 耗尽后抛出
"""
import os

from openpi.training.lingyu_dataloader_v2.utils import _mcap_utils
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

# 本地挂载中真实存在的 mcap 及其在桶内的对应 key (同一份数据)
LOCAL_MCAP = "/mnt/data/shihaoran/mcaps_place_apple_on_plate/4MI0_20260923_100302/rec_20260923_100302_0.mcap"
OSS_MCAP_URL = ("shihaoran/collect-algo-2601016/DF-33/4MI0/"
                "20260721-4MI0_20260923-100302-383890_20260923-101920-781126/rec_20260923_100302_0.mcap")


def test_local_dispatch_matches_direct_read(monkeypatch):
    """绝对路径分派: 结果与直接 open/read 一致, 且全程不触碰 s3_client"""
    # 分派若误走 S3, 此函数被调用即报错
    def _no_s3_allowed():
        raise AssertionError("本地绝对路径不应触碰 s3_client")
    monkeypatch.setattr(_mcap_utils, "s3_client", _no_s3_allowed)

    assert _mcap_utils.fetch_mcap_length(LOCAL_MCAP) == os.path.getsize(LOCAL_MCAP)
    with open(LOCAL_MCAP, "rb") as mcap_file:
        for offset, length in ((0, 16), (8, 4096), (1 << 20, 65536)):
            mcap_file.seek(offset)
            expected_bytes = mcap_file.read(length)
            fetched_bytes = _mcap_utils.fetch_mcap_bytes(LOCAL_MCAP, length, offset)
            assert fetched_bytes == expected_bytes, f"offset={offset} 本地读与直接读不一致"
    # 文件末尾截断: 与 S3 range 语义一致, 返回字节数可小于 length
    file_size = os.path.getsize(LOCAL_MCAP)
    tail_bytes = _mcap_utils.fetch_mcap_bytes(LOCAL_MCAP, 64, file_size - 8)
    assert len(tail_bytes) == 8, f"末尾应只读到 8 字节, 实际 {len(tail_bytes)}"
    logger.info("本地分派与直接读一致, 末尾截断语义正确, size=%d", file_size)


def test_local_matches_oss_same_file():
    """同一文件本地 pread 与 S3 get_object 字节级一致"""
    for offset, length in ((0, 8), (4096, 1024), (500_000_000, 65536)):
        local_bytes = _mcap_utils.fetch_mcap_bytes(LOCAL_MCAP, length, offset)
        oss_bytes = _mcap_utils.fetch_mcap_bytes(OSS_MCAP_URL, length, offset)
        assert local_bytes == oss_bytes, f"offset={offset} length={length} 本地与 S3 读不一致"
    assert _mcap_utils.fetch_mcap_length(LOCAL_MCAP) == _mcap_utils.fetch_mcap_length(OSS_MCAP_URL)
    logger.info("本地与 S3 读同一文件字节级一致")


def test_local_transient_errno_retry(monkeypatch, tmp_path):
    """瞬态 errno 时丢弃坏 fd 并重试, 最终读到正确数据"""
    mcap_path = tmp_path / "fake.mcap"
    mcap_path.write_bytes(b"\x89MCAP0\r\n" + b"payload" * 4)
    real_pread = os.pread
    failed_times = {"count": 0}

    def _flaky_pread(fd, length, offset):
        # 前两次模拟 ossfs2 底层连接被中止 (与线上报错同为 errno 103)
        if failed_times["count"] < 2:
            failed_times["count"] += 1
            raise ConnectionAbortedError(103, "Software caused connection abort")
        return real_pread(fd, length, offset)

    monkeypatch.setattr(os, "pread", _flaky_pread)
    monkeypatch.setattr(_mcap_utils.time, "sleep", lambda _: None)
    _mcap_utils._drop_local_fd(str(mcap_path))  # 清理模块级 fd 缓存的跨测试串扰

    fetched_bytes = _mcap_utils.fetch_mcap_bytes(str(mcap_path), 8, 0)
    assert fetched_bytes == b"\x89MCAP0\r\n"
    assert failed_times["count"] == 2
    # 坏 fd 被踢出后重新 open, 缓存中的是可用 fd
    assert str(mcap_path) in _mcap_utils._local_fds
    logger.info("瞬态 errno 重试 2 次后成功, 坏 fd 已被替换")


def test_local_fatal_errno_no_retry(monkeypatch, tmp_path):
    """非瞬态 errno (文件不存在) 不重试, 直接抛出"""
    def _never_called(fd, length, offset):
        raise AssertionError("ENOENT 发生在 os.open 阶段, 不应触发 pread")
    monkeypatch.setattr(os, "pread", _never_called)

    try:
        _mcap_utils.fetch_mcap_bytes(str(tmp_path / "missing.mcap"), 8, 0)
        raise AssertionError("文件不存在应抛 FileNotFoundError")
    except FileNotFoundError:
        pass
    logger.info("非瞬态 errno 未触发重试, 直接抛出")


def test_local_retry_exhausted(monkeypatch, tmp_path):
    """瞬态错误持续发生时, 耗尽 LOCAL_READ_ATTEMPTS 次后抛出原异常"""
    mcap_path = tmp_path / "fake.mcap"
    mcap_path.write_bytes(b"\x89MCAP0\r\n")

    def _always_reset(fd, length, offset):
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(os, "pread", _always_reset)
    monkeypatch.setattr(_mcap_utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(_mcap_utils, "LOCAL_READ_ATTEMPTS", 3)
    _mcap_utils._drop_local_fd(str(mcap_path))

    try:
        _mcap_utils.fetch_mcap_bytes(str(mcap_path), 8, 0)
        raise AssertionError("耗尽重试后应抛出 ConnectionResetError")
    except ConnectionResetError:
        pass
    logger.info("重试耗尽后抛出原异常")
