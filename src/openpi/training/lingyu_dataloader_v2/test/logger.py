import logging
import os

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")


def get_logger(caller_file: str) -> logging.Logger:
    """Return a logger that writes to <caller_stem>.log beside the caller file.

    Usage in every test file:
        from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
        logger = get_logger(__file__)
    """
    stem = os.path.splitext(os.path.basename(caller_file))[0]
    named_logger = logging.getLogger(stem)
    named_logger.setLevel(logging.INFO)
    # basicConfig() is a no-op when pytest has already initialized root logger
    # handlers, so handlers are added directly to guarantee file output.
    if not named_logger.handlers:
        # 日志写到调用者所在目录，子目录中的 test 文件不再把 .log 落到上一级
        log_path = os.path.join(os.path.dirname(os.path.abspath(caller_file)), f"{stem}.log")
        _fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        _fh.setFormatter(_fmt)
        _sh = logging.StreamHandler()
        _sh.setFormatter(_fmt)
        named_logger.addHandler(_fh)
        named_logger.addHandler(_sh)
    return named_logger
