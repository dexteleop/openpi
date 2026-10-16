"""全量串行计算 norm stats, 记录耗时与得数。

只计算不写 norm_stats.json, 不影响 assets 里已有的结果。

运行(须在仓库根目录)::
    python -m pytest src/openpi/training/lingyu_dataloader_v2/test/test_compute_norm_stats_lingyu.py -s
"""
from pathlib import Path
import sys
import time

import numpy as np

import openpi.training.config_lingyu as _config
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

# scripts/ 不在 src 下, 把仓库根目录加进 sys.path; forkserver 起的 worker 会继承它, 也能 import 到脚本里的类
sys.path.insert(0, str(Path(__file__).parents[5]))
from scripts import compute_norm_stats_lingyu as norm_script

logger = get_logger(__file__)

CONFIG_NAME = "pi0_teleavatar_v2_lingyu"
STATS_FIELDS = ("mean", "std", "q01", "q99")


def test():
    """全部 mcap: 串行方式计算 norm stats, 记录耗时与各 key 的得数。"""
    config = _config.get_config(CONFIG_NAME)
    data_config = norm_script.create_norm_data_config(config)

    start_time = time.perf_counter()
    norm_stats = norm_script.compute_serial_norm_stats(data_config, config, max_frames=None)
    logger.info(f"serial 耗时: {time.perf_counter() - start_time:.1f}s")

    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    for key, stats in norm_stats.items():
        for field in STATS_FIELDS:
            logger.info(f"{key}.{field} = {np.asarray(getattr(stats, field))}")
