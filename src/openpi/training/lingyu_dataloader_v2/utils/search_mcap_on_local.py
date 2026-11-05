"""
本地挂载模式: csv 的 "rosbag文件夹路径" 列直接是本地目录(绝对路径, 或挂载根下的目录名),
每个目录下恰好一个 .mcap 文件, mcap_url 即本地绝对路径。

与 search_mcap_on_s3.find_mcap_url_and_prompt_pairs 同签名同返回结构:
返回的 {mcap 绝对路径: prompt} 被 build_sample_idx 写进索引 locations,
训练读取侧(_mcap_utils.fetch_mcap_bytes)按 url 以 "/" 开头自动分派为本地 pread, 无需任何配置。

环境变量(可选):
    export MCAP_LOCAL_ROOT="/mnt/data/shihaoran/mcaps_place_apple_on_plate"
    # 仅当 csv 写相对目录名时用于拼接; csv 写绝对路径则不需要
"""
import glob
import logging
import os

from openpi.training.lingyu_dataloader_v2.utils._read_csv_columns import read_rosbag_dirs_from_csv
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import PROJECT_ROOT

# 挂载根: csv 写相对目录名(如 4MI0_20260923_100302)时目录位于此根之下
MCAP_LOCAL_ROOT = os.environ.get("MCAP_LOCAL_ROOT", "").strip("/")

logger = logging.getLogger(__name__)


def resolve_local_dir(rosbag_dir: str) -> str | None:
    """csv 的一行 rosbag 目录 -> 本地目录绝对路径; 两种形式都不存在返回 None"""
    # csv 直接写绝对路径: 目录存在即用
    if os.path.isdir(rosbag_dir):
        return rosbag_dir
    # csv 写挂载根下的目录名: 拼接后存在才用
    if MCAP_LOCAL_ROOT:
        dir_under_root = os.path.join(MCAP_LOCAL_ROOT, rosbag_dir.strip("/"))
        if os.path.isdir(dir_under_root):
            return dir_under_root
    return None


def find_mcap_url_and_prompt_pairs() -> dict[str, str]:
    """
    读 csv 中每个本地 rosbag 目录下的 .mcap, 返回 {mcap 绝对路径: prompt}。
    """
    csv_files = glob.glob(os.path.join(PROJECT_ROOT, "*.csv"))
    # Check whether this csv_file exists
    assert len(csv_files) == 1, f"项目根目录应有且仅有一个 csv, 实际: {csv_files}"

    rosbag_dirs = read_rosbag_dirs_from_csv(csv_files[0])
    logger.info(f"csv 中 rosbag 目录数: {len(rosbag_dirs)}")

    mcap_urls = {}
    for rosbag_dir, prompt in rosbag_dirs.items():
        local_dir = resolve_local_dir(rosbag_dir)
        # 目录缺失与无 mcap 都用 WARNING: 数据缺失应在更高日志级别可见, 但不中断其余目录
        if local_dir is None:
            logger.warning(f"[目录不存在] {rosbag_dir}")
            continue
        found = sorted(glob.glob(os.path.join(local_dir, "*.mcap")))
        if not found:
            logger.warning(f"[无 mcap] {local_dir}")
            continue
        mcap_urls.update(dict.fromkeys(found, prompt))

    return mcap_urls
