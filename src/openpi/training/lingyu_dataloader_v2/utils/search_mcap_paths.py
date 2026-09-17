"""
数据平台现在是将文件夹直接挂载到了虚拟机上
利用数据平台上生成的xxx.csv中的“rosbag文件夹路径”列来找到所有mcap的路径
"""
import csv
import glob
import os

from openpi.training.lingyu_dataloader_v2.utils._prompt_translator import translate_prompts

# 项目根目录 (openpi/) 与 minio 挂载前缀
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
MINIO_PREFIX = "/mnt/minio/mybucket"


DIR_COLUMN = "rosbag文件夹路径"
WORK_COLUMN = "所属作业"


def _read_rosbag_dirs_from_csv(csv_path: str) -> dict[str, str]:
    """
    读取 csv 中 rosbag文件夹路径、所属作业 两列, 去掉单元格首尾的制表符/空格,
    返回 {rosbag 目录: 所属作业}。
    """
    # utf-8-sig 自动去掉文件头的 BOM
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # 使用 assert 判断是否有这两列
        for column_name in (DIR_COLUMN, WORK_COLUMN):
            assert column_name in reader.fieldnames, f"csv 中缺少列 {column_name!r}, 实际列: {reader.fieldnames}"

        return {row[DIR_COLUMN].strip(): row[WORK_COLUMN].strip()
                for row in reader if row[DIR_COLUMN].strip()}


def find_mcap_paths() -> dict[str, str]:
    """
    给路径添加 minio 前缀, 检查目录是否存在及目录下是否有 .mcap,
    返回 {mcap 路径: 所属作业(英文)}。
    """
    csv_files = glob.glob(os.path.join(PROJECT_ROOT, "*.csv"))
    # 判断是否这个csv_file是否存在
    assert len(csv_files) == 1, f"项目根目录应有且仅有一个 csv, 实际: {csv_files}"

    rosbag_dirs = _read_rosbag_dirs_from_csv(csv_files[0])
    print(f"csv 中 rosbag 目录数: {len(rosbag_dirs)}")

    mcap_paths = {}
    for rosbag_dir, work_name in rosbag_dirs.items():
        # 查找mcap目录，保证路径存在
        full_dir = os.path.join(MINIO_PREFIX, rosbag_dir)
        if not os.path.isdir(full_dir):
            print(f"[缺失目录] {full_dir}")
            continue
        found = sorted(glob.glob(os.path.join(full_dir, "*.mcap")))
        if not found:
            print(f"[无 mcap] {full_dir}")
            continue

        # 将 {mcap_path, work_name} 放到 mcap_paths 中
        mcap_paths.update(dict.fromkeys(found, work_name))

    # 作业名中英文相间, 统一翻成英文; 先去重再翻, 同一作业名只请求一次
    work_names = list(dict.fromkeys(mcap_paths.values()))
    work_names_en = dict(zip(work_names, translate_prompts(work_names, max_workers=100)))

    return {path: work_names_en[work_name] for path, work_name in mcap_paths.items()}
