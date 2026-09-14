"""
数据平台现在是将文件夹直接挂载到了虚拟机上
利用数据平台上生成的xxx.csv中的“rosbag文件夹路径”列来找到所有mcap的路径
"""
import csv
import glob
import os

# 项目根目录 (openpi/) 与 minio 挂载前缀
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
MINIO_PREFIX = "/mnt/minio/mybucket"


def _read_rosbag_dirs_from_csv(csv_path: str, column_name: str = "rosbag文件夹路径") -> list[str]:
    """
    读取 csv 中 rosbag文件夹路径 一列, 去掉单元格首尾的制表符/空格。
    """
    # utf-8-sig 自动去掉文件头的 BOM
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # 使用 assert 判断是否有这一列
        assert column_name in reader.fieldnames, f"csv 中缺少列 {column_name!r}, 实际列: {reader.fieldnames}"
        return [row[column_name].strip() for row in reader if row[column_name].strip()]


def find_mcap_paths() -> list[str]:
    """
    给路径添加 minio 前缀, 检查目录是否存在及目录下是否有 .mcap,
    返回所有找到的 .mcap 路径。
    """
    csv_files = glob.glob(os.path.join(PROJECT_ROOT, "*.csv"))
    # 判断是否这个csv_file是否存在
    assert len(csv_files) == 1, f"项目根目录应有且仅有一个 csv, 实际: {csv_files}"

    rosbag_dirs = _read_rosbag_dirs_from_csv(csv_files[0])
    print(f"csv 中 rosbag 目录数: {len(rosbag_dirs)}")

    mcap_paths = []
    for rosbag_dir in rosbag_dirs:
        full_dir = os.path.join(MINIO_PREFIX, rosbag_dir)
        if not os.path.isdir(full_dir):
            print(f"[缺失目录] {full_dir}")
            continue
        found = sorted(glob.glob(os.path.join(full_dir, "*.mcap")))
        if not found:
            print(f"[无 mcap] {full_dir}")
            continue
        mcap_paths.extend(found)
    return mcap_paths

