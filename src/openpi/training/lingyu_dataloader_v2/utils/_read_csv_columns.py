"""
解析数据平台导出的 xxx.csv, 取出“rosbag文件夹路径”与“所属作业”两列
"""
import csv

DIR_COLUMN = "rosbag文件夹路径"
WORK_COLUMN = "所属作业"


def read_rosbag_dirs_from_csv(csv_path: str) -> dict[str, str]:
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
