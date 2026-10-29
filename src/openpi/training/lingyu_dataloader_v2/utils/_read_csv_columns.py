"""
Parse the xxx.csv exported by the data platform, taking the "rosbag folder path" and "owning job" columns
"""
import csv

DIR_COLUMN = "rosbag文件夹路径"
WORK_COLUMN = "所属作业"


def read_rosbag_dirs_from_csv(csv_path: str) -> dict[str, str]:
    """
    Read the rosbag folder path and owning job columns from the csv, strip tabs/spaces from cell edges,
    and return {rosbag directory: owning job}.
    """
    # utf-8-sig strips the file header's BOM automatically
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Use assert to check both columns exist
        for column_name in (DIR_COLUMN, WORK_COLUMN):
            assert column_name in reader.fieldnames, f"csv 中缺少列 {column_name!r}, 实际列: {reader.fieldnames}"

        return {row[DIR_COLUMN].strip(): row[WORK_COLUMN].strip()
                for row in reader if row[DIR_COLUMN].strip()}
