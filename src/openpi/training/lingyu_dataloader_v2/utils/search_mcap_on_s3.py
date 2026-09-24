"""
数据平台的 rosbag 存在 S3 上, csv 中的“rosbag文件夹路径”即桶内对象前缀,
故 mcap_url 就是桶内 key, 不再经过本地挂载。

运行前需设置环境变量:
    export S3_ENDPOINT="http://x.x.x.x:19000"
    export REGION_NAME="us-east-1"
    export ACCESS_KEY_ID="xxx"
    export SECRET_ACCESS_KEY="xxx"
    export BUCKET_NAME="xxx"
"""
import glob
import os
import boto3

from openpi.training.lingyu_dataloader_v2.utils._prompt_translator import translate_prompts
from openpi.training.lingyu_dataloader_v2.utils._read_csv_columns import read_rosbag_dirs_from_csv

# 项目根目录 (openpi/), 数据平台导出的 csv 放在这里
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
BUCKET_NAME = os.environ["BUCKET_NAME"]


def make_client():
    """按环境变量建立 S3 客户端; 缺少变量时 os.environ[] 直接抛 KeyError"""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SECRET_ACCESS_KEY"],
    )


def list_buckets() -> list[str]:
    """返回端点上全部桶名"""
    return [bucket["Name"] for bucket in make_client().list_buckets()["Buckets"]]


def find_mcap_url_and_prompt_pairs() -> dict[str, str]:
    """
    把 csv 中的 rosbag 目录当作桶内前缀逐个列举, 检查前缀下是否有 .mcap,
    返回 {mcap url: 所属作业(英文)}。
    """
    csv_files = glob.glob(os.path.join(PROJECT_ROOT, "*.csv"))
    # 判断是否这个csv_file是否存在
    assert len(csv_files) == 1, f"项目根目录应有且仅有一个 csv, 实际: {csv_files}"

    rosbag_dirs = read_rosbag_dirs_from_csv(csv_files[0])
    print(f"csv 中 rosbag 目录数: {len(rosbag_dirs)}")

    client = make_client()
    mcap_urls = {}
    for rosbag_dir, prompt in rosbag_dirs.items():
        # key 前缀必须以 / 结尾, 否则会匹配到同名前缀的兄弟目录
        response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=rosbag_dir.rstrip("/") + "/")
        found = sorted(obj["Key"] for obj in response.get("Contents", ())
                       if obj["Key"].endswith(".mcap"))
        if not found:
            print(f"[无 mcap] {rosbag_dir}")
            continue

        # 将 {mcap_url, prompt} 放到 mcap_urls 中
        mcap_urls.update(dict.fromkeys(found, prompt))

    # 作业名中英文相间, 统一翻成英文; 先去重再翻, 同一作业名只请求一次
    prompts = list(dict.fromkeys(mcap_urls.values()))
    prompts_en = dict(zip(prompts, translate_prompts(prompts, max_workers=100)))

    return {mcap_url: prompts_en[prompt] for mcap_url, prompt in mcap_urls.items()}
