"""
数据平台的 rosbag 存在 S3 上, csv 中的“rosbag文件夹路径”即桶内对象前缀,
故 mcap_url 就是桶内 key, 不再经过本地挂载。

运行前需设置环境变量(现指向本地 docker minio):
    export S3_ENDPOINT="http://127.0.0.1:9000"
    export REGION_NAME="us-east-1"
    export ACCESS_KEY_ID="xxx"
    export SECRET_ACCESS_KEY="xxx"
    export BUCKET_NAME="xxx"
    export KEY_PREFIX="shihaoran"  # 可选: 桶与 csv 路径之间的前缀, 不设则为空
"""
import glob
import logging
import os
import posixpath
import boto3
from botocore.config import Config
from pyarrow.fs import AwsStandardS3RetryStrategy, S3FileSystem

from openpi.training.lingyu_dataloader_v2.utils._prompt_translator import translate_prompts
from openpi.training.lingyu_dataloader_v2.utils._read_csv_columns import read_rosbag_dirs_from_csv

# 项目根目录 (openpi/), 数据平台导出的 csv 放在这里
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
BUCKET_NAME = os.environ["BUCKET_NAME"]
# 桶内 key = KEY_PREFIX/csv 中的 rosbag 路径; 列举结果已含前缀, 下游 get_object 无需再拼
KEY_PREFIX = os.environ.get("KEY_PREFIX", "").strip("/")
# pyarrow(含 pyiceberg 内部)每建一个 S3FileSystem, AWS SDK 都会探测 EC2 元数据服务, 非 EC2 机器上
# 首次卡 6 s、之后每次 2 s; 端点与密钥都已显式给出, 用不到它, 故在所有 S3 连接的公共入口统一关掉
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
# pyarrow S3 的超时与重试: AWS SDK 默认连接 1 s / 请求 3 s / 3 次, 256 个 DataLoader worker 同时启动、CPU 打满时
# 实测 HeadObject 超时(curlCode 28)直接让训练退出; 取值与 _mcap_utils 读 mcap 的 boto3 客户端一致(读超时 20 s, 10 次)
ARROW_S3_CONNECT_TIMEOUT = 10
ARROW_S3_REQUEST_TIMEOUT = 20
ARROW_S3_MAX_ATTEMPTS = 10

logger = logging.getLogger(__name__)


def make_client(config: Config | None = None):
    """按环境变量建立 S3 客户端; 缺少变量时 os.environ[] 直接抛 KeyError
    config: 可选 botocore Config(如连接池大小), None 时使用 botocore 默认值
    """
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SECRET_ACCESS_KEY"],
        config=config,
    )


def make_arrow_filesystem() -> S3FileSystem:
    """按同一组环境变量建立 pyarrow 的 S3 文件系统, 供按 row group 直接读桶内的 episode parquet
    路径写成 "桶名/key", 不带 s3:// 前缀
    """
    return S3FileSystem(
        endpoint_override=os.environ["S3_ENDPOINT"],
        region=os.environ["REGION_NAME"],
        access_key=os.environ["ACCESS_KEY_ID"],
        secret_key=os.environ["SECRET_ACCESS_KEY"],
        connect_timeout=ARROW_S3_CONNECT_TIMEOUT,
        request_timeout=ARROW_S3_REQUEST_TIMEOUT,
        # 标准重试: 超时、连接错误与 5xx/限流都会指数退避后重试
        retry_strategy=AwsStandardS3RetryStrategy(max_attempts=ARROW_S3_MAX_ATTEMPTS),
    )


def iceberg_s3_properties() -> dict[str, str]:
    """同一组环境变量换成 pyiceberg FileIO 的属性名, 使 Iceberg 能登记并读取 s3:// 数据文件"""
    return {
        "s3.endpoint": os.environ["S3_ENDPOINT"],
        "s3.region": os.environ["REGION_NAME"],
        "s3.access-key-id": os.environ["ACCESS_KEY_ID"],
        "s3.secret-access-key": os.environ["SECRET_ACCESS_KEY"],
    }


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
    logger.info(f"csv 中 rosbag 目录数: {len(rosbag_dirs)}")

    client = make_client()
    mcap_urls = {}
    for rosbag_dir, prompt in rosbag_dirs.items():
        # key 前缀必须以 / 结尾, 否则会匹配到同名前缀的兄弟目录
        # KEY_PREFIX 为空时 posixpath.join 退化为原路径
        key_prefix = posixpath.join(KEY_PREFIX, rosbag_dir.strip("/")) + "/"
        response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=key_prefix)
        found = sorted(obj["Key"] for obj in response.get("Contents", ())
                       if obj["Key"].endswith(".mcap"))
        if not found:
            logger.warning(f"[无 mcap] {rosbag_dir}")  # 缺数据用 WARNING, 调高级别时仍可见
            continue

        # 将 {mcap_url, prompt} 放到 mcap_urls 中
        mcap_urls.update(dict.fromkeys(found, prompt))

    # 作业名中英文相间, 统一翻成英文; 先去重再翻, 同一作业名只请求一次
    prompts = list(dict.fromkeys(mcap_urls.values()))
    prompts_en = dict(zip(prompts, translate_prompts(prompts, max_workers=100)))

    return {mcap_url: prompts_en[prompt] for mcap_url, prompt in mcap_urls.items()}
