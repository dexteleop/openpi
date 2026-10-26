"""
The data platform's rosbags live on S3, and the csv's "rosbag folder path" is the object prefix inside the bucket,
so mcap_url is simply the in-bucket key, no longer going through a local mount.

Environment variables required before running:
    export ACCESS_KEY_ID="xxx"
    export SECRET_ACCESS_KEY="xxx"
    export S3_ENDPOINT="https://oss-cn-wulanchabu-internal.aliyuncs.com"
    export REGION_NAME="cn-wulanchabu"
    export BUCKET_NAME="lingyu-wulan-pai-oss"
    export KEY_PREFIX="xxx"  # Optional: prefix between the bucket and the csv paths; empty if unset
"""
import glob
import logging
import os
import posixpath
import boto3
from botocore.config import Config

from openpi.training.lingyu_dataloader_v2.utils._read_csv_columns import read_rosbag_dirs_from_csv

# Project root (openpi/); the data platform's exported csv goes here
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
BUCKET_NAME = os.environ["BUCKET_NAME"]
# In-bucket key = KEY_PREFIX + the rosbag path from the csv; listing results already include the prefix, so downstream get_object need not re-join
KEY_PREFIX = os.environ.get("KEY_PREFIX", "").strip("/")

logger = logging.getLogger(__name__)


def make_client(config: Config | None = None):
    """Build an S3 client from environment variables; missing variables raise KeyError directly from os.environ[]
    config: optional botocore Config (e.g. pool size); None uses botocore defaults
    """
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SECRET_ACCESS_KEY"],
        config=config,
    )


def list_buckets() -> list[str]:
    """Return all bucket names on the endpoint"""
    return [bucket["Name"] for bucket in make_client().list_buckets()["Buckets"]]


def find_mcap_url_and_prompt_pairs() -> dict[str, str]:
    """
    List each rosbag directory from the csv as an in-bucket prefix, check whether .mcap files exist under it,
    and return {mcap url: owning job (English)}.
    """
    csv_files = glob.glob(os.path.join(PROJECT_ROOT, "*.csv"))
    # Check whether this csv_file exists
    assert len(csv_files) == 1, f"项目根目录应有且仅有一个 csv, 实际: {csv_files}"

    rosbag_dirs = read_rosbag_dirs_from_csv(csv_files[0])
    logger.info(f"csv 中 rosbag 目录数: {len(rosbag_dirs)}")

    client = make_client()
    mcap_urls = {}
    for rosbag_dir, prompt in rosbag_dirs.items():
        # The key prefix must end with /, otherwise a sibling directory with the same name prefix would match
        # When KEY_PREFIX is empty, posixpath.join degrades to the original path
        key_prefix = posixpath.join(KEY_PREFIX, rosbag_dir.strip("/")) + "/"
        response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=key_prefix)
        found = sorted(obj["Key"] for obj in response.get("Contents", ())
                       if obj["Key"].endswith(".mcap"))
        if not found:
            logger.warning(f"[无 mcap] {rosbag_dir}")  # Missing data uses WARNING, still visible at higher levels
            continue

        # Put the {mcap_url, prompt} pairs into mcap_urls
        mcap_urls.update(dict.fromkeys(found, prompt))

    return mcap_urls
