"""
search_mcap_on_local 的发现逻辑测试:
  1. resolve_local_dir 对绝对路径 / 挂载根下相对目录名 / 不存在目录三种输入的解析
  2. find_mcap_url_and_prompt_pairs 在临时 csv + 临时 mcap 目录上端到端返回 {绝对路径: prompt}
"""
from openpi.training.lingyu_dataloader_v2.utils import search_mcap_on_local
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)


def test_resolve_local_dir(tmp_path, monkeypatch):
    """csv 目录列三种形式: 绝对路径直接用, 相对目录名拼挂载根, 都不存在返回 None"""
    # 临时挂载根下造一个 rosbag 目录
    rosbag_dir = tmp_path / "4MI0_20260923_100302"
    rosbag_dir.mkdir()
    monkeypatch.setattr(search_mcap_on_local, "MCAP_LOCAL_ROOT", str(tmp_path))

    # 绝对路径: 目录存在直接返回
    assert search_mcap_on_local.resolve_local_dir(str(rosbag_dir)) == str(rosbag_dir)
    # 相对目录名: 拼接挂载根后存在, 返回拼接结果
    assert search_mcap_on_local.resolve_local_dir("4MI0_20260923_100302") == str(rosbag_dir)
    # 两种形式都解析不到: 返回 None
    assert search_mcap_on_local.resolve_local_dir("no_such_dir") is None
    logger.info("resolve_local_dir 三种形式解析通过, 挂载根=%s", tmp_path)


def test_find_mcap_url_and_prompt_pairs(tmp_path, monkeypatch):
    """临时 csv + 临时 mcap 目录: 返回 {mcap 绝对路径: prompt}, 缺失目录/无 mcap 的行被跳过"""
    # csv 根目录: 只放一个 csv; 每行一个 rosbag 目录
    csv_root = tmp_path / "csv_root"
    csv_root.mkdir()
    mcap_a_dir = tmp_path / "D6BQ_20261013_132847"
    mcap_a_dir.mkdir()
    (mcap_a_dir / "rec_20261013_132847_0.mcap").write_bytes(b"\x89MCAP0\r\n")
    empty_dir = tmp_path / "no_mcap_here"
    empty_dir.mkdir()
    (csv_root / "data.csv").write_text(
        "rosbag文件夹路径,所属作业\n"
        f"{mcap_a_dir},Put an apple into the bowl.\n"
        f"{empty_dir},Put an apple into the bowl.\n"
        "/tmp/never_exists_dir,Put an apple into the bowl.\n",
        encoding="utf-8-sig")
    monkeypatch.setattr(search_mcap_on_local, "PROJECT_ROOT", str(csv_root))

    mcap_urls = search_mcap_on_local.find_mcap_url_and_prompt_pairs()
    expected = {f"{mcap_a_dir}/rec_20261013_132847_0.mcap": "Put an apple into the bowl."}
    assert mcap_urls == expected, f"期望 {expected}, 实际 {mcap_urls}"
    logger.info("端到端发现通过: %s", mcap_urls)
