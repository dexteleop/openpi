"""拆解 spawn 方式下一个 DataLoader worker 的启动耗时, 并验证 worker 为什么是一个接一个起来的。

复现 torch DataLoader 起 worker 的方式: 在 for 循环里对每个 worker 调用
spawn 上下文的 Process(target, args=(dataset, ...)).start()。子进程不取样, 只回报各阶段时间戳,
不会向 OSS 发任何请求, 也不占 GPU。

核对::
    1. 主模块(与 compute_norm_stats_lingyu.py 相同的 import)里各包的 import 耗时
    2. dataset 的 pickle 体积, 以及 pickle / unpickle 耗时
    3. 子进程各阶段: 解释器启动 -> 重新执行主模块的 import -> unpickle dataset -> 进入 target
    4. 父进程 Process.start() 的阻塞时长: 参数很小时 vs 参数为真实 dataset 时
    5. 改用 forkserver 后的 TorchDataLoader: worker 启动耗时, 且取出的 batch 与主进程直接取样完全一致
       (只取 _SAMPLE_IDXS 这一个 batch, 约 2 x 360 次 S3 请求)

必须以脚本方式运行, 子进程才会像 compute_norm_stats_lingyu.py 那样把本文件当 __mp_main__ 重新执行::
    python src/openpi/training/lingyu_dataloader_v2/test/test_spawn_worker_startup.py
"""
import time

# 子进程重新执行本文件时, 这两个时间戳界定了"重新 import 主模块"这一段
MODULE_IMPORT_START = time.time()

import collections
import dataclasses
import logging
import multiprocessing
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 不碰正在跑的任务所用的 GPU

_SCRIPTS_DIR = Path(__file__).resolve().parents[5] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import compute_norm_stats_lingyu as norm_script  # noqa: E402  与被测脚本完全相同的一组顶层 import

import jax  # noqa: E402
import numpy as np  # noqa: E402

from openpi.training.lingyu_dataloader_v2.test.logger import get_logger  # noqa: E402

MODULE_IMPORT_END = time.time()

# 子进程里本文件名为 __mp_main__, 不能再建 FileHandler(mode="w"), 否则会清空父进程的日志
logger = get_logger(__file__) if __name__ == "__main__" else logging.getLogger(__name__)

_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"
_NUM_WORKERS = 4      # 每种参数起几个 worker, 足以看出串行规律
_TOP_PACKAGES = 15    # import 耗时榜单展示的包数
_LOADER_WORKERS = 8   # TorchDataLoader 验证用的 worker 数, spawn 下约需 8 x 5.3s
_SAMPLE_IDXS = [0, 1]  # TorchDataLoader 只取这一个 batch, 用作 sampler


def build_norm_dataset(config_name: str):
    """与 compute_norm_stats_lingyu.create_torch_dataloader 相同地构造 dataset(不建 DataLoader)。"""
    config = norm_script._config.get_config(config_name)
    data_config = dataclasses.replace(config.data, compute_norm_stats=True).create(config.assets_dirs, config.model)
    dataset = norm_script.LingyuDatasetV2(data_config.iceberg_dir, load_images=False)
    return norm_script._data_loader.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, norm_script.RemoveStrings()],
    )


def measure_import_breakdown() -> float:
    """在新解释器里 -X importtime 导入被测脚本, 按顶层包汇总 self 耗时并记日志, 返回墙钟耗时。"""
    import_cmd = [sys.executable, "-X", "importtime", "-c",
                  f"import sys; sys.path.insert(0, {str(_SCRIPTS_DIR)!r}); import compute_norm_stats_lingyu"]
    start_time = time.perf_counter()
    completed = subprocess.run(import_cmd, capture_output=True, text=True, env=os.environ.copy())
    wall_seconds = time.perf_counter() - start_time
    assert completed.returncode == 0, f"导入被测脚本失败: {completed.stderr[-2000:]}"

    self_by_package = collections.Counter()
    for line in completed.stderr.splitlines():
        matched = re.match(r"import time:\s+(\d+) \|\s+\d+ \|\s*(\S+)", line)
        if matched:
            self_by_package[matched[2].split(".")[0]] += int(matched[1])
    self_total = sum(self_by_package.values()) / 1e6
    logger.info(f"[import] 新解释器导入被测脚本: 墙钟 {wall_seconds:.2f}s, 各模块 self 耗时合计 {self_total:.2f}s, "
                f"共 {len(self_by_package)} 个顶层包")
    for package, self_us in self_by_package.most_common(_TOP_PACKAGES):
        logger.info(f"[import]   {package:24s} {self_us / 1e6:6.3f}s")
    return wall_seconds


def read_process_start_time() -> float:
    """从 /proc 读本进程被内核创建的绝对时间(10ms 精度), 用于计算解释器启动段。"""
    boot_time = next(int(line.split()[1]) for line in open("/proc/stat") if line.startswith("btime"))
    start_ticks = int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19])
    return boot_time + start_ticks / os.sysconf("SC_CLK_TCK")


def report_worker_timeline(payload, result_queue) -> None:
    """spawn 子进程的 target: 拿到 payload 即说明 unpickle 已完成, 回报各阶段时间戳后退出。"""
    main_module = sys.modules["__mp_main__"]
    result_queue.put({
        "pid": os.getpid(),
        "process_start": read_process_start_time(),
        "import_start": main_module.MODULE_IMPORT_START,
        "import_end": main_module.MODULE_IMPORT_END,
        "target_enter": time.time(),
        "payload_type": type(payload).__name__,
    })


def spawn_workers(payload, num_workers: int, label: str) -> None:
    """像 torch DataLoader 一样逐个 start() worker, 记录父进程每次 start() 的阻塞时长与子进程时间线。"""
    assert num_workers > 0, "num_workers 必须为正"
    spawn_context = multiprocessing.get_context("spawn")
    result_queue = spawn_context.Queue()
    workers = []
    loop_start = time.time()
    for worker_idx in range(num_workers):
        worker = spawn_context.Process(target=report_worker_timeline, args=(payload, result_queue), daemon=True)
        start_call = time.perf_counter()
        worker.start()
        logger.info(f"[{label}] worker {worker_idx}: 父进程 start() 阻塞 {time.perf_counter() - start_call:.2f}s")
        workers.append(worker)
    logger.info(f"[{label}] {num_workers} 个 worker 全部 start() 返回, 父进程循环共 {time.time() - loop_start:.2f}s")

    timelines = sorted((result_queue.get(timeout=300) for _ in workers), key=lambda t: t["process_start"])
    for worker in workers:
        worker.join(timeout=60)
    for timeline in timelines:
        logger.info(
            f"[{label}] pid {timeline['pid']}: 创建于 +{timeline['process_start'] - loop_start:5.2f}s | "
            f"解释器启动 {timeline['import_start'] - timeline['process_start']:.2f}s | "
            f"重新 import 主模块 {timeline['import_end'] - timeline['import_start']:.2f}s | "
            f"unpickle {timeline['payload_type']} {timeline['target_enter'] - timeline['import_end']:.2f}s | "
            f"就绪于 +{timeline['target_enter'] - loop_start:5.2f}s")


def check_forkserver_loader(dataset) -> None:
    """用 data_loader_lingyu.TorchDataLoader(forkserver) 取一个 batch, 记录 worker 启动耗时并与主进程取样逐元素比对。"""
    data_loader = norm_script._data_loader.TorchDataLoader(
        dataset, local_batch_size=len(_SAMPLE_IDXS), sampler=_SAMPLE_IDXS,
        num_batches=1, num_workers=_LOADER_WORKERS, framework="pytorch")
    torch_loader = data_loader.torch_loader
    assert torch_loader.multiprocessing_context.get_start_method() == "forkserver", "TorchDataLoader 未使用 forkserver"

    start_time = time.perf_counter()
    torch_iter = iter(torch_loader)  # 返回时全部 worker 已 start()
    startup_seconds = time.perf_counter() - start_time
    start_time = time.perf_counter()
    loader_batch = next(torch_iter)
    batch_seconds = time.perf_counter() - start_time
    logger.info(f"[forkserver loader] {_LOADER_WORKERS} 个 worker 启动 {startup_seconds:.2f}s, "
                f"取第一个 batch {batch_seconds:.2f}s")
    del torch_iter

    expected_batch = norm_script._data_loader._collate_fn([dataset[sample_idx] for sample_idx in _SAMPLE_IDXS])
    assert jax.tree.structure(loader_batch) == jax.tree.structure(expected_batch), "batch 结构与主进程取样不一致"
    for loader_leaf, expected_leaf in zip(jax.tree.leaves(loader_batch), jax.tree.leaves(expected_batch)):
        np.testing.assert_array_equal(loader_leaf, expected_leaf)
    logger.info(f"[forkserver loader] batch 与主进程直接取样逐元素一致: "
                f"{ {key: np.shape(value) for key, value in loader_batch.items()} }")


def test():
    """依次拆解 import 耗时、dataset 序列化开销, 再对比两种参数下 worker 的启动方式。"""
    measure_import_breakdown()

    dataset = build_norm_dataset(_CONFIG_NAME)
    start_time = time.perf_counter()
    dataset_bytes = pickle.dumps(dataset)
    dump_seconds = time.perf_counter() - start_time
    start_time = time.perf_counter()
    pickle.loads(dataset_bytes)  # 字节是本进程刚 dumps 出来的, 来源可信
    load_seconds = time.perf_counter() - start_time
    logger.info(f"[pickle] dataset: {len(dataset)} samples / {len(dataset._dataset.episodes)} episodes, "
                f"pickle {len(dataset_bytes) / 1024:.0f} KiB, dumps {dump_seconds * 1000:.0f}ms, "
                f"loads {load_seconds * 1000:.0f}ms; Linux 管道缓冲默认 64 KiB")

    spawn_workers(None, _NUM_WORKERS, "小参数 None")
    spawn_workers(dataset, _NUM_WORKERS, "真实 dataset")
    check_forkserver_loader(dataset)


if __name__ == "__main__":
    test()
