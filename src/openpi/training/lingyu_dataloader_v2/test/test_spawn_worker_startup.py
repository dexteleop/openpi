"""Break down one DataLoader worker's startup cost under spawn, and verify why workers start one after another.

Reproduce how torch DataLoader starts workers: in a for loop call
spawn-context Process(target, args=(dataset, ...)).start() per worker. Children do not sample, only report per-stage timestamps,
sending no requests to OSS and using no GPU.

Checks::
    1. Import cost of each package in the main module (the same imports as compute_norm_stats_lingyu.py)
    2. The dataset's pickle size, plus pickle / unpickle durations
    3. Child stages: interpreter start -> re-import of the main module -> unpickle dataset -> enter target
    4. How long Process.start() blocks the parent: tiny args vs args being the real dataset
    5. TorchDataLoader switched to forkserver: worker startup cost, and fetched batches exactly matching main-process sampling
       (fetching only the single _SAMPLE_IDXS batch, about 2 x 360 S3 requests)

Must run as a script so children re-execute this file as __mp_main__ just like compute_norm_stats_lingyu.py::
    python src/openpi/training/lingyu_dataloader_v2/test/test_spawn_worker_startup.py
"""
import time

# When a child re-executes this file, these two timestamps bound the "re-import the main module" span
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

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # Do not touch GPUs used by running jobs

_SCRIPTS_DIR = Path(__file__).resolve().parents[5] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import compute_norm_stats_lingyu as norm_script  # noqa: E402  exactly the same top-level imports as the script under test

import jax  # noqa: E402
import numpy as np  # noqa: E402

from openpi.training.lingyu_dataloader_v2.test.logger import get_logger  # noqa: E402

MODULE_IMPORT_END = time.time()

# In children this file is named __mp_main__; do not create a FileHandler(mode="w") again, or the parent's log would be truncated
logger = get_logger(__file__) if __name__ == "__main__" else logging.getLogger(__name__)

_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"
_NUM_WORKERS = 4      # Workers started per parameter set, enough to expose the serial pattern
_TOP_PACKAGES = 15    # How many packages to show in the import-time ranking
_LOADER_WORKERS = 8   # Worker count for the TorchDataLoader check; under spawn about 8 x 5.3s
_SAMPLE_IDXS = [0, 1]  # The single batch TorchDataLoader fetches, used as the sampler


def build_norm_dataset(config_name: str):
    """Build the dataset exactly like compute_norm_stats_lingyu.create_torch_dataloader (no DataLoader built)."""
    config = norm_script._config.get_config(config_name)
    data_config = dataclasses.replace(config.data, compute_norm_stats=True).create(config.assets_dirs, config.model)
    dataset = norm_script.LingyuDatasetV2(data_config.iceberg_dir, load_images=False)
    return norm_script._data_loader.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, norm_script.RemoveStrings()],
    )


def measure_import_breakdown() -> float:
    """Import the script under test with -X importtime in a fresh interpreter, aggregate self time by top-level package into the log, and return wall-clock time."""
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
    """Read this process's absolute creation time by the kernel from /proc (10ms resolution), for the interpreter-start segment."""
    boot_time = next(int(line.split()[1]) for line in open("/proc/stat") if line.startswith("btime"))
    start_ticks = int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19])
    return boot_time + start_ticks / os.sysconf("SC_CLK_TCK")


def report_worker_timeline(payload, result_queue) -> None:
    """spawn child's target: reaching here means unpickle finished; report per-stage timestamps and exit."""
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
    """start() workers one by one like torch DataLoader, recording each start()'s parent blocking time and child timelines."""
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
    """Fetch one batch with data_loader_lingyu.TorchDataLoader (forkserver), record worker startup cost, and compare element-wise with main-process sampling."""
    data_loader = norm_script._data_loader.TorchDataLoader(
        dataset, local_batch_size=len(_SAMPLE_IDXS), sampler=_SAMPLE_IDXS,
        num_batches=1, num_workers=_LOADER_WORKERS, framework="pytorch")
    torch_loader = data_loader.torch_loader
    assert torch_loader.multiprocessing_context.get_start_method() == "forkserver", "TorchDataLoader 未使用 forkserver"

    start_time = time.perf_counter()
    torch_iter = iter(torch_loader)  # When this returns, all workers have started
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
    """Break down import time and dataset serialization cost, then compare worker startup under the two parameter sets."""
    measure_import_breakdown()

    dataset = build_norm_dataset(_CONFIG_NAME)
    start_time = time.perf_counter()
    dataset_bytes = pickle.dumps(dataset)
    dump_seconds = time.perf_counter() - start_time
    start_time = time.perf_counter()
    pickle.loads(dataset_bytes)  # The bytes were just dumped in this process, so the source is trusted
    load_seconds = time.perf_counter() - start_time
    logger.info(f"[pickle] dataset: {len(dataset)} samples / {len(dataset._dataset.episodes)} episodes, "
                f"pickle {len(dataset_bytes) / 1024:.0f} KiB, dumps {dump_seconds * 1000:.0f}ms, "
                f"loads {load_seconds * 1000:.0f}ms; Linux 管道缓冲默认 64 KiB")

    spawn_workers(None, _NUM_WORKERS, "小参数 None")
    spawn_workers(dataset, _NUM_WORKERS, "真实 dataset")
    check_forkserver_loader(dataset)


if __name__ == "__main__":
    test()
