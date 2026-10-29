"""Break down the per-worker startup cost of DataLoader workers under forkserver, and the wait when closing the DataLoader.

The main module imports train at top level just like scripts/train.py, so the children's cost of re-executing the main module matches training.
It builds a torch DataLoader directly with the same parameters as data_loader_lingyu.TorchDataLoader (forkserver + the same preload set),
only swapping worker_init_fn for a version that reports timestamps; children neither sample nor touch the GPU.

Checks::
    1. Why preloading '__main__' does not work: forkserver only honors 'main_path' in the preparation data,
       while spawn.get_preparation_data produces 'init_main_from_path'
    2. The parent re-pickles the dataset on every start(); record its size and duration
    3. Each worker's timeline under three preload modes:
       forked -> re-execute the main module (all of import train's dependencies) -> unpickle dataset -> enter worker_init_fn
       status quo / additionally preloading 'train' (3.11's forkserver.main accepts sys_path but never sets it, so import train fails silently) /
       preloading all top-level dependencies of train.py by package name
    4. Closing: while a worker is sampling one batch (64 samples), _shutdown_workers joins each worker with timeout=5s in turn
    5. The fixed data_loader_lingyu.TorchDataLoader: the blocking duration of each start(), and the cost of
       terminating all workers directly when the iterator is closed (workers are all sampling too), leaving no surviving workers afterwards
The dataset matches training (reads images, starts decode processes on each GPU); on close all workers are sampling their first batch.

Must run as a script so that forkserver's children re-execute this file as __mp_main__ just like train.py::
    python src/openpi/training/lingyu_dataloader_v2/test/test_forkserver_worker_startup.py
"""
import sys
import time

# When a child re-executes this file, these two timestamps bound the "re-execute the main module" span; the module-count difference is the modules newly imported in it
MODULE_IMPORT_START = time.time()
MODULE_COUNT_START = len(sys.modules)

import cProfile  # noqa: E402

# Profile the whole span when a worker (__mp_main__) re-executes this file, to see where the time goes; the main process does not profile
_MAIN_RERUN_PROFILER = cProfile.Profile() if __name__ == "__mp_main__" else None
if _MAIN_RERUN_PROFILER is not None:
    _MAIN_RERUN_PROFILER.enable()

import dataclasses  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import multiprocessing.forkserver  # noqa: E402
import multiprocessing.spawn  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import pstats  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # JAX never touches the GPU; the GPU is reserved for decode processes

_SCRIPTS_DIR = Path(__file__).resolve().parents[5] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import train  # noqa: E402  same top-level imports as scripts/train.py as the main module

import random  # noqa: E402

import torch  # noqa: E402

import openpi.training.data_loader as _data_loader  # noqa: E402
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import (  # noqa: E402
    LingyuDatasetV2, load_video_decode_config)
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger  # noqa: E402

MODULE_IMPORT_END = time.time()
MODULE_COUNT_END = len(sys.modules)
# Children write their timeline and profile here; the path is passed via an environment variable to forkserver-spawned children
_TIMELINE_DIR_ENV = "FORKSERVER_TIMELINE_DIR"
if _MAIN_RERUN_PROFILER is not None:
    _MAIN_RERUN_PROFILER.disable()
    if _TIMELINE_DIR_ENV in os.environ:  # GPU decode processes started via spawn are also __mp_main__, but lack this environment variable
        _MAIN_RERUN_PROFILER.dump_stats(f"{os.environ[_TIMELINE_DIR_ENV]}/{os.getpid()}.prof")

# In children this file is named __mp_main__; do not create a FileHandler(mode="w") again, or the parent's log would be truncated
logger = get_logger(__file__) if __name__ == "__main__" else None

_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"
_ICEBERG_DIR = str(Path(__file__).resolve().parents[1] / "iceberg_warehouse")
_NUM_WORKERS = 8  # Enough to expose each worker's fixed overhead
_BATCH_SIZE = 64  # Same as training; on close every worker holds a full batch
# Two preload modes: the status quo (hardcoded in TorchDataLoader) and additionally preloading train
_PRELOAD_VARIANTS = {
    "现状 preload": ["openpi.training.data_loader", "openpi.training.config"],
    # forkserver.main accepts sys_path but never sets it; scripts/ is not on its sys.path, so import train's ImportError is silently swallowed
    "额外预加载 train": ["openpi.training.data_loader", "openpi.training.config", "train"],
    # Preload all modules of train.py's top-level imports directly by package name, independent of sys.path
    "预加载 train.py 的全部依赖": [
        "openpi.training.data_loader", "openpi.training.config", "etils.epath", "flax.nnx",
        "flax.training.common_utils", "flax.traverse_util", "jax", "optax", "tqdm_loggable.auto", "wandb",
        "openpi.models.model", "openpi.shared.array_typing", "openpi.shared.nnx_utils", "openpi.training.checkpoints",
        "openpi.training.optimizer", "openpi.training.sharding", "openpi.training.utils",
        "openpi.training.weight_loaders"],
}
_PROFILE_TOP = 20  # How many top cumulative-time entries from the profile to record


def build_train_dataset():
    """The same dataset and transforms as train.py -> data_loader_lingyu.create_torch_data_loader."""
    config = train._config.get_config(_CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    # Only startup and shutdown matter, not values, so norm stats are skipped; all other transforms match training
    video_key_to_crop, image_resolution = load_video_decode_config(data_config)
    dataset = LingyuDatasetV2(_ICEBERG_DIR, video_key_to_crop=video_key_to_crop, image_resolution=image_resolution)
    return _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=True)


def read_process_start_time() -> float:
    """Read this process's absolute creation time by the kernel from /proc (10ms resolution).

    Computed as current time - (uptime - uptime at process creation); /proc/stat's btime is not used: it is rounded to seconds,
    measured 1.05s earlier than the real boot time on this machine, which would add a phantom ~1s to every worker's "fork -> re-execute main module".
    """
    uptime_seconds = float(open("/proc/uptime").read().split()[0])
    start_ticks = int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19])
    return time.time() - (uptime_seconds - start_ticks / os.sysconf("SC_CLK_TCK"))


def report_worker_timeline(worker_id: int) -> None:
    """worker_init_fn: reaching here means the main module has been re-executed and the dataset unpickled; write each stage's timestamps as json."""
    main_module = sys.modules["__mp_main__"]
    # /proc/self/stat fields 10/12/14/15: minflt, majflt, utime, stime (in clock ticks)
    proc_stat = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
    clock_ticks = os.sysconf("SC_CLK_TCK")
    timeline = {
        "new_modules": main_module.MODULE_COUNT_END - main_module.MODULE_COUNT_START,
        "minflt": int(proc_stat[7]), "majflt": int(proc_stat[9]),
        "cpu_seconds": (int(proc_stat[11]) + int(proc_stat[12])) / clock_ticks,
        "worker_id": worker_id,
        "process_start": read_process_start_time(),
        "import_start": main_module.MODULE_IMPORT_START,
        "import_end": main_module.MODULE_IMPORT_END,
        "init_enter": time.time(),
    }
    timeline_dir = os.environ[_TIMELINE_DIR_ENV]
    Path(timeline_dir, f"{worker_id}.json").write_text(json.dumps(timeline))


def check_main_preload_key() -> None:
    """Check why the '__main__' preload fails: forkserver's expected key does not match spawn's provided key."""
    preparation_keys = set(multiprocessing.spawn.get_preparation_data("ignore"))
    logger.info(f"[preload] spawn.get_preparation_data 的键: {sorted(preparation_keys)}")
    logger.info(f"[preload] ForkServer.ensure_running 只转交 {{'main_path', 'sys_path'}} -> "
                f"'main_path' in 键={'main_path' in preparation_keys}, "
                f"故 forkserver.main(main_path=None), '__main__' 预加载被跳过; Python {sys.version.split()[0]}")
    assert "main_path" not in preparation_keys and "init_main_from_path" in preparation_keys


def measure_parent_pickle(dataset) -> None:
    """The parent re-pickles the Process object (including the dataset) on every start(); record its size and duration."""
    start_time = time.perf_counter()
    dataset_bytes = pickle.dumps(dataset)
    dump_seconds = time.perf_counter() - start_time
    logger.info(f"[pickle] dataset {len(dataset_bytes) / 1024:.0f} KiB, dumps {dump_seconds * 1000:.0f}ms "
                f"(每个 worker 一次); Linux 管道缓冲 64 KiB, 子进程读完前父进程的 write 一直阻塞")


def start_loader_workers(dataset, preload: list[str], label: str):
    """Build a forkserver DataLoader with TorchDataLoader's parameters, start all workers, and return (iterator, main process's per-start durations)."""
    mp_context = multiprocessing.get_context("forkserver")
    mp_context.set_forkserver_preload(preload)
    torch_loader = torch.utils.data.DataLoader(
        dataset, batch_size=_BATCH_SIZE, num_workers=_NUM_WORKERS, multiprocessing_context=mp_context,
        persistent_workers=True, collate_fn=_data_loader._collate_fn, worker_init_fn=report_worker_timeline,
        prefetch_factor=2, in_order=False, drop_last=True,
        sampler=random.sample(range(len(dataset)), _NUM_WORKERS * 2 * _BATCH_SIZE))
    start_seconds = []
    original_start = multiprocessing.process.BaseProcess.start
    loop_start = time.time()
    # Timing only: wrap start to record how long each start() blocks the parent
    multiprocessing.process.BaseProcess.start = lambda process: start_seconds.append(
        _timed_call(original_start, process))
    try:
        loader_iter = iter(torch_loader)  # When this returns, all workers have started
    finally:
        multiprocessing.process.BaseProcess.start = original_start
    logger.info(f"[{label}] {_NUM_WORKERS} 个 worker 全部 start() 返回, 共 {time.time() - loop_start:.2f}s; "
                f"每次 start() 阻塞 {[round(seconds, 2) for seconds in start_seconds]}")
    return loader_iter, loop_start


def _timed_call(function, *args) -> float:
    """Call function(*args) and return the elapsed seconds."""
    start_time = time.perf_counter()
    function(*args)
    return time.perf_counter() - start_time


def log_worker_timelines(timeline_dir: str, loop_start: float, label: str) -> None:
    """Wait for all workers to finish writing timelines, then record each stage's duration."""
    deadline = time.time() + 600
    while len(list(Path(timeline_dir).glob("*.json"))) < _NUM_WORKERS and time.time() < deadline:
        time.sleep(0.5)
    timelines = sorted((json.loads(json_path.read_text()) for json_path in Path(timeline_dir).glob("*.json")),
                       key=lambda timeline: timeline["process_start"])
    assert len(timelines) == _NUM_WORKERS, f"只有 {len(timelines)} 个 worker 回报了时间线"
    for timeline in timelines:
        logger.info(
            f"[{label}] worker {timeline['worker_id']}: fork 于 +{timeline['process_start'] - loop_start:5.2f}s | "
            f"fork->重新执行主模块 {timeline['import_start'] - timeline['process_start']:.2f}s | "
            f"重新执行主模块 {timeline['import_end'] - timeline['import_start']:.2f}s | "
            f"unpickle+init {timeline['init_enter'] - timeline['import_end']:.2f}s | "
            f"就绪于 +{timeline['init_enter'] - loop_start:5.2f}s | 重新执行主模块新 import {timeline['new_modules']} 个模块 | "
            f"子进程 CPU {timeline['cpu_seconds']:.2f}s, 缺页 minor {timeline['minflt']} / major {timeline['majflt']}")
    worker_seconds = [later["init_enter"] - earlier["init_enter"] for earlier, later in zip(timelines, timelines[1:])]
    logger.info(f"[{label}] 相邻 worker 就绪间隔平均 {sum(worker_seconds) / len(worker_seconds):.2f}s")

    # The last worker's profile: the top cumulative-time calls in the re-execute-main-module span
    profile_path = sorted(Path(timeline_dir).glob("*.prof"), key=os.path.getmtime)[-1]
    profile_text = io.StringIO()
    pstats.Stats(str(profile_path), stream=profile_text).sort_stats("cumulative").print_stats(_PROFILE_TOP)
    logger.info(f"[{label}] 重新执行主模块的 profile({profile_path.name}):\n{profile_text.getvalue()}")


def measure_shutdown(loader_iter, label: str) -> None:
    """Close while workers are sampling: _shutdown_workers joins each with timeout=MP_STATUS_CHECK_INTERVAL in turn; record each join's duration."""
    join_seconds = []
    original_join = multiprocessing.process.BaseProcess.join
    multiprocessing.process.BaseProcess.join = lambda process, timeout=None: join_seconds.append(
        _timed_call(original_join, process, timeout))
    start_time = time.perf_counter()
    try:
        loader_iter._shutdown_workers()
    finally:
        multiprocessing.process.BaseProcess.join = original_join
    logger.info(f"[{label}] 关闭共 {time.perf_counter() - start_time:.2f}s, "
                f"MP_STATUS_CHECK_INTERVAL={torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL}s, "
                f"每个 join 耗时 {[round(seconds, 2) for seconds in join_seconds]}")


def check_fixed_torch_data_loader(dataset) -> None:
    """The fixed TorchDataLoader: after preloading train.py's dependencies start() no longer blocks for seconds one by one, and closing the iterator terminates workers directly."""
    multiprocessing.forkserver._forkserver._stop()  # Start a fresh forkserver with TorchDataLoader's own preload settings
    data_loader = _data_loader.TorchDataLoader(
        dataset, local_batch_size=_BATCH_SIZE, shuffle=True, num_workers=_NUM_WORKERS, framework="pytorch")
    start_seconds = []
    original_start = multiprocessing.process.BaseProcess.start
    multiprocessing.process.BaseProcess.start = lambda process: start_seconds.append(
        _timed_call(original_start, process))
    start_time = time.perf_counter()
    try:
        batch_iter = iter(data_loader)
        first_batch = next(batch_iter)  # iter(torch loader) inside the generator starts all workers, then waits for the first batch
    finally:
        multiprocessing.process.BaseProcess.start = original_start
    logger.info(f"[修复后 TorchDataLoader] 第一个 batch(含 forkserver 与 worker 启动) {time.perf_counter() - start_time:.2f}s, "
                f"每次 start() 阻塞 {[round(seconds, 2) for seconds in start_seconds]}, "
                f"actions {tuple(first_batch['actions'].shape)}")
    assert len(start_seconds) == _NUM_WORKERS, f"应起 {_NUM_WORKERS} 个 worker, 实际 {len(start_seconds)}"
    # The first start() includes the forkserver's own startup and preload; only the rest is meaningful
    assert max(start_seconds[1:]) < 1.0, "worker 仍在逐个阻塞, train.py 的依赖没有被预加载"

    workers = list(data_loader.torch_loader._iterator._workers)
    start_time = time.perf_counter()
    batch_iter.close()  # Same as train.py's data_iter.close() when training ends
    close_seconds = time.perf_counter() - start_time
    logger.info(f"[修复后 TorchDataLoader] 关闭 {len(workers)} 个正在取样的 worker 共 {close_seconds:.2f}s, "
                f"存活 {sum(worker.is_alive() for worker in workers)} 个")
    assert not any(worker.is_alive() for worker in workers), "关闭后仍有 worker 存活"
    assert close_seconds < torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL, "关闭仍在逐个 join 等超时"


def test():
    """Check the preload failure reason and pickle overhead, then compare worker startup timelines and shutdown cost under the two preload modes."""
    check_main_preload_key()
    dataset = build_train_dataset()
    measure_parent_pickle(dataset)
    for label, preload in _PRELOAD_VARIANTS.items():
        # Each preload mode needs a fresh forkserver: an already-running server does not preload again
        multiprocessing.forkserver._forkserver._stop()
        timeline_dir = tempfile.mkdtemp(prefix="forkserver_timeline_")
        os.environ[_TIMELINE_DIR_ENV] = timeline_dir
        loader_iter, loop_start = start_loader_workers(dataset, preload, label)
        log_worker_timelines(timeline_dir, loop_start, label)
        measure_shutdown(loader_iter, label)
    check_fixed_torch_data_loader(dataset)


if __name__ == "__main__":
    test()
