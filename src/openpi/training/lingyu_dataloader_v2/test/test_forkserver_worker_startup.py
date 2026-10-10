"""拆解 forkserver 下 DataLoader worker 逐个启动时每个 worker 的耗时, 以及关闭 DataLoader 时的等待。

主模块与 scripts/train.py 一样在顶层 import train, 故子进程重新执行主模块时的开销与训练时相同。
用与 data_loader_lingyu.TorchDataLoader 相同的参数直接建 torch DataLoader(forkserver + 同一组 preload),
只把 worker_init_fn 换成回报时间戳的版本, 子进程不取样也不碰 GPU。

核对::
    1. 预加载 '__main__' 为何不生效: forkserver 只认 preparation data 里的 'main_path',
       spawn.get_preparation_data 产出的却是 'init_main_from_path'
    2. 父进程每次 start() 都要重新 pickle 一遍 dataset, 记下体积与耗时
    3. 三种 preload 下每个 worker 的时间线:
       fork 出来 -> 重新执行主模块(import train 的全部依赖) -> unpickle dataset -> 进入 worker_init_fn
       现状 / 额外预加载 'train'(3.11 的 forkserver.main 收下 sys_path 却从不设置, import train 静默失败) /
       按包名预加载 train.py 的全部顶层依赖
    4. 关闭: worker 正在取一个 batch(64 个样本)时, _shutdown_workers 对每个 worker 依次 join(timeout=5s)
    5. 修复后的 data_loader_lingyu.TorchDataLoader: 每次 start() 的阻塞时长, 以及关掉迭代器时
       直接 terminate 全部 worker 的耗时(worker 同样都在取样), 关完后不留存活的 worker
dataset 与训练相同(读图像, 会在各 GPU 起解码进程), 关闭时 worker 都在取各自的第一个 batch。

必须以脚本方式运行, forkserver 的子进程才会像 train.py 那样把本文件当 __mp_main__ 重新执行::
    python src/openpi/training/lingyu_dataloader_v2/test/test_forkserver_worker_startup.py
"""
import sys
import time

# 子进程重新执行本文件时, 这两个时间戳界定了"重新执行主模块"这一段; 模块数之差即这一段新 import 的模块数
MODULE_IMPORT_START = time.time()
MODULE_COUNT_START = len(sys.modules)

import cProfile  # noqa: E402

# worker(__mp_main__)重新执行本文件时全程 profile, 看这段时间花在哪; 主进程不 profile
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

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # JAX 不碰 GPU; GPU 只给解码进程用

_SCRIPTS_DIR = Path(__file__).resolve().parents[5] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import train  # noqa: E402  与 scripts/train.py 作为主模块时同一组顶层 import

import random  # noqa: E402

import torch  # noqa: E402

import openpi.training.data_loader_lingyu as _data_loader  # noqa: E402
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import (  # noqa: E402
    LingyuDatasetV2, load_video_decode_config)
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger  # noqa: E402

MODULE_IMPORT_END = time.time()
MODULE_COUNT_END = len(sys.modules)
# 子进程把时间线与 profile 写到这里; 路径经环境变量传给 forkserver 起的子进程
_TIMELINE_DIR_ENV = "FORKSERVER_TIMELINE_DIR"
if _MAIN_RERUN_PROFILER is not None:
    _MAIN_RERUN_PROFILER.disable()
    if _TIMELINE_DIR_ENV in os.environ:  # spawn 起的 GPU 解码进程也是 __mp_main__, 但没有这个环境变量
        _MAIN_RERUN_PROFILER.dump_stats(f"{os.environ[_TIMELINE_DIR_ENV]}/{os.getpid()}.prof")

# 子进程里本文件名为 __mp_main__, 不能再建 FileHandler(mode="w"), 否则会清空父进程的日志
logger = get_logger(__file__) if __name__ == "__main__" else None

_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"
_ICEBERG_DIR = str(Path(__file__).resolve().parents[1] / "iceberg_warehouse")
_NUM_WORKERS = 8  # 足以看出每个 worker 的固定开销
_BATCH_SIZE = 64  # 与训练相同, 关闭时每个 worker 手上都是一整个 batch
# 现状(TorchDataLoader 里写死的)与额外预加载 train 两种 preload
_PRELOAD_VARIANTS = {
    "现状 preload": ["openpi.training.data_loader_lingyu", "openpi.training.config_lingyu"],
    # forkserver.main 收下 sys_path 却从不设置, scripts/ 不在其 sys.path 上, import train 的 ImportError 被静默吞掉
    "额外预加载 train": ["openpi.training.data_loader_lingyu", "openpi.training.config_lingyu", "train"],
    # 直接按包名预加载 train.py 顶层 import 的全部模块, 不依赖 sys.path
    "预加载 train.py 的全部依赖": [
        "openpi.training.data_loader_lingyu", "openpi.training.config_lingyu", "etils.epath", "flax.nnx",
        "flax.training.common_utils", "flax.traverse_util", "jax", "optax", "tqdm_loggable.auto", "wandb",
        "openpi.models.model", "openpi.shared.array_typing", "openpi.shared.nnx_utils", "openpi.training.checkpoints",
        "openpi.training.optimizer", "openpi.training.sharding", "openpi.training.utils",
        "openpi.training.weight_loaders"],
}
_PROFILE_TOP = 20  # 记录 profile 里累计耗时最多的前几项


def build_train_dataset():
    """与 train.py -> data_loader_lingyu.create_torch_data_loader 相同的 dataset 与 transforms。"""
    config = train._config.get_config(_CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    # 只看启动与关闭, 不关心数值, 故跳过 norm stats; 其余 transforms 与训练一致
    video_key_to_crop, image_resolution = load_video_decode_config(data_config)
    dataset = LingyuDatasetV2(_ICEBERG_DIR, video_key_to_crop=video_key_to_crop, image_resolution=image_resolution)
    return _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=True)


def read_process_start_time() -> float:
    """从 /proc 读本进程被内核创建的绝对时间(10ms 精度)。

    用 当前时间 - (开机时长 - 进程创建时的开机时长) 推算; 不用 /proc/stat 的 btime: 它是取整的秒数,
    本机实测比真实开机时刻早 1.05s, 会让每个 worker 的 "fork->重新执行主模块" 都凭空多出约 1s。
    """
    uptime_seconds = float(open("/proc/uptime").read().split()[0])
    start_ticks = int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19])
    return time.time() - (uptime_seconds - start_ticks / os.sysconf("SC_CLK_TCK"))


def report_worker_timeline(worker_id: int) -> None:
    """worker_init_fn: 进到这里说明主模块已重新执行完、dataset 已 unpickle 完, 把各阶段时间戳写成 json。"""
    main_module = sys.modules["__mp_main__"]
    # /proc/self/stat 第 10/12/14/15 项: minflt, majflt, utime, stime(单位 clock tick)
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
    """核对 '__main__' 预加载失效的原因: forkserver 要的键与 spawn 给的键对不上。"""
    preparation_keys = set(multiprocessing.spawn.get_preparation_data("ignore"))
    logger.info(f"[preload] spawn.get_preparation_data 的键: {sorted(preparation_keys)}")
    logger.info(f"[preload] ForkServer.ensure_running 只转交 {{'main_path', 'sys_path'}} -> "
                f"'main_path' in 键={'main_path' in preparation_keys}, "
                f"故 forkserver.main(main_path=None), '__main__' 预加载被跳过; Python {sys.version.split()[0]}")
    assert "main_path" not in preparation_keys and "init_main_from_path" in preparation_keys


def measure_parent_pickle(dataset) -> None:
    """父进程每次 start() 都会把 Process 对象(含 dataset)重新 pickle 一遍, 记下体积与耗时。"""
    start_time = time.perf_counter()
    dataset_bytes = pickle.dumps(dataset)
    dump_seconds = time.perf_counter() - start_time
    logger.info(f"[pickle] dataset {len(dataset_bytes) / 1024:.0f} KiB, dumps {dump_seconds * 1000:.0f}ms "
                f"(每个 worker 一次); Linux 管道缓冲 64 KiB, 子进程读完前父进程的 write 一直阻塞")


def start_loader_workers(dataset, preload: list[str], label: str):
    """按 TorchDataLoader 的参数建 forkserver DataLoader 并起全部 worker, 返回 (迭代器, 父进程逐个 start 的耗时)。"""
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
    # 只为计时: 包一层 start, 记录父进程每次 start() 的阻塞时长
    multiprocessing.process.BaseProcess.start = lambda process: start_seconds.append(
        _timed_call(original_start, process))
    try:
        loader_iter = iter(torch_loader)  # 返回时全部 worker 已 start()
    finally:
        multiprocessing.process.BaseProcess.start = original_start
    logger.info(f"[{label}] {_NUM_WORKERS} 个 worker 全部 start() 返回, 共 {time.time() - loop_start:.2f}s; "
                f"每次 start() 阻塞 {[round(seconds, 2) for seconds in start_seconds]}")
    return loader_iter, loop_start


def _timed_call(function, *args) -> float:
    """调用 function(*args), 返回耗时秒数。"""
    start_time = time.perf_counter()
    function(*args)
    return time.perf_counter() - start_time


def log_worker_timelines(timeline_dir: str, loop_start: float, label: str) -> None:
    """等全部 worker 写完时间线, 逐个记录各阶段耗时。"""
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

    # 最后一个 worker 的 profile: 重新执行主模块这段, 累计耗时最多的调用
    profile_path = sorted(Path(timeline_dir).glob("*.prof"), key=os.path.getmtime)[-1]
    profile_text = io.StringIO()
    pstats.Stats(str(profile_path), stream=profile_text).sort_stats("cumulative").print_stats(_PROFILE_TOP)
    logger.info(f"[{label}] 重新执行主模块的 profile({profile_path.name}):\n{profile_text.getvalue()}")


def measure_shutdown(loader_iter, label: str) -> None:
    """worker 正在取样时关闭: _shutdown_workers 依次 join(timeout=MP_STATUS_CHECK_INTERVAL), 记录每个 join 的耗时。"""
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
    """修复后的 TorchDataLoader: 预加载 train.py 依赖后 start() 不再逐个阻塞数秒, 关迭代器时直接 terminate worker。"""
    multiprocessing.forkserver._forkserver._stop()  # 用 TorchDataLoader 自己设置的 preload 起新的 forkserver
    data_loader = _data_loader.TorchDataLoader(
        dataset, local_batch_size=_BATCH_SIZE, shuffle=True, num_workers=_NUM_WORKERS, framework="pytorch")
    start_seconds = []
    original_start = multiprocessing.process.BaseProcess.start
    multiprocessing.process.BaseProcess.start = lambda process: start_seconds.append(
        _timed_call(original_start, process))
    start_time = time.perf_counter()
    try:
        batch_iter = iter(data_loader)
        first_batch = next(batch_iter)  # 生成器里的 iter(torch loader) 起全部 worker, 再等第一个 batch
    finally:
        multiprocessing.process.BaseProcess.start = original_start
    logger.info(f"[修复后 TorchDataLoader] 第一个 batch(含 forkserver 与 worker 启动) {time.perf_counter() - start_time:.2f}s, "
                f"每次 start() 阻塞 {[round(seconds, 2) for seconds in start_seconds]}, "
                f"actions {tuple(first_batch['actions'].shape)}")
    assert len(start_seconds) == _NUM_WORKERS, f"应起 {_NUM_WORKERS} 个 worker, 实际 {len(start_seconds)}"
    # 第一个 start() 含 forkserver 自身的启动与预加载, 只看其后
    assert max(start_seconds[1:]) < 1.0, "worker 仍在逐个阻塞, train.py 的依赖没有被预加载"

    workers = list(data_loader.torch_loader._iterator._workers)
    start_time = time.perf_counter()
    batch_iter.close()  # 与 train.py 结束时的 data_iter.close() 相同
    close_seconds = time.perf_counter() - start_time
    logger.info(f"[修复后 TorchDataLoader] 关闭 {len(workers)} 个正在取样的 worker 共 {close_seconds:.2f}s, "
                f"存活 {sum(worker.is_alive() for worker in workers)} 个")
    assert not any(worker.is_alive() for worker in workers), "关闭后仍有 worker 存活"
    assert close_seconds < torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL, "关闭仍在逐个 join 等超时"


def test():
    """核对 preload 失效原因与 pickle 开销, 再对比两种 preload 下 worker 的启动时间线与关闭耗时。"""
    check_main_preload_key()
    dataset = build_train_dataset()
    measure_parent_pickle(dataset)
    for label, preload in _PRELOAD_VARIANTS.items():
        # 每种 preload 都要一个新的 forkserver: 已在运行的 server 不会重新预加载
        multiprocessing.forkserver._forkserver._stop()
        timeline_dir = tempfile.mkdtemp(prefix="forkserver_timeline_")
        os.environ[_TIMELINE_DIR_ENV] = timeline_dir
        loader_iter, loop_start = start_loader_workers(dataset, preload, label)
        log_worker_timelines(timeline_dir, loop_start, label)
        measure_shutdown(loader_iter, label)
    check_fixed_torch_data_loader(dataset)


if __name__ == "__main__":
    test()
