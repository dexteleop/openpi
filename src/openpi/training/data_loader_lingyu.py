from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import time
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import LingyuDatasetV2
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import load_video_decode_config
import openpi.transforms as _transforms

# torch.utils.data.DataLoader picks map-style vs iterable-style with an
# isinstance() check, so a dataset handed to it must really subclass this one.

T_co = TypeVar("T_co", covariant=True)
# scripts/train.py 的全部顶层 import(按包名): forkserver 预加载它们, worker 重新执行 train.py 时就都已在 sys.modules 里
_TRAIN_SCRIPT_IMPORTS = [
    "etils.epath", "flax.nnx", "flax.training.common_utils", "flax.traverse_util", "jax", "optax",
    "tqdm_loggable.auto", "wandb", "openpi.models.model", "openpi.shared.array_typing", "openpi.shared.nnx_utils",
    "openpi.training.checkpoints", "openpi.training.optimizer", "openpi.training.sharding", "openpi.training.utils",
    "openpi.training.weight_loaders",
]


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def create_lingyu_dataset_v2(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig
) -> Dataset:
    logging.info(f"action_dim: {model_config.action_dim}")
    logging.info(f"action_horizon: {model_config.action_horizon}")
    logging.info(f"max_token_len: {model_config.max_token_len}")
    # GPU 解码的裁剪与缩放参数取自本 data_config 的 openpi 图像变换, 不另行配置
    video_key_to_crop, image_resolution = load_video_decode_config(data_config)
    return LingyuDatasetV2(data_config.iceberg_dir, video_key_to_crop=video_key_to_crop,
                           image_resolution=image_resolution)


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_lingyu_dataset_v2(data_config, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            # forkserver: 重型模块只在 server 进程里 import 一次, worker 从 server fork 出来;
            # spawn 下每个 worker 都要重跑主模块的 import(~5s), 且因 dataset pickle 超过管道缓冲而逐个串行启动。
            # Python 3.11 默认的 ['__main__'] 预加载不生效, 须显式列出模块名; 且 forkserver 不设置 sys_path,
            # 只能 import site-packages/src 下的包。worker 仍会重新执行主模块(train.py), 其顶层依赖若没预加载,
            # 每个 worker 要现 import 约 1170 个模块(~3.5s), 又因上面的管道阻塞逐个串行, 故按包名全部预加载
            mp_context = multiprocessing.get_context("forkserver")
            mp_context.set_forkserver_preload([__name__, "openpi.training.config_lingyu", *_TRAIN_SCRIPT_IMPORTS])

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            prefetch_factor=2 if num_workers > 0 else None,
            in_order=False if num_workers > 0 else True,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    device_batch = jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    device_batch = jax.tree.map(torch.as_tensor, batch)
                try:
                    yield device_batch
                except GeneratorExit:
                    # 使用方提前关掉了迭代器(train.py 结束时 data_iter.close()), 手上的 batch 已无用
                    _terminate_workers(data_iter)
                    raise


def _terminate_workers(torch_iter) -> None:
    """直接 terminate 全部 worker 再收尾, 不走 torch 默认的逐个 join。

    torch 的 _shutdown_workers 对每个 worker 依次 join(timeout=5s), 而 worker 要取完手上整个 batch
    才响应退出, 关闭时 worker 几乎都在取样, 256 个 worker 最多要等约 21 分钟。
    """
    workers = getattr(torch_iter, "_workers", None)
    if workers is None:  # num_workers=0, 没有 worker 进程
        return
    # 先摘掉 SIGCHLD 监视, 否则 torch 会把被 terminate 的 worker 当成异常退出, 在主进程里报错
    if torch_iter._worker_pids_set:
        torch.utils.data._utils.signal_handling._remove_worker_pids(id(torch_iter))
        torch_iter._worker_pids_set = False
    terminate_start = time.perf_counter()
    for worker in workers:
        worker.terminate()
    torch_iter._shutdown_workers()  # 此时 join 的都是已退出的进程, 只剩关队列等收尾
    logging.info(f"Terminated {len(workers)} data loader workers in {time.perf_counter() - terminate_start:.1f}s")


def _fetch_and_put(data_iter, sharding):
    """Fetch one batch from the worker pool and place it on the devices."""
    batch = next(data_iter)
    return jax.tree.map(lambda x: jax.make_array_from_process_local_data(sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
