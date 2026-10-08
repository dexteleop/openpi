"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from collections.abc import Sequence
import dataclasses
import glob
import io
import os
import tarfile

import numpy as np
from torch.utils.data import IterableDataset as TorchIterableDataset
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config_lingyu as _config
import openpi.training.data_loader_lingyu as _data_loader
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import LingyuDatasetV2
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    # 只读 state/action 的 topic, 视频既不下载也不解码
    dataset = LingyuDatasetV2(data_config.iceberg_dir, load_images=False)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    assert isinstance(config.data, _config.LingyuTeleavatarV2DataConfig), \
        f"{config_name} 不是 LingyuTeleavatarV2DataConfig, 不能用本脚本计算 norm stats"
    # 强制走只含 state/action 的 transforms, 训练 config 里无需打开 compute_norm_stats
    data_config = dataclasses.replace(config.data, compute_norm_stats=True).create(config.assets_dirs, config.model)

    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.num_workers, max_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # repo_id 为 None 时直接写到 assets_dirs 下
    output_path = config.assets_dirs if data_config.repo_id is None else config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
