"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro
import multiprocessing as mp

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class RemoveImages(transforms.DataTransformFn):
    """Remove image and image_mask keys since they don't need norm stats."""
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if k not in ("image", "image_mask")}


class TracedTransform:
    """Wrapper to trace and print the data flow through a transform (only once).

    Uses multiprocessing.Value so that across all worker processes spawned
    by the DataLoader, each transform prints at most once. The shared flag
    survives pickle serialization because multiprocessing.Value implements
    __reduce__ to reconstruct a handle to the same shared-memory block in
    child processes created via the "spawn" start method.
    """

    def __init__(self, obj):
        self.obj = obj
        self.name = getattr(obj, "__name__", obj.__class__.__name__)
        self._printed = mp.get_context("spawn").Value("b", False)

    def __call__(self, x: dict) -> dict:
        print_once = False
        with self._printed.get_lock():
            if not self._printed.value:
                self._printed.value = True
                print_once = True

        if print_once:
            print(f"\n[Run Transform] -> {self.name}")
            in_keys = set(x.keys())
            print(f"   -> Input Keys  : {list(in_keys)}")
        result = self.obj(x)
        if print_once:
            out_keys = set(result.keys())
            print(f"   <- Output Keys : {list(out_keys)}")

        return result


def trace_transform(transform_obj):
    return TracedTransform(transform_obj)


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    
    raw_transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
        RemoveStrings(),
        RemoveImages(),
    ]
    traced_transforms = [trace_transform(t) for t in raw_transforms]
    
    dataset = _data_loader.TransformedDataset(
        dataset,
        traced_transforms,
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


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    
    raw_transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
        RemoveStrings(),
    ]
    traced_transforms = [trace_transform(t) for t in raw_transforms]
    
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        traced_transforms,
        is_batched=True,
    )
    
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
        
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            val = np.asarray(batch[key])
            stats[key].update(val)

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
