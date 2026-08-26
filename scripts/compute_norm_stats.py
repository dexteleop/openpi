"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from collections.abc import Sequence
import glob
import io
import os
import tarfile

import numpy as np
from torch.utils.data import IterableDataset as TorchIterableDataset
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class SkipVideoDataset:
    """Wraps a raw LeRobotDataset to skip expensive video decoding.

    Replaces video frame tensors with dummy data,
    since compute_norm_stats only needs state and actions.
    Must wrap the LeRobotDataset *before* TransformedDataset.
    """

    def __init__(self, dataset):
        self._dataset = dataset
        self._video_keys = set(dataset.meta.video_keys)

    def __getitem__(self, index):
        # Access the underlying hf_dataset row directly (no video decode)
        item = self._dataset.hf_dataset[index]
        ep_idx = item["episode_index"].item()

        # Handle delta_timestamps (action sequences) without video
        if self._dataset.delta_indices is not None:
            query_indices, padding = self._dataset._get_query_indices(index, ep_idx)
            query_result = self._dataset._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Insert dummy image tensors for video keys
        for vid_key in self._video_keys:
            item[vid_key] = np.zeros((3, 224, 224), dtype=np.float32)

        # Add task string
        task_idx = item["task_index"].item()
        item["task"] = self._dataset.meta.tasks[task_idx]

        return item

    def __len__(self):
        return len(self._dataset)


def _create_torch_dataset_skip_video(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> _data_loader.Dataset:
    """Same as create_torch_dataset but wraps LeRobotDataset with SkipVideoDataset before transforms."""
    from lerobot.common.datasets import lerobot_dataset

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set.")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    # Wrap *before* TransformedDataset so we have access to raw LeRobotDataset internals
    print("Skipping video decoding (using dummy images) for faster norm stats computation.")
    dataset = SkipVideoDataset(dataset)

    if data_config.prompt_from_task:
        from openpi import transforms as _transforms
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    skip_video: bool = False,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if skip_video:
        dataset = _create_torch_dataset_skip_video(data_config, action_horizon, model_config)
    else:
        dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
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


class WDSStateActionDataset(TorchIterableDataset):
    """Iterates the WDS .tar shards reading only .state_action.npz.

    compute_norm_stats needs nothing but state / action, so this bypasses the
    whole training WDS pipeline (webdataset + decord video decode + VideoReader
    warmup, which costs ~5s per mp4). Reading the tars sequentially with
    tarfile is enough: no shuffling is needed since RunningStats is
    order-independent, and every sample must be visited exactly once anyway.

    Yields the same tuple layout as the training pipeline's decode_images
    (key, frames, state, action) so the config's WDSTuple2Dict repack transform
    applies unchanged; `frames` carries dummy images (see _DUMMY_FRAME).
    """

    # 1:1 aspect so TeleavatarInputs._extract_stereo_view's `width >= 2*height`
    # guard leaves it alone, and small so stacking it into a batch is cheap.
    # Never read: RunningStats only touches state / actions.
    _DUMMY_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)

    def __init__(self, dataset_dir: str, topics: Sequence[str]):
        self._tars = sorted(glob.glob(os.path.join(dataset_dir, "*.tar")))
        if not self._tars:
            raise FileNotFoundError(f"No .tar files found in {dataset_dir}")
        self._topics = list(topics)
        self._num_samples: int | None = None

    def __iter__(self):
        for tar_path in self._tars:
            with tarfile.open(tar_path) as tf:
                for member in tf:
                    if not member.name.endswith(".state_action.npz"):
                        continue
                    npz = np.load(io.BytesIO(tf.extractfile(member).read()))
                    frames = dict.fromkeys(self._topics, self._DUMMY_FRAME)
                    yield (
                        member.name.split(".")[0],
                        frames,
                        npz["state"].astype(np.float32),
                        npz["action"].astype(np.float32),
                    )

    def __len__(self) -> int:
        # Counting member names (no payload read) over all shards; ~3s for the
        # 478-shard / 47760-sample set. Cached, since main() needs it before
        # iterating to size the tqdm bar.
        if self._num_samples is None:
            total = 0
            for tar_path in self._tars:
                with tarfile.open(tar_path) as tf:
                    total += sum(1 for m in tf if m.name.endswith(".state_action.npz"))
            self._num_samples = total
        return self._num_samples


def create_wds_dataloader(
    data_config: _config.DataConfig,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.DataLoader, int]:
    """Norm-stats loader for the Lingyu WDS .tar datasets.

    Same transform chain as create_torch_dataloader (repack -> data_transforms
    -> RemoveStrings), only the source dataset differs, so the resulting
    norm_stats.json is identical in format to the LeRobot path.
    """
    assert data_config.wds_data_dir is not None, "wds_data_dir must be set for the WDS loader."

    # The repack group starts with WDSTuple2Dict, whose topic_camera_mapping is
    # the authoritative topic list; reuse it instead of hardcoding the topics.
    topics = next(
        (tf.topic_camera_mapping for tf in data_config.repack_transforms.inputs if isinstance(tf, transforms.WDSTuple2Dict)),
        {},
    )
    dataset = WDSStateActionDataset(data_config.wds_data_dir, topics)
    num_samples = len(dataset)

    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < num_samples:
        num_batches = max_frames // batch_size
    else:
        num_batches = num_samples // batch_size

    # num_workers=0: reading npz off local disk is not the bottleneck, and a
    # single pass keeps every sample counted exactly once.
    data_loader = _data_loader.WDSDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=0,
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
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
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


def main(config_name: str, max_frames: int | None = None, skip_video: bool = True):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.wds_data_dir is not None:
        data_loader, num_batches = create_wds_dataloader(data_config, config.batch_size, max_frames)
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames,
            skip_video=skip_video,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
