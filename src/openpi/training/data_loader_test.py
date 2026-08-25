import dataclasses
import typing

import jax
import torch.utils.data

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _FakeLeRobotDatasetMeta:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.fps = 10
        self.tasks = {0: "do the thing"}


class _FakeLeRobotDataset:
    """Stand-in for lerobot_dataset.LeRobotDataset, keyed by repo_id, that avoids touching real data."""

    LENGTHS: typing.ClassVar = {"repo_a": 5, "repo_b": 7}

    def __init__(self, repo_id: str, delta_timestamps=None):
        self._repo_id = repo_id

    def __len__(self) -> int:
        return self.LENGTHS[self._repo_id]

    def __getitem__(self, index):
        return {"task_index": 0, "value": index}


def test_create_torch_dataset_concatenates_multiple_repo_ids(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeLeRobotDatasetMeta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)

    data_config = _config.DataConfig(repo_id=["repo_a", "repo_b"], action_sequence_keys=("actions",))
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon=1, model_config=None)

    assert isinstance(dataset, torch.utils.data.ConcatDataset)
    assert len(dataset) == sum(_FakeLeRobotDataset.LENGTHS.values())
    assert dataset[0] == {"task_index": 0, "value": 0}
    # Index 5 falls in the second (repo_b) dataset, at local index 0.
    assert dataset[5] == {"task_index": 0, "value": 0}


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
