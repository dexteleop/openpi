import dataclasses
import io
import json
import logging
import pathlib
import tarfile

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.policies import teleavatar_v2_policy as _teleavatar
from openpi.shared import image_tools
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.lingyu_dataloader import webdataset_load_tar as _wds_tar

def _require_wds_data() -> _config.TrainConfig:
    """Return the WDS train config, skipping if its .tar shards are missing."""
    config = _config.get_config("pi0_teleavatar_v2_lingyu_wds")
    data_dir = config.data.base_config.wds_data_dir if config.data.base_config else None
    if data_dir is None or not list(pathlib.Path(data_dir).glob("*.tar")):
        pytest.skip(f"WDS shards not found in {data_dir!r}")
    return config


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


def test_wds_data_loader_batches():
    """End-to-end: create_data_loader must yield batch_size-shaped model inputs."""
    config = _require_wds_data()
    # Keep the batch small (frames are decoded on the fly and the head camera is
    # 1920x3840 each), but it must stay divisible by the device count: the
    # default sharding splits dim 0 across all devices.
    config = dataclasses.replace(
        config, batch_size=jax.device_count(), num_workers=0
    )

    loader = _data_loader.create_data_loader(
        config,
        # skip_norm_stats: the WDS norm_stats.json uses a flat schema that
        # openpi's _normalize.load does not accept, so the config resolves
        # norm_stats=None and Normalize would otherwise raise.
        skip_norm_stats=True,
        num_batches=2,
    )
    assert loader.data_config().wds_data_dir == config.data.base_config.wds_data_dir

    batches = list(loader)
    assert len(batches) == 2

    for observation, actions in batches:
        leaves = jax.tree.leaves((observation, actions))
        shapes = [tuple(x.shape) for x in leaves]
        assert all(x.shape[0] == config.batch_size for x in leaves), shapes
        assert actions.shape == (
            config.batch_size,
            config.model.action_horizon,
            config.model.action_dim,
        )
        assert sorted(observation.images) == [
            "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb",
        ]
        for name, image in observation.images.items():
            assert image.shape == (config.batch_size, 224, 224, 3), (name, image.shape)


def test_wds_data_loader_multi_worker():
    """num_workers>0 must work: the dataset has to survive being sent to workers."""
    config = _require_wds_data()
    loader = _data_loader.create_data_loader(
        config, skip_norm_stats=True, num_batches=2
    )
    batches = list(loader)

    assert len(batches) == 2
    for _, actions in batches:
        assert actions.shape == (
            config.batch_size,
            config.model.action_horizon,
            config.model.action_dim,
        )


# --- Content correctness: 10 batches vs ground truth read from the tars. -----

VERIFY_BATCHES = 10
# Image tolerance, in uint8 LSBs. resize_with_pad rounds a float32 bilinear
# result to uint8, and the reference call here compiles to a different XLA
# fusion than the one inside the transform chain, so pixels whose float value
# sits near .5 can round to either neighbour. Measured on this dataset: max 1
# LSB on ~4.4% of pixels. The bound is therefore on BOTH amplitude and count --
# a wrong frame, wrong stereo half or wrong camera moves whole regions by far
# more than 1 LSB and fails on amplitude alone.
VERIFY_IMAGE_ATOL = 1.001
VERIFY_IMAGE_MAX_DIFF_FRAC = 0.10
# Minimum L2 gap to the SECOND-closest ground-truth state. Measured over these
# shards the smallest gap is ~6e-3 (consecutive frames of one episode), so this
# only rejects a genuinely non-unique fingerprint, not normal near-neighbours.
VERIFY_STATE_MARGIN = 1e-3
VERIFY_LOG_PATH = pathlib.Path(__file__).parent / "lingyu_dataloader" / "wds_verify.log"


def _verify_logger() -> logging.Logger:
    """Per-run .log of every matched sample, for post-hoc inspection."""
    logger = logging.getLogger("wds_verify")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(VERIFY_LOG_PATH, mode="w")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def log_verification(fn):
    """Decorator: give the test a logger and record pass/fail around it.

    Deliberately NOT functools.wraps: that sets __wrapped__, and pytest then
    collects the wrapped signature and demands a `logger` FIXTURE. The wrapper
    must present a zero-argument signature instead.
    """

    def wrapper():
        logger = _verify_logger()
        logger.info("[START] %s", fn.__name__)
        try:
            result = fn(logger=logger)
        except BaseException as e:  # includes pytest.skip
            logger.info("[END] %s: %s", type(e).__name__, e)
            raise
        logger.info("[END] passed")
        return result

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _build_ground_truth(dataset_dir: str):
    """Read every tar directly and index samples by their 14-dim state.

    Returns (states, rows): `states` is an (N, 14) matrix of the state vector
    TeleavatarInputs extracts, used as a per-sample fingerprint; `rows` holds
    (key, shard, json meta, raw 62-dim action) for the matched row. Read with
    the stdlib tarfile, not webdataset, so the reference shares no code with
    the pipeline under test.
    """
    states, rows = [], []
    for tar_path in sorted(pathlib.Path(dataset_dir).glob("*.tar")):
        with tarfile.open(tar_path) as tf:
            metas, arrays = {}, {}
            for member in tf:
                key, _, ext = member.name.partition(".")
                blob = tf.extractfile(member).read()
                if ext == "json":
                    metas[key] = json.loads(blob)
                else:
                    arrays[key] = dict(np.load(io.BytesIO(blob), allow_pickle=True))
            for key, meta in metas.items():
                state = arrays[key]["state"].astype(np.float32).reshape(-1)
                # Same slices as TeleavatarInputs: left arm 0:7, right arm 8:15.
                states.append(np.concatenate([state[0:7], state[8:15]]))
                rows.append((key, tar_path.name, meta, arrays[key]["action"].astype(np.float32)))
    return np.stack(states), rows


def _expected_actions(raw_action: np.ndarray) -> np.ndarray:
    """Rebuild the 16-dim action TeleavatarInputs should produce from a raw row.

    Unnormalized: the loader is built with skip_norm_stats=True, which makes the
    Normalize stage a no-op (an empty norm_stats dict selects no keys), so the
    values that reach the model are these raw ones.
    """
    selected = np.concatenate(
        [raw_action[:, 0:7], raw_action[:, 39:40], raw_action[:, 8:15], raw_action[:, 47:48]],
        axis=1,
    )
    selected[:, 7] = _teleavatar._gripper_effort_to_trigger(selected[:, 7])
    selected[:, 15] = _teleavatar._gripper_effort_to_trigger(selected[:, 15])
    return selected


def _expected_base_image(meta: dict) -> np.ndarray:
    """Decode the head-camera frame the sample's json points at, and crop+resize it."""
    topic = "/xr_video_topic/ffmpeg"
    # decode_frame returns (1, C, H, W) float32 in [0, 1]; drop the leading
    # frame axis so _parse_image sees the (C, H, W) layout it rearranges.
    frame = _wds_tar.decode_frame(
        _wds_tar.get_mp4_path(meta["dataset_path"], topic), meta[topic]
    )[0]
    frame = _teleavatar._extract_stereo_view(
        _teleavatar._parse_image(frame), "left", rotate=False
    )
    return np.asarray(image_tools.resize_with_pad(frame, 224, 224))


@log_verification
def test_wds_ten_batches_match_tar_ground_truth(*, logger):
    """Prove 10 WDS batches carry the RIGHT samples, not merely right-shaped ones.

    Shape checks (test_wds_data_loader_batches) pass just as happily on a
    pipeline that decodes the wrong frame or misaligns state and action. This
    test builds an independent index straight from the .tar shards, then for
    every element of 10 batches:

      1. nearest-neighbour matches the state row back to its source sample
         (states are unique per sample, so the match identifies the sample; the
         runner-up distance is logged as the margin),
      2. recomputes that row's actions from the raw 62-dim action and compares --
         this is what catches state/action misalignment,
      3. re-decodes the mp4 frame the sample's own json names and compares the
         head camera pixel-wise, which catches a wrong frame index, a wrong
         camera, or a wrong stereo half,
      4. checks the padding tail (dims 14:32 / 16:32) is still zero.

    Finally it asserts all 10*batch_size elements are DISTINCT samples: a loader
    that re-served one buffer would satisfy every per-sample check above.

    Runs with skip_norm_stats=True, so what is compared is the pipeline's own
    output with Normalize disabled -- the state and action values are raw and
    the assertions do not depend on norm_stats matching the shards.
    """
    config = _require_wds_data()
    config = dataclasses.replace(config, batch_size=jax.device_count(), num_workers=0)
    data_config = config.data.create(config.assets_dirs, config.model)

    states, rows = _build_ground_truth(data_config.wds_data_dir)
    logger.info("[INDEX] %s samples from %s", len(rows), data_config.wds_data_dir)

    loader = _data_loader.create_data_loader(
        config, skip_norm_stats=True, num_batches=VERIFY_BATCHES
    )
    batches = list(loader)
    assert len(batches) == VERIFY_BATCHES

    seen_keys = []
    for b, (observation, actions) in enumerate(batches):
        state = np.asarray(observation.state)
        action = np.asarray(actions)
        base_image = np.asarray(observation.images["base_0_rgb"])

        assert np.isfinite(state).all() and np.isfinite(action).all(), f"batch {b} has NaN/Inf"
        assert np.all(state[:, 14:] == 0), f"batch {b}: state padding is not zero"
        assert np.all(action[:, :, 16:] == 0), f"batch {b}: action padding is not zero"

        # Without Normalize the state is the raw vector, so it should match a
        # ground-truth row EXACTLY -- the chain only slices and zero-pads it.
        for i in range(config.batch_size):
            distances = np.linalg.norm(states - state[i, :14], axis=1)
            j = int(np.argmin(distances))
            best = float(distances[j])
            runner_up = float(np.partition(distances, 1)[1])
            key, shard, meta, raw_action = rows[j]
            assert best == 0.0, (
                f"batch {b}[{i}]: no ground-truth sample matches this state exactly "
                f"(nearest distance {best:.3e}) -- the loader served a state that is "
                f"not in the shards, or something rescaled it"
            )
            # Uniqueness margin: the second-closest row must be plainly different,
            # or the identification above would not pin down a single sample.
            assert runner_up > VERIFY_STATE_MARGIN, (
                f"batch {b}[{i}]: ambiguous match to {key} "
                f"(best={best:.3e}, runner_up={runner_up:.3e})"
            )

            expected_action = _expected_actions(raw_action)
            action_err = float(np.abs(expected_action - action[i, :, :16]).max())
            assert action_err == 0.0, (
                f"batch {b}[{i}] ({key}): actions do not belong to the matched state "
                f"(max|diff|={action_err:.3e}) -- state and action are misaligned"
            )

            # Observation.from_dict already mapped uint8 -> float32 [-1, 1], so
            # measure the difference in uint8 LSBs to keep the tolerance readable.
            expected_image = _expected_base_image(meta)
            expected_float = expected_image.astype(np.float32) / 255.0 * 2.0 - 1.0
            diff = np.abs(base_image[i] - expected_float) * (255.0 / 2.0)
            diff_frac = float((diff > 0.5).mean())
            assert diff.max() <= VERIFY_IMAGE_ATOL and diff_frac <= VERIFY_IMAGE_MAX_DIFF_FRAC, (
                f"batch {b}[{i}] ({key}): base_0_rgb is not the frame this sample's json "
                f"names (max|diff|={diff.max():.2f} LSB, differing pixels={diff_frac:.3%})"
            )
            # A frozen/blank frame would pass a diff check against itself only if
            # the reference were equally blank; guard the reference too.
            assert expected_image.std() > 1.0, f"batch {b}[{i}]: reference frame is flat"

            logger.info(
                "[MATCH] batch=%s idx=%s key=%s shard=%s d=%.3e runner_up=%.3e "
                "action_err=%.3e img_max_diff=%.2fLSB img_diff_frac=%.4f",
                b, i, key, shard, best, runner_up, action_err, float(diff.max()), diff_frac,
            )
            seen_keys.append(key)

    assert len(set(seen_keys)) == len(seen_keys), (
        f"the loader repeated samples within {VERIFY_BATCHES} batches: "
        f"{len(seen_keys) - len(set(seen_keys))} duplicates"
    )
    logger.info("[UNIQUE] %s distinct samples over %s batches", len(set(seen_keys)), VERIFY_BATCHES)
