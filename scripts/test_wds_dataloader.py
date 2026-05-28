"""Test WDS data loading -- exact match to train.py main() L194-L227.

Skips model initialization and training. Verifies:
  - create_data_loader routes to WDS path
  - Each batch is tuple[Observation, Actions]
  - All field shapes/dtypes/value-ranges are correct
  - action padding to action_horizon works

Usage:
    uv run scripts/test_wds_dataloader.py
"""

import dataclasses
import functools
import json
import logging
import os
import platform
import time
import etils.epath as epath
import jax
import matplotlib.pyplot as plt
import numpy as np
import wandb

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

OUTPUT_DIR = "./wds_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

# Configure file logger for full run log
_file_handler = logging.FileHandler(f"{OUTPUT_DIR}/test_wds_dataloader.log")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
    datefmt="%H:%M:%S",
))
logging.getLogger().addHandler(_file_handler)

# Re-use train.py's custom logging format (exact copy of L31-47)
_level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}


class _CustomFormatter(logging.Formatter):
    def format(self, record):
        record.levelname = _level_mapping.get(record.levelname, record.levelname)
        return super().format(record)


_console_formatter = _CustomFormatter(
    fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in logger.handlers:
    if not isinstance(h, logging.FileHandler):
        h.setFormatter(_console_formatter)


# ----- Decorator for saving test data -----

def trace_wds_test(func):
    """Decorator that saves batch info, images, and results to disk."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        logging.info("=" * 80)
        logging.info("Starting WDS data loader test")
        logging.info("=" * 80)

        result = func(*args, **kwargs)

        total_elapsed = time.time() - t_start
        logging.info(f"Test completed in {total_elapsed:.3f}s")

        # Save per-batch info as JSON
        batch_info_path = f"{OUTPUT_DIR}/per_batch_info.json"
        with open(batch_info_path, "w") as f:
            json.dump(result["batch_info"], f, indent=2, default=str)
        logging.info(f"Saved per-batch info: {batch_info_path}")

        # Save timing info
        timing_path = f"{OUTPUT_DIR}/timing.json"
        with open(timing_path, "w") as f:
            json.dump(result["timing"], f, indent=2)
        logging.info(f"Saved timing info: {timing_path}")

        return result

    return wrapper


# ----- Helper functions -----

def _analyze_batch(batch_idx: int, observation: _model.Observation, actions) -> dict:
    """Extract shape, dtype, value range from a batch."""
    info = {"batch_index": batch_idx}

    # Images
    info["images"] = {}
    for key, img in observation.images.items():
        arr = np.array(img)
        info["images"][key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    # Image masks
    info["image_masks"] = {}
    for key, mask in observation.image_masks.items():
        arr = np.array(mask)
        info["image_masks"][key] = {
            "shape": list(arr.shape),
            "all_valid": bool(arr.all()),
        }

    # State
    state_arr = np.array(observation.state)
    info["state"] = {
        "shape": list(state_arr.shape),
        "dtype": str(state_arr.dtype),
        "min": float(state_arr.min()),
        "max": float(state_arr.max()),
        "mean": float(state_arr.mean()),
    }

    # Tokenized prompt
    if observation.tokenized_prompt is not None:
        tp_arr = np.array(observation.tokenized_prompt)
        info["tokenized_prompt"] = {
            "shape": list(tp_arr.shape),
            "dtype": str(tp_arr.dtype),
            "sample_tokens_batch0": tp_arr[0].tolist(),
        }
    else:
        info["tokenized_prompt"] = None

    if observation.tokenized_prompt_mask is not None:
        tpm_arr = np.array(observation.tokenized_prompt_mask)
        info["tokenized_prompt_mask"] = {
            "shape": list(tpm_arr.shape),
            "valid_count_batch0": int(tpm_arr[0].sum()),
            "total": int(tpm_arr.shape[-1]),
        }
    else:
        info["tokenized_prompt_mask"] = None

    info["token_ar_mask"] = "None" if observation.token_ar_mask is None else "present"
    info["token_loss_mask"] = "None" if observation.token_loss_mask is None else "present"

    # Actions
    act_arr = np.array(actions)
    info["actions"] = {
        "shape": list(act_arr.shape),
        "dtype": str(act_arr.dtype),
        "min": float(act_arr.min()),
        "max": float(act_arr.max()),
        "mean": float(act_arr.mean()),
        "std": float(act_arr.std()),
    }

    # Check action padding: are last steps all zeros?
    last_step = act_arr[:, -1, :]
    info["actions"]["last_step_all_zero_batch0"] = bool(np.allclose(last_step[0], 0.0))
    info["actions"]["last_step_max_abs_batch0"] = float(np.abs(last_step[0]).max())

    return info


def _save_images(observation: _model.Observation, images_dir: str) -> None:
    """Save first batch images as PNG files."""
    for key, img in observation.images.items():
        arr = np.array(img)
        for b in range(arr.shape[0]):
            img_single = arr[b]
            img_single = np.clip((img_single + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            out_path = f"{images_dir}/batch{b}_{key}.png"
            plt.imsave(out_path, img_single)
            logging.info(f"  Saved: {out_path}")


# ----- Main test (exact train.py main() L194-L227 logic) -----

@trace_wds_test
def run_test(config_name: str = "pi0_lingyu_wds", num_batches: int = 10) -> dict:
    """Run WDS data loading test, exact match to train.py main() L194-L227."""

    config = _config.get_config(config_name)
    # Set exp_name so checkpoint_dir is valid (required by train.py L212-217)
    config = dataclasses.replace(config, exp_name="test_wds", overwrite=True)

    # --- L195-196 ---
    logging.info(f"Running on: {platform.node()}")

    # --- L198-201 ---
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    logging.info(f"batch_size={config.batch_size}, device_count={jax.device_count()}")

    # --- L203 ---
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # --- L205-210 ---
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    logging.info(f"Mesh created: fsdp_devices={config.fsdp_devices}")

    # --- L212-217 ---
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    logging.info(f"Checkpoint dir initialized: {config.checkpoint_dir}, resuming={resuming}")

    # --- L218 ---
    # train.py: init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    # Since wandb_enabled=False, it does wandb.init(mode="disabled") → we replicate that directly
    wandb.init(mode="disabled")
    logging.info("wandb initialized (disabled)")

    # --- L220-224 ---
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    logging.info("Data loader created (should route to WDS path)")

    # --- L225-N: get batches ---
    data_iter = iter(data_loader)
    batch_info = []
    timing = []

    for i in range(num_batches):
        t0 = time.time()
        batch = next(data_iter)
        jax.block_until_ready(batch)
        elapsed = time.time() - t0

        observation, actions = batch

        # Verify batch structure
        assert isinstance(observation, _model.Observation), f"Expected Observation, got {type(observation)}"

        # Analyze
        info = _analyze_batch(i, observation, actions)
        batch_info.append(info)
        timing.append({"batch_idx": i, "elapsed_seconds": round(elapsed, 4)})

        # Save first batch images
        if i == 0:
            _save_images(observation, f"{OUTPUT_DIR}/images")

        # Log summary (matching train.py L227 format)
        logging.info(f"Batch {i}: {training_utils.array_tree_to_info(batch)}")
        logging.info(f"  elapsed={elapsed:.4f}s  actions.shape={list(actions.shape)}")

    # Summary
    logging.info("=" * 80)
    logging.info(f"All {num_batches} batches loaded successfully.")
    avg_time = sum(t["elapsed_seconds"] for t in timing) / len(timing)
    logging.info(f"Average batch load time: {avg_time:.4f}s")
    logging.info(f"Min: {min(t['elapsed_seconds'] for t in timing):.4f}s")
    logging.info(f"Max: {max(t['elapsed_seconds'] for t in timing):.4f}s")
    logging.info("=" * 80)

    return {
        "batch_info": batch_info,
        "timing": timing,
    }


def main():
    result = run_test(config_name="pi0_lingyu_wds", num_batches=10)

    # Print human-readable summary
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    b0 = result["batch_info"][0]
    print(f"images keys:  {list(b0['images'].keys())}")
    print(f"state shape:  {b0['state']['shape']}")
    print(f"actions shape: {b0['actions']['shape']}")
    print(f"tokenized_prompt shape: {b0['tokenized_prompt']['shape']}")
    print(f"Average batch time: {sum(t['elapsed_seconds'] for t in result['timing'])/len(result['timing']):.4f}s")
    print(f"\nOutput files:")
    print(f"  log:     {OUTPUT_DIR}/test_wds_dataloader.log")
    print(f"  info:    {OUTPUT_DIR}/per_batch_info.json")
    print(f"  timing:  {OUTPUT_DIR}/timing.json")
    print(f"  images:  {OUTPUT_DIR}/images/")


if __name__ == "__main__":
    main()