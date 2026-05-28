"""Trace the full data loading pipeline without any model loading or training.

Usage:
    python scripts/trace_data_pipeline.py
"""
import dataclasses
import functools
import json
import logging
import os
import pickle
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./data_pipeline_trace_output"


def save_data_decorator(output_dir: str = OUTPUT_DIR):
    """Decorator that saves data pipeline output for inspection."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(output_dir, exist_ok=True)

            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Function '{func.__name__}' completed in {elapsed:.4f}s")

            # Save full result as pickle for detailed inspection
            pickle_path = os.path.join(output_dir, f"{func.__name__}_result.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"Saved full result to {pickle_path}")

            return result

        return wrapper

    return decorator


@save_data_decorator()
def run_data_pipeline(
    config_name: str = "debug",
    num_batches: int = 2,
) -> dict[str, Any]:
    """Run the full data loading pipeline and return detailed info about each batch."""

    config = _config.get_config(config_name)
    config = dataclasses.replace(config, batch_size=2)

    logger.info("=" * 80)
    logger.info("STEP 1: Config details")
    logger.info("=" * 80)
    logger.info(f"Config name: {config.name}")
    logger.info(f"Data config type: {type(config.data).__name__}")
    logger.info(f"Model type: {config.model.model_type}")
    logger.info(f"action_dim: {config.model.action_dim}")
    logger.info(f"action_horizon: {config.model.action_horizon}")
    logger.info(f"max_token_len: {config.model.max_token_len}")
    logger.info(f"batch_size: {config.batch_size}")

    logger.info("=" * 80)
    logger.info("STEP 2: Creating data loader")
    logger.info("=" * 80)
    data_loader = _data_loader.create_data_loader(
        config,
        skip_norm_stats=True,
        num_batches=num_batches,
        shuffle=False,
    )
    logger.info(f"DataLoader type: {type(data_loader).__name__}")
    logger.info(f"DataConfig: {data_loader.data_config()}")

    batches = []
    for batch_idx, batch in enumerate(data_loader):
        observation, actions = batch

        logger.info("=" * 80)
        logger.info(f"STEP 3: Batch {batch_idx} analysis")
        logger.info("=" * 80)

        # Analyze Observation
        obs_info = _analyze_observation(observation)
        act_info = _analyze_actions(actions)

        batch_info = {
            "batch_index": batch_idx,
            "observation": obs_info,
            "actions": act_info,
        }
        batches.append(batch_info)

        # Save images from first batch
        if batch_idx == 0:
            _save_observation_images(observation, output_dir=OUTPUT_DIR)

    # Generate summary
    summary = {
        "config_name": config_name,
        "model_type": str(config.model.model_type),
        "action_dim": config.model.action_dim,
        "action_horizon": config.model.action_horizon,
        "max_token_len": config.model.max_token_len,
        "batch_size": config.batch_size,
        "num_batches_collected": len(batches),
        "batches": batches,
        "data_flow_diagram": {
            "step1_LeRobot_dataset": "LeRobotDataset reads (observation.images, observation.state, action) from disk",
            "step2_repack_transforms": "RepackTransform remaps keys: e.g. 'observation.images.top' -> 'images.cam_high'",
            "step3_data_transforms": "Robot-specific transforms (e.g. AlohaInputs, DroidInputs, TeleavatarInputs) -- may convert state/action spaces",
            "step4_normalize": "Normalize(state, actions, image) using precomputed norm_stats (z-score or quantile)",
            "step5_model_transforms": "Model-specific transforms: InjectDefaultPrompt -> ResizeImages -> TokenizePrompt -> PadStatesAndActions",
            "step6_torch_dataloader": "TorchDataLoader applies batching, shuffling, sharding. Collate_fn stacks samples into batches.",
            "step7_Observation_from_dict": "Observation.from_dict() converts dict to structured Observation object, uint8 images -> float32 [-1,1]",
            "step8_final_yield": "DataLoaderImpl yields tuple(Observation, Actions) where Actions = batch['actions']",
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved pipeline summary to {summary_path}")

    return summary


def _analyze_observation(obs: _model.Observation) -> dict[str, Any]:
    """Analyze the observation and return a structured description."""
    info = {}

    # Images
    info["images"] = {}
    for key, img in obs.images.items():
        info["images"][key] = {
            "shape": list(img.shape),
            "dtype": str(img.dtype),
            "min": float(img.min()),
            "max": float(img.max()),
            "mean": float(img.mean()),
        }
        logger.info(f"  images['{key}']: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.4f},{img.max():.4f}]")

    # Image masks
    info["image_masks"] = {}
    for key, mask in obs.image_masks.items():
        info["image_masks"][key] = {
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "all_true": bool(mask.all()),
        }
        logger.info(f"  image_masks['{key}']: shape={mask.shape}, all_true={mask.all()}")

    # State
    info["state"] = {
        "shape": list(obs.state.shape),
        "dtype": str(obs.state.dtype),
        "min": float(obs.state.min()),
        "max": float(obs.state.max()),
        "mean": float(obs.state.mean()),
    }
    logger.info(f"  state: shape={obs.state.shape}, dtype={obs.state.dtype}, range=[{obs.state.min():.4f},{obs.state.max():.4f}]")

    # Tokenized prompt
    if obs.tokenized_prompt is not None:
        info["tokenized_prompt"] = {
            "shape": list(obs.tokenized_prompt.shape),
            "dtype": str(obs.tokenized_prompt.dtype),
        }
        logger.info(f"  tokenized_prompt: shape={obs.tokenized_prompt.shape}")
    else:
        info["tokenized_prompt"] = None
        logger.info("  tokenized_prompt: None")

    if obs.tokenized_prompt_mask is not None:
        info["tokenized_prompt_mask"] = {
            "shape": list(obs.tokenized_prompt_mask.shape),
            "dtype": str(obs.tokenized_prompt_mask.dtype),
        }
        logger.info(f"  tokenized_prompt_mask: shape={obs.tokenized_prompt_mask.shape}")
    else:
        info["tokenized_prompt_mask"] = None

    # FAST model specific
    if obs.token_ar_mask is not None:
        info["token_ar_mask"] = {
            "shape": list(obs.token_ar_mask.shape),
        }
        logger.info(f"  token_ar_mask: shape={obs.token_ar_mask.shape}")
    else:
        info["token_ar_mask"] = None

    if obs.token_loss_mask is not None:
        info["token_loss_mask"] = {
            "shape": list(obs.token_loss_mask.shape),
        }
        logger.info(f"  token_loss_mask: shape={obs.token_loss_mask.shape}")
    else:
        info["token_loss_mask"] = None

    return info


def _analyze_actions(actions: _model.Actions) -> dict[str, Any]:
    """Analyze actions tensor."""
    # actions is at.Float[ArrayT, "*b ah ad"]
    info = {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "min": float(actions.min()),
        "max": float(actions.max()),
        "mean": float(actions.mean()),
        "interpretation": {
            "batch_size": actions.shape[0],
            "action_horizon": actions.shape[1],
            "action_dim": actions.shape[-1],
        },
    }
    logger.info(f"  actions: shape={actions.shape}, dtype={actions.dtype}")
    logger.info(f"    -> batch_size={actions.shape[0]}, action_horizon={actions.shape[1]}, action_dim={actions.shape[-1]}")
    return info


def _save_observation_images(obs: _model.Observation, output_dir: str) -> None:
    """Save the observation images as PNG files."""
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for key, img in obs.images.items():
        # img is [batch, H, W, 3] in [-1, 1] float32
        batch_size = img.shape[0]
        for b in range(batch_size):
            img_single = np.array(img[b])
            # Convert from [-1, 1] to [0, 255] uint8
            img_single = np.clip((img_single + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            img_path = os.path.join(images_dir, f"{key}_batch{b}.png")
            plt.imsave(img_path, img_single)
            logger.info(f"  Saved image: {img_path}")


def main():
    summary = run_data_pipeline(config_name="debug", num_batches=2)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY: Data handed to the training step")
    print("=" * 80)
    print(f"""
The data loader yields `tuple[Observation, Actions]` where:

Observation has these fields:
  - images:         dict[str, Float[*b, H, W, 3]]    — camera images in [-1, 1] float32
  - image_masks:    dict[str, Bool[*b]]               — per-image valid/invalid masks
  - state:          Float[*b, s]                      — low-dimensional robot state (joints, gripper, etc.)
  - tokenized_prompt:       Int[*b, l] | None         — tokenized language instruction (PI0/PI05)
  - tokenized_prompt_mask:  Bool[*b, l] | None        — prompt mask
  - token_ar_mask:          Int[*b, l] | None         — autoregressive mask (PI0-FAST only)
  - token_loss_mask:        Bool[*b, l] | None        — token loss mask (PI0-FAST only)

Actions:
  - Float[*b, ah, ad]                                — action sequence: [batch, action_horizon, action_dim]

where:
  *b = batch_size
  H, W = image height, width (typically 224x224 after transforms)
  s = state dimension
  l = token sequence length (= max_token_len)
  ah = action_horizon
  ad = action_dim
""")

    print(f"Detailed results saved to: {OUTPUT_DIR}/")
    print(f"  - {OUTPUT_DIR}/pipeline_summary.json")
    print(f"  - {OUTPUT_DIR}/run_data_pipeline_result.pkl")
    print(f"  - {OUTPUT_DIR}/images/")


if __name__ == "__main__":
    main()