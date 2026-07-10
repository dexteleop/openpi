"""Policy transforms for the V1 (officially released) Teleavatar robot.

Frozen snapshot kept for v1 users: upside-down stereo head camera
(rotate_head_camera=True), mono 848x480 left/right cameras, 48-dim
state/action, and the v1 asymmetric gripper effort mapping. Active
development targets the v2 robot in teleavatar_policy.py.
"""

import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger(__name__)


def make_teleavatar_example() -> dict:
    """Creates a random input example for the Teleavatar policy."""
    return {
        "observation/state": np.random.rand(48),
        "observation/images/left_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/right_color": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/images/head_camera": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "actions": np.random.rand(48),
        "prompt": "pick a cube and place it on another cube",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format following LeRobot conventions."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _extract_left_head_view(image: np.ndarray, *, rotate: bool = False) -> np.ndarray:
    """Crop the left eye from a side-by-side stereo head image, optionally
    rotating 180° first.

    The crop is gated on the ``width == 2 * height`` shape check so an
    already-cropped ``(H, H, 3)`` frame (e.g. an inference path that
    pre-crops) is left untouched.
    """
    height, width = image.shape[:2]
    if width == 2 * height:
        if rotate:
            image = np.rot90(image, k=2)
        image = image[:, :height, :]
    return image


def _left_gripper_effort_to_normalized(effort: np.ndarray) -> np.ndarray:
    """Convert left gripper effort to normalized [0, 1] controller range."""
    return np.where(effort > 0, 0.5 - effort / 7.0, 0.5 - effort)


def _right_gripper_effort_to_normalized(effort: np.ndarray) -> np.ndarray:
    """Convert right gripper effort to normalized [0, 1] controller range."""
    return np.where(effort < 0, effort / 7.0 + 0.5, effort + 0.5)


def _left_gripper_normalized_to_effort(data: np.ndarray) -> np.ndarray:
    """Convert normalized [0, 1] controller value to left gripper effort."""
    grip = -(data - 0.5)
    return np.where(grip > 0, grip * 7.0, grip)


def _right_gripper_normalized_to_effort(data: np.ndarray) -> np.ndarray:
    """Convert normalized [0, 1] controller value to right gripper effort."""
    grip = data - 0.5
    return np.where(grip < 0, grip * 7.0, grip)


@dataclasses.dataclass(frozen=True)
class TeleavatarInputs(transforms.DataTransformFn):
    """
    Converts inputs to the model format for Teleavatar robot.

    **Input format (48-dim observation/state from LeRobot dataset):**
    Layout: [positions(16), velocities(16), efforts(16)]
    - Indices 0-15: Joint positions (7 left arm, 1 left gripper, 7 right arm, 1 right gripper)
    - Indices 16-31: Joint velocities (same layout)
    - Indices 32-47: Joint efforts (same layout)

    **Model state format (14-dim):**
    We extract: [left_arm_pos(7), right_arm_pos(7)]
    - Indices 0-6: Left arm positions (from input[0:7])
    - Indices 7-13: Right arm positions (from input[8:15])

    This matches the data format in convert_teleavatar_data_to_lerobot.py
    """
    model_type: _model.ModelType
    # Whether to rotate 180° before cropping the head frame. Property of the
    # source data, not of train/inference — set it identically for both. The
    # 2:1-width guard inside _extract_left_head_view already no-ops for
    # already-cropped (square) frames, so this only matters for raw stereo.
    #   True  → upside-down raw stereo (camera mounted upside-down, e.g. the
    #           officially released robot — its training data is upside-down too)
    #   False → frame already right-side-up before this transform
    rotate_head_camera: bool = False

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W) during training, but runtime sends uint8 (H,W,C)
        left_color = _parse_image(data["observation/images/left_color"])
        right_color = _parse_image(data["observation/images/right_color"])
        head_color = _parse_image(data["observation/images/head_camera"])
        # Crop the left eye from the side-by-side stereo head frame, rotating
        # 180° iff the configured source orientation says so. The width-check
        # guard inside _extract_left_head_view makes this a no-op when the head
        # frame already arrives square (cropped by ros2_interface).
        head_color = _extract_left_head_view(
            head_color, rotate=self.rotate_head_camera
        )

        # Extract 14-dim state from 48-dim observation
        # Input layout: [positions(0-15), velocities(16-31), efforts(32-47)]
        state_14d = np.concatenate([
            data["observation/state"][0:7],    # Left arm positions (indices 0-6)
            data["observation/state"][8:15],   # Right arm positions (indices 8-14)
        ], axis=0)

        # Create inputs dict. Do not change the keys in the dict below.
        # Pi0 models support three image inputs: one third-person view and two wrist views.
        # Map teleavatar cameras to the expected model inputs.
        inputs = {
            "state": state_14d,
            "image": {
                "base_0_rgb": head_color,       
                "left_wrist_0_rgb": left_color,  
                "right_wrist_0_rgb": right_color,  
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.True_,
            },
        }

        # Extract 16-dim actions from 48-dim during training
        # Actions are only available during training, not during inference
        if "action" in data:
            # data["action"] has shape [action_horizon, 48]
            # Layout: [positions(0-15), velocities(16-31), efforts(32-47)]
            action_data = data["action"]

            # Extract 16 dimensions: joint positions (14) + gripper efforts (2)
            # Note: State only uses 14 dims (joint positions), but actions include gripper efforts
            selected_actions = np.concatenate([
                action_data[:, 0:7],    # Left arm positions
                action_data[:, 39:40],  # Left gripper effort (index 39 = 32+7)
                action_data[:, 8:15],   # Right arm positions
                action_data[:, 47:48],  # Right gripper effort (index 47 = 32+15)
            ], axis=1)  # Concatenate along action dimension

            # Convert raw gripper efforts to the normalized [0, 1] controller
            # range. This runs before the dataset norm-stats normalization in
            # model_transforms, so norm stats must be recomputed after enabling
            # this (scripts/compute_norm_stats.py). Action layout is
            # [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)].
            selected_actions[:, 7] = _left_gripper_effort_to_normalized(selected_actions[:, 7])
            selected_actions[:, 15] = _right_gripper_effort_to_normalized(selected_actions[:, 15])

            inputs["actions"] = selected_actions

        # Pass the prompt (aka language instruction) to the model. During
        # training this should be filled by PromptFromLeRobotTask + the
        # "prompt": "prompt" entry in the repack structure (see
        # LeRobotTeleavatarDataConfig.create). The fallback below only kicks
        # in for inference callers that don't supply a prompt; if it ever
        # triggers during training, every sample shares one fixed string and
        # the language channel is dead — make that visible.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            inputs["prompt"] = "perform the manipulation task"
            logger.warning(
                "TeleavatarInputs: no 'prompt' in sample; using hardcoded "
                "fallback. Expected during inference without a client prompt, "
                "but indicates a training-pipeline bug if seen on training data."
            )

        return inputs


@dataclasses.dataclass(frozen=True)
class TeleavatarOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is
    used for inference only.

    For teleavatar, we return 16 actions:
    - Joint positions for joints 1-7 for both arms (14 values)
    - Joint efforts for left and right grippers (2 values)

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 actions for teleavatar.
        # Since the model may output more dimensions due to padding, we extract just what we need.
        # For your own dataset, replace `16` with the action dimension of your dataset.
        actions = np.asarray(data["actions"][:, :16])

        # Convert normalized [0, 1] gripper values back to effort for robot
        # execution (inverse of the effort->normalized map in TeleavatarInputs).
        actions[:, 7] = _left_gripper_normalized_to_effort(actions[:, 7])
        actions[:, 15] = _right_gripper_normalized_to_effort(actions[:, 15])
        return {"actions": actions}