import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger(__name__)


def make_teleavatar_example() -> dict:
    """Creates a random input example for the Teleavatar policy (v2 robot formats)."""
    return {
        "observation/state": np.random.rand(62),
        "observation/images/left_color": np.random.randint(256, size=(800, 2560, 3), dtype=np.uint8),
        "observation/images/right_color": np.random.randint(256, size=(800, 2560, 3), dtype=np.uint8),
        "observation/images/head_camera": np.random.randint(256, size=(1920, 3840, 3), dtype=np.uint8),
        "actions": np.random.rand(62),
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


def _extract_stereo_view(image: np.ndarray, side: str, *, rotate: bool = False) -> np.ndarray:
    """Crop one eye from a side-by-side stereo frame, optionally rotating 180°
    first (rotation applies to the full stereo frame, before the crop).

    The crop is gated on ``width >= 2 * height`` so an already-cropped frame
    (e.g. an inference path that pre-crops) is left untouched: raw stereo
    frames are at least 2:1 (v2 head 3840×1920 → 2:1, v2 left/right
    2560×800 → 3.2:1) while cropped eyes fall below it (1:1 and 1.6:1
    respectively). The v1 robot's mono 848×480 left/right images (~1.77:1)
    also pass through unchanged.
    """
    height, width = image.shape[:2]
    if width >= 2 * height:
        if rotate:
            image = np.rot90(image, k=2)
        half = width // 2
        image = image[:, :half, :] if side == "left" else image[:, half:, :]
    return image


# v2 platform gripper force control: the controller takes a float32 trigger
# value in [0, 1] and maps it to a gripper effort (Nm) with a piecewise-linear,
# non-proportional curve, identical for both arms (unlike v1, which was
# asymmetric). Sign convention: positive effort = opening, negative = closing;
# zero crossing at trigger = 0.10.
#   trigger <  0.10:  effort = +2.0 * (1 - trigger / 0.10)     (opening: +2.0 → 0)
#   trigger >= 0.10:  effort = -1.6 * (trigger - 0.10) / 0.90  (closing:  0 → -1.6)


def _gripper_effort_to_trigger(effort: np.ndarray) -> np.ndarray:
    """Convert gripper effort (Nm) to the normalized [0, 1] trigger value.

    Exact inverse of the platform's trigger→effort curve (v2, both arms).
    Deliberately unclipped so the effort→trigger→effort round trip is exact;
    measured efforts slightly outside [-1.6, +2.0] map slightly outside [0, 1]
    and are handled by the dataset norm stats.
    """
    return np.where(effort > 0, 0.10 * (1.0 - effort / 2.0), 0.10 - effort * 0.90 / 1.6)


def _gripper_trigger_to_effort(trigger: np.ndarray) -> np.ndarray:
    """Convert a normalized [0, 1] trigger value to gripper effort (Nm)."""
    return np.where(trigger < 0.10, 2.0 * (1.0 - trigger / 0.10), -1.6 * (trigger - 0.10) / 0.90)


@dataclasses.dataclass(frozen=True)
class TeleavatarInputs(transforms.DataTransformFn):
    """
    Converts inputs to the model format for Teleavatar robot.

    **Input format (observation/state from LeRobot dataset):**
    Layout: [positions(16), velocities(16), efforts(16)] plus, on the v2
    robot, EE poses(14) appended at indices 48-61 (62-dim total; v1 datasets
    are 48-dim). The indices used below are identical in both layouts:
    - Indices 0-15: Joint positions (7 left arm, 1 left gripper, 7 right arm, 1 right gripper)
    - Indices 16-31: Joint velocities (same layout)
    - Indices 32-47: Joint efforts (same layout)
    - Indices 48-61 (v2 only): EE position(3) + orientation quaternion(4) per arm — unused

    **Model state format (14-dim):**
    We extract: [left_arm_pos(7), right_arm_pos(7)]
    - Indices 0-6: Left arm positions (from input[0:7])
    - Indices 7-13: Right arm positions (from input[8:15])

    **Cameras (v2 robot):** all three streams are side-by-side stereo; one eye
    is cropped out per camera (see __call__). On the v1 robot only the head
    camera is stereo — the left/right images are mono and the shape guard in
    _extract_stereo_view leaves them untouched.
    """
    model_type: _model.ModelType
    # Whether to rotate 180° before cropping the head frame. Property of the
    # source data, not of train/inference — set it identically for both. The
    # width guard inside _extract_stereo_view already no-ops for
    # already-cropped frames, so this only matters for raw stereo.
    #   True  → upside-down raw stereo (camera mounted upside-down, e.g. the
    #           v1 officially released robot — its training data is upside-down too)
    #   False → frame already right-side-up before this transform (v2 robot)
    rotate_head_camera: bool = False

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W) during training, but runtime sends uint8 (H,W,C)
        left_color = _parse_image(data["observation/images/left_color"])
        right_color = _parse_image(data["observation/images/right_color"])
        head_color = _parse_image(data["observation/images/head_camera"])
        # Crop one eye from each side-by-side stereo frame. The head keeps its
        # left eye (rotated 180° first iff the configured source orientation
        # says so). The left/right cameras keep their INNER eye — right eye of
        # the left camera, left eye of the right camera — so both look at the
        # middle of the desktop workspace. The width guard inside
        # _extract_stereo_view makes all three no-ops when frames already
        # arrive cropped (by ros2_interface) or mono (v1 robot left/right).
        head_color = _extract_stereo_view(
            head_color, "left", rotate=self.rotate_head_camera
        )
        left_color = _extract_stereo_view(left_color, "right")
        right_color = _extract_stereo_view(right_color, "left")

        # Extract 14-dim state from the observation vector (62-dim on v2,
        # 48-dim on v1 — position indices are identical in both layouts).
        # Input layout: [positions(0-15), velocities(16-31), efforts(32-47), (v2) ee_pose(48-61)]
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

        # Extract 16-dim actions from the dataset action vector during training
        # (62-dim on v2, 48-dim on v1 — the indices used are identical).
        # Actions are only available during training, not during inference
        if "action" in data:
            # data["action"] has shape [action_horizon, 62] (v2) or [action_horizon, 48] (v1)
            # Layout: [positions(0-15), velocities(16-31), efforts(32-47), (v2) ee_pose(48-61)]
            action_data = data["action"]

            # Extract 16 dimensions: joint positions (14) + gripper efforts (2)
            # Note: State only uses 14 dims (joint positions), but actions include gripper efforts
            selected_actions = np.concatenate([
                action_data[:, 0:7],    # Left arm positions
                action_data[:, 39:40],  # Left gripper effort (index 39 = 32+7)
                action_data[:, 8:15],   # Right arm positions
                action_data[:, 47:48],  # Right gripper effort (index 47 = 32+15)
            ], axis=1)  # Concatenate along action dimension

            # Convert raw gripper efforts (Nm) to the normalized [0, 1]
            # trigger range (same curve for both arms on v2). This runs before
            # the dataset norm-stats normalization in model_transforms, so norm
            # stats must be recomputed after changing this mapping
            # (scripts/compute_norm_stats.py). Action layout is
            # [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)].
            selected_actions[:, 7] = _gripper_effort_to_trigger(selected_actions[:, 7])
            selected_actions[:, 15] = _gripper_effort_to_trigger(selected_actions[:, 15])

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

        # Convert normalized [0, 1] trigger values back to effort (Nm) for
        # robot execution (inverse of the effort→trigger map in
        # TeleavatarInputs; same curve for both arms on v2).
        actions[:, 7] = _gripper_trigger_to_effort(actions[:, 7])
        actions[:, 15] = _gripper_trigger_to_effort(actions[:, 15])
        return {"actions": actions}