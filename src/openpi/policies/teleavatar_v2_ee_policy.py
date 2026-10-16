"""End-effector transforms for the TeleAvatar V2 robot.

The raw TeleAvatar V2 LeRobot action/state layout is kept unchanged so the
existing rosbag converter can be reused.  This module projects both arms into
a compact end-effector representation:

    [left EE (position 3, rotation-6D 6, gripper trigger 1),
     right EE (position 3, rotation-6D 6, gripper trigger 1)]

The rotation-6D convention intentionally matches the UMI deployment code in
this workspace: the first two *rows* of a rotation matrix are flattened.
The measured gripper joint position is converted from openness (1=open,
0=closed) to the normalized trigger convention used by the robot API
(0=open, 1=closed).  Action gripper effort is converted to the same convention.
Actions are encoded as waypoints relative to the current end-effector pose;
``TeleavatarEEOutputs`` composes both arms back to absolute poses for deployment.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.teleavatar_v2_policy import _extract_stereo_view
from openpi.policies.teleavatar_v2_policy import _gripper_effort_to_trigger
from openpi.policies.teleavatar_v2_policy import _parse_image

EE_ARM_ACTION_DIM = 10
EE_ACTION_DIM = 2 * EE_ARM_ACTION_DIM
RAW_EE_OFFSET = {"left": 48, "right": 55}
# The state contains measured joint positions at these indices.  The action
# contains gripper effort at the corresponding effort indices below.
RAW_GRIPPER_POSITION_INDEX = {"left": 7, "right": 15}
RAW_GRIPPER_EFFORT_INDEX = {"left": 39, "right": 47}


def _as_float_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _gripper_position_to_trigger(position: np.ndarray) -> np.ndarray:
    """Convert measured normalized openness to the robot API trigger convention."""
    position = _as_float_array(position, name="gripper position")
    return 1.0 - np.clip(position, 0.0, 1.0)


def _pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, qx, qy, qz, qw]`` to homogeneous matrices."""
    pose = _as_float_array(pose, name="pose")
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected pose[..., 7], got {pose.shape}")
    quaternion = pose[..., 3:7]
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Pose contains a zero-length quaternion")
    quaternion = quaternion / norm
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    matrix = np.zeros((*pose.shape[:-1], 4, 4), dtype=np.float32)
    # Quaternion convention is xyzw, matching ROS geometry_msgs.
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    matrix[..., :3, 3] = pose[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Flatten the first two rotation-matrix rows, matching UMI's format."""
    return np.asarray(matrix[..., :2, :], dtype=np.float32).reshape((*matrix.shape[:-2], 6))


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Recover an orthonormal rotation matrix from the UMI row-wise 6D form."""
    rot6d = _as_float_array(rot6d, name="rotation-6D")
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation-6D[..., 6], got {rot6d.shape}")
    row_1 = rot6d[..., :3]
    row_2 = rot6d[..., 3:]
    row_1 = row_1 / np.maximum(np.linalg.norm(row_1, axis=-1, keepdims=True), 1e-7)
    row_2 = row_2 - np.sum(row_1 * row_2, axis=-1, keepdims=True) * row_1
    row_2 = row_2 / np.maximum(np.linalg.norm(row_2, axis=-1, keepdims=True), 1e-7)
    row_3 = np.cross(row_1, row_2, axis=-1)
    return np.stack((row_1, row_2, row_3), axis=-2).astype(np.float32)


def _matrix_to_pose10(matrix: np.ndarray) -> np.ndarray:
    pose = np.zeros((*matrix.shape[:-2], 9), dtype=np.float32)
    pose[..., :3] = matrix[..., :3, 3]
    pose[..., 3:] = _matrix_to_rot6d(matrix[..., :3, :3])
    return pose


def _pose10_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = _as_float_array(pose, name="pose10")
    if pose.shape[-1] != 9:
        raise ValueError(f"Expected pose10[..., 9], got {pose.shape}")
    matrix = np.zeros((*pose.shape[:-1], 4, 4), dtype=np.float32)
    matrix[..., :3, :3] = _rot6d_to_matrix(pose[..., 3:])
    matrix[..., :3, 3] = pose[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


@dataclasses.dataclass(frozen=True)
class TeleavatarEEInputs(transforms.DataTransformFn):
    """Map raw TeleAvatar V2 records to a bimanual 20D EE representation."""

    model_type: _model.ModelType
    rotate_head_camera: bool = False

    def __call__(self, data: dict) -> dict:
        left_color = _extract_stereo_view(_parse_image(data["observation/images/left_color"]), "right")
        right_color = _extract_stereo_view(_parse_image(data["observation/images/right_color"]), "left")
        head_color = _extract_stereo_view(
            _parse_image(data["observation/images/head_camera"]), "left", rotate=self.rotate_head_camera
        )

        raw_state = _as_float_array(data["observation/state"], name="observation/state")
        if raw_state.shape[-1] < 62:
            raise ValueError("TeleAvatar V2 EE training requires the 62D state with EE poses")
        state_parts = []
        state_matrices = {}
        for arm in ("left", "right"):
            state_matrix = _pose7_to_matrix(raw_state[RAW_EE_OFFSET[arm] : RAW_EE_OFFSET[arm] + 7])
            state_matrices[arm] = state_matrix
            state_pose = _matrix_to_pose10(state_matrix)
            # Joint-state position is measured openness (1=open, 0=closed),
            # while the API command is a trigger (0=open, 1=closed).
            state_gripper = _gripper_position_to_trigger(raw_state[RAW_GRIPPER_POSITION_INDEX[arm]])
            state_parts.append(np.concatenate((state_pose, np.atleast_1d(state_gripper)), axis=-1))
        state = np.concatenate(state_parts, axis=-1).astype(np.float32)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_color,
                "left_wrist_0_rgb": left_color,
                "right_wrist_0_rgb": right_color,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "action" in data:
            raw_action = _as_float_array(data["action"], name="action")
            if raw_action.shape[-1] < 62:
                raise ValueError("TeleAvatar V2 EE training requires the 62D action with EE poses")
            action_parts = []
            for arm in ("left", "right"):
                target_matrix = _pose7_to_matrix(raw_action[..., RAW_EE_OFFSET[arm] : RAW_EE_OFFSET[arm] + 7])
                current_inv = np.linalg.inv(state_matrices[arm])
                relative_matrix = current_inv @ target_matrix
                relative_pose = _matrix_to_pose10(relative_matrix)
                trigger = np.atleast_1d(
                    np.clip(_gripper_effort_to_trigger(raw_action[..., RAW_GRIPPER_EFFORT_INDEX[arm]]), 0.0, 1.0)
                )
                action_parts.append(np.concatenate((relative_pose, trigger[..., None]), axis=-1))
            inputs["actions"] = np.concatenate(action_parts, axis=-1).astype(np.float32)

        inputs["prompt"] = data.get("prompt", "perform the manipulation task")
        return inputs

    def rtc_action_anchor(self, data: dict) -> np.ndarray:
        """Return the absolute EE frames used to encode the relative actions.

        RTC needs this because the relative pose action is expressed in the observation
        frame. Unlike joint deltas, changing that frame is an SE(3) composition rather than
        an elementwise subtraction.
        """
        raw_state = _as_float_array(data["observation/state"], name="observation/state")
        if raw_state.ndim != 1 or raw_state.shape[-1] < 62:
            raise ValueError("TeleAvatar V2 EE RTC anchoring requires a 62D unbatched state")
        return np.stack(
            [
                _pose7_to_matrix(raw_state[RAW_EE_OFFSET[arm] : RAW_EE_OFFSET[arm] + 7])
                for arm in ("left", "right")
            ],
            axis=0,
        ).astype(np.float32)

    def rtc_reanchor(
        self,
        actions: np.ndarray,
        previous_anchor: np.ndarray,
        current_anchor: np.ndarray,
    ) -> np.ndarray:
        """Move relative EE waypoints from the previous observation frame to the current one."""
        values = _as_float_array(actions, name="RTC previous actions").copy()
        if values.ndim != 2 or values.shape[-1] < EE_ACTION_DIM:
            raise ValueError(f"Expected RTC actions with shape (N, >= {EE_ACTION_DIM}), got {values.shape}")
        previous = _as_float_array(previous_anchor, name="RTC previous anchor")
        current = _as_float_array(current_anchor, name="RTC current anchor")
        if previous.shape != (2, 4, 4) or current.shape != (2, 4, 4):
            raise ValueError(
                f"Expected two 4x4 RTC anchors, got previous={previous.shape}, current={current.shape}"
            )

        for arm_index, start in enumerate((0, EE_ARM_ACTION_DIM)):
            frame_change = np.linalg.inv(current[arm_index]) @ previous[arm_index]
            relative = _pose10_to_matrix(values[:, start : start + 9])
            values[:, start : start + 9] = _matrix_to_pose10(frame_change @ relative)
        return values


@dataclasses.dataclass(frozen=True)
class TeleavatarEEOutputs(transforms.DataTransformFn):
    """Convert relative bimanual 20D waypoints back to absolute EE actions."""

    def __call__(self, data: dict) -> dict:
        actions = np.array(data["actions"][..., :EE_ACTION_DIM], dtype=np.float32, copy=True)
        single_action = actions.ndim == 1
        if single_action:
            actions = actions[None, ...]
        state = _as_float_array(data["state"], name="state")
        if state.shape[-1] < EE_ACTION_DIM:
            raise ValueError(f"Expected at least {EE_ACTION_DIM} state dimensions, got {state.shape}")

        for start in (0, EE_ARM_ACTION_DIM):
            current_matrix = _pose10_to_matrix(state[..., start : start + 9])
            relative_matrix = _pose10_to_matrix(actions[..., start : start + 9])
            absolute_matrix = current_matrix[..., None, :, :] @ relative_matrix
            actions[..., start : start + 9] = _matrix_to_pose10(absolute_matrix)
            actions[..., start + 9] = np.clip(actions[..., start + 9], 0.0, 1.0)
        if single_action:
            actions = actions[0]
        return {"actions": actions}
