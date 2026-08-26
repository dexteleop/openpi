#!/usr/bin/env python3
"""
Sensor/actuator interface wrapper for Teleavatar robot.
Joint states and action commands go over ROS2; camera images arrive as a
single RTP/H265 composite stream decoded with GStreamer (see
rtp_video_interface.py in this directory).

Image strategy (v2 robot, RTP composite stream):
- The S100 downsamples, exposure-time-aligns and stitches the head camera
  (2 eyes) and both wrist cameras (2 eyes each) into one 2720×1280 composite
  frame, H.265-encodes it and pushes it over RTP (default port 8890,
  payload 96, ~45 fps). Cameras are no longer published as ROS2 topics.
- RTPH265VideoInterface decodes the stream and splits it into six per-eye
  views at fixed crop regions (TELEAVATAR_SPLIT_REGIONS).
- get_observation() maps the three views the policy needs onto the training
  keys (see _POLICY_TO_RTP_VIEW): head_camera ← head_left_eye (960×960),
  left_color ← left_wrist_right_eye (400×640), right_color ←
  right_wrist_left_eye (400×640) — the inner eyes facing the desk center.
  The mapping was verified on-robot to reproduce the same single-eye views
  TeleavatarInputs crops from the raw training videos, so its aspect-ratio
  guard passes them through untouched and train/deploy see the same pixels.
"""

import logging
import pathlib
import sys
import time
from threading import Lock
from typing import Dict, Optional

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

# Make the repo root importable regardless of cwd (for the package import below).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.teleavatar_v2.rtp_video_interface import RTPH265VideoInterface  # noqa: E402

# Maps the observation keys the policy expects (same as the training dataset
# keys) to the RTP split-view names. Note the RTP interface also exposes a
# "head_camera" key of its own — that one is the FULL 2720×1280 composite,
# not the head view, so the mapping below must be used instead of passing
# the RTP dict through.
_POLICY_TO_RTP_VIEW = {
    "head_camera": "head_left_eye",
    "left_color": "left_wrist_right_eye",
    "right_color": "right_wrist_left_eye",
}


class TeleavatarROS2Interface(Node):
    """Thread-safe interface for Teleavatar sensors (ROS2 joints + RTP video) and actuators."""

    def __init__(
        self,
        node_name: str = "teleavatar_openpi_interface",
        rtp_port: int = 8890,
        rtp_payload: int = 96,
        sensor_timeout: float = 1.0,
        control_frequency: float = 30.0,
        interp_frequency: float = 200.0,
        interpolate: bool = True,
    ):
        super().__init__(node_name)

        self.logger = self.get_logger()
        self.lock = Lock()
        # Sensor data older than this (seconds) is treated as dead (video
        # ~45 fps, joint states ~100 Hz): get_observation returns None
        # instead of the frozen last sample.
        self.sensor_timeout = sensor_timeout

        # Outgoing arm commands are clamped to the arm_config.yml joint
        # limits before publishing (see _clamp_arm_cmd).
        arm_config = yaml.safe_load(open(_REPO_ROOT / "arm_config.yml"))
        self.joint_lower = {
            'left_arm': np.array(arm_config["arms"]["left_arm"]["lower"]),
            'right_arm': np.array(arm_config["arms"]["right_arm"]["lower"]),
        }
        self.joint_upper = {
            'left_arm': np.array(arm_config["arms"]["left_arm"]["upper"]),
            'right_arm': np.array(arm_config["arms"]["right_arm"]["upper"]),
        }

        # Storage for latest joint data (images live in the RTP interface)
        self.latest_joint_states: Dict[str, JointState] = {}
        self.joint_timestamps: Dict[str, float] = {}

        self.left_joint_names = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']
        self.right_joint_names = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
        self.left_gripper_names = ['l_joint8']
        self.right_gripper_names = ['r_joint8']

        # Camera images arrive over RTP, not ROS2 (see module docstring). The
        # interface logs its own fps/decode-latency stats periodically.
        self._video = RTPH265VideoInterface(port=rtp_port, payload=rtp_payload)
        self._video.start()

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        # publish_action() sets a target; a timer ramps des_q linearly from the last
        # published position over one control period, smoothing the platform's zero-order-hold
        # staircase (which otherwise steps the inner velocity loop into spikes).
        self._interpolate = interpolate
        self._cmd_lock = Lock()
        self._ctrl_period = 1.0 / max(control_frequency, 1e-3)  # ramp duration per command
        self._interp_period = 1.0 / max(interp_frequency, 1.0)
        self._ramp_from: Optional[np.ndarray] = None   # (14,) arm des_q at ramp start
        self._ramp_to: Optional[np.ndarray] = None     # (14,) arm des_q target
        self._ramp_t0: Optional[float] = None          # monotonic time at ramp start
        self._ramp_duration = self._ctrl_period        # seconds to traverse the current ramp
        self._last_cmd_pos: Optional[np.ndarray] = None  # (14,) last published arm des_q
        self._gripper_target = np.zeros(2)             # [left, right] effort (Nm)
        self._have_target = False
        self._enable_counter = 0
        if self._interpolate:
            self._interp_timer = self.create_timer(self._interp_period, self._interp_publish)
            self.logger.info(
                f"des_q interpolation ON: {control_frequency:.0f} Hz command -> "
                f"{interp_frequency:.0f} Hz publish (ramp {self._ctrl_period*1e3:.0f} ms)"
            )
        else:
            self._interp_timer = None
            self.logger.info(
                f"des_q interpolation OFF: publishing raw commands at ~{control_frequency:.0f} Hz "
                "(platform ZOH-holds them to its inner-loop rate -- may jitter)"
            )

        self.logger.info("TeleavatarROS2Interface initialized (waiting for sensor data in background)")

    def _setup_subscribers(self):
        """Setup ROS2 subscribers for joint states."""
        self.create_subscription(
            JointState,
            '/left_arm/joint_states',
            lambda msg: self._joint_state_callback(msg, 'left_arm'),
            10
        )
        self.create_subscription(
            JointState,
            '/right_arm/joint_states',
            lambda msg: self._joint_state_callback(msg, 'right_arm'),
            10
        )

        self.logger.info("ROS2 subscribers initialized")

    def _setup_publishers(self):
        """Setup ROS2 publishers for action commands.

        The v2 platform accepts joint position commands directly on
        /api/<arm>/joint_cmd and does its own low-level control, so arm
        commands go straight there — no 100Hz PD relay (arm_pd_controller.py)
        in between anymore. Do not run that node alongside this interface;
        both publish to the same topics.
        """
        self.action_publishers = {
            'left_arm': self.create_publisher(JointState, '/api/left_arm/joint_cmd', 10),
            'right_arm': self.create_publisher(JointState, '/api/right_arm/joint_cmd', 10),
            'left_gripper': self.create_publisher(Float32, '/api/left_gripper/cmd', 10),
            'right_gripper': self.create_publisher(Float32, '/api/right_gripper/cmd', 10),
        }
        self.enable_pub = self.create_publisher(Float32, '/api/fsm/enable', 10)
        self.logger.info("ROS2 publishers initialized")

    def _joint_state_callback(self, msg: JointState, joint_group: str):
        """Callback for joint state messages."""
        with self.lock:
            self.latest_joint_states[joint_group] = msg
            self.joint_timestamps[joint_group] = time.time()

    def destroy_node(self):
        """Stop the RTP video pipeline before tearing down the ROS2 node."""
        try:
            self._video.stop()
        finally:
            super().destroy_node()

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for initial sensor data (first RTP video frame + joint states).

        NOTE: This should be called AFTER the ROS2 node starts spinning,
        otherwise joint callbacks will never be triggered! (The RTP video
        thread runs independently of the ROS2 executor.)

        Returns:
            True if all data received, False if timeout
        """
        required_joints = ['left_arm', 'right_arm']

        start_time = time.time()
        self.logger.info("Waiting for initial sensor data...")

        last_status_time = start_time
        while time.time() - start_time < timeout:
            video_ready = self._video.has_initial_frame()
            with self.lock:
                joints_ready = all(joint in self.latest_joint_states for joint in required_joints)
                have_joints = [joint for joint in required_joints if joint in self.latest_joint_states]

            if video_ready and joints_ready:
                self.logger.info("✓ All sensor data received!")
                return True

            # Log progress every 2 seconds
            if time.time() - last_status_time > 2.0:
                self.logger.info(f"  Progress: video={video_ready}, joints={have_joints}")
                last_status_time = time.time()

            time.sleep(0.1)

        # Timeout - log what's missing
        with self.lock:
            missing_joints = [joint for joint in required_joints if joint not in self.latest_joint_states]

        self.logger.error(
            f"✗ Timeout waiting for sensor data after {timeout}s. "
            f"Missing: video={not self._video.has_initial_frame()} (RTP port {self._video.port}), "
            f"joints={missing_joints}"
        )
        return False

    def get_observation(self) -> Optional[Dict]:
        """Get current observation from all sensors.

        Returns:
            Dictionary with 'images' and 'state' keys, or None if any sensor
            is missing or older than sensor_timeout. Callers must treat None
            as "do not act" — never fall back to a previous observation.
        """
        now = time.time()
        dead: list = []

        if self._video.stream_ended():
            dead.append("video: RTP pipeline stopped (EOS/error)")

        # Split views from the RTP stream (already copies; no shared buffers).
        rtp_images, rtp_stamps = self._video.get_latest_images_with_timestamps()
        for view in _POLICY_TO_RTP_VIEW.values():
            stamp = rtp_stamps.get(view)
            if view not in rtp_images or stamp is None:
                dead.append(f"video:{view}: not received")
            elif now - stamp > self.sensor_timeout:
                dead.append(f"video:{view}: {now - stamp:.1f}s stale")

        with self.lock:
            left_arm = self.latest_joint_states.get('left_arm')
            right_arm = self.latest_joint_states.get('right_arm')
            joint_stamps = dict(self.joint_timestamps)

        for joint_group in ('left_arm', 'right_arm'):
            stamp = joint_stamps.get(joint_group)
            if stamp is None:
                dead.append(f"joints:{joint_group}: not received")
            elif now - stamp > self.sensor_timeout:
                dead.append(f"joints:{joint_group}: {now - stamp:.1f}s stale")

        if dead:
            self.logger.error(
                "Observation unavailable — dead sensors: " + "; ".join(dead),
                throttle_duration_sec=1.0,
            )
            return None

        # Build 48-dimensional state vector
        # Layout: [positions(16), velocities(16), efforts(16)]
        # The v2 training data appends 14 EE-pose dims (indices 48-61),
        # but the policy only reads joint positions (indices 0:7 and
        # 8:15), which are identical in both layouts — so the client
        # doesn't need to send them.
        state_48d = np.zeros(48, dtype=np.float32)

        # Positions (indices 0-15)
        state_48d[0:7] = self._extract_joint_field(left_arm, 'position', 7)
        state_48d[8:15] = self._extract_joint_field(right_arm, 'position', 7)

        # Velocities (indices 16-31)
        state_48d[16:23] = self._extract_joint_field(left_arm, 'velocity', 7)
        state_48d[24:31] = self._extract_joint_field(right_arm, 'velocity', 7)

        # Efforts (indices 32-47)
        state_48d[32:39] = self._extract_joint_field(left_arm, 'effort', 7)
        state_48d[40:47] = self._extract_joint_field(right_arm, 'effort', 7)

        return {
            'images': {policy_key: rtp_images[view] for policy_key, view in _POLICY_TO_RTP_VIEW.items()},
            'state': state_48d,
        }

    def _extract_joint_field(self, msg: JointState, field: str, num_joints: int) -> np.ndarray:
        """Extract joint data field (position/velocity/effort) from JointState message."""
        data = getattr(msg, field, [])

        if len(data) >= num_joints:
            return np.array(data[:num_joints], dtype=np.float32)
        else:
            # Pad with zeros if not enough data
            result = np.zeros(num_joints, dtype=np.float32)
            result[:len(data)] = data
            return result

    def _clamp_arm_cmd(self, arm: str, positions: np.ndarray) -> np.ndarray:
        """Clamp an outgoing 7-dim arm position command to the arm_config.yml
        joint limits. A throttled warning is logged whenever clamping engages
        — frequent warnings mean the policy is commanding out-of-range
        motions."""
        target = np.asarray(positions, dtype=np.float64)
        clamped = np.clip(target, self.joint_lower[arm], self.joint_upper[arm])
        touched = np.flatnonzero(~np.isclose(clamped, target, atol=1e-9))
        if touched.size:
            names = self.left_joint_names if arm == 'left_arm' else self.right_joint_names
            details = ", ".join(
                f"{names[i]} {target[i]:.3f}->{clamped[i]:.3f}" for i in touched
            )
            self.logger.warning(
                f"{arm} command clamped to joint limits: {details}",
                throttle_duration_sec=2.0,
            )
        return clamped

    def publish_action(self, actions: np.ndarray, ramp_duration: Optional[float] = None):
        """Set the latest action as the interpolation target (16-dim:
        [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]).

        ramp_duration overrides the default one-control-period ramp. Use a
        longer one for the first command after a sensor outage: the arm has
        been frozen at _last_cmd_pos, and traversing a large pose delta in the
        usual 33 ms would be a jerk.
        """
        if actions.shape != (16,):
            self.logger.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        arm14 = np.concatenate([actions[0:7], actions[8:15]]).astype(np.float64)
        grip2 = np.array([actions[7], actions[15]], dtype=np.float64)
        if not self._interpolate:
            self._publish_cmd(arm14, grip2)
            return
        now = time.monotonic()
        with self._cmd_lock:
            if self._last_cmd_pos is None:
                start = self._current_arm_positions()  # take off from the measured pose
                self._last_cmd_pos = start if start is not None else arm14.copy()
            self._ramp_from = self._last_cmd_pos.copy()
            self._ramp_to = arm14
            self._ramp_t0 = now
            self._ramp_duration = max(
                self._ctrl_period if ramp_duration is None else ramp_duration, 1e-3
            )
            self._gripper_target = grip2
            self._have_target = True

    def _current_arm_positions(self) -> Optional[np.ndarray]:
        """Latest measured [left(7), right(7)] joint positions, or None if unavailable."""
        with self.lock:
            left = self.latest_joint_states.get('left_arm')
            right = self.latest_joint_states.get('right_arm')
        if left is None or right is None:
            return None
        return np.concatenate([
            self._extract_joint_field(left, 'position', 7),
            self._extract_joint_field(right, 'position', 7),
        ]).astype(np.float64)

    def _interp_publish(self):
        """Ramp des_q toward the latest target over _ramp_duration; alpha clamps to [0, 1],
        so a late command holds at the target and an early one resumes from the current pose.

        Holding at the target is also what freezes the arm during a sensor
        outage: the control loop stops calling publish_action, alpha saturates
        at 1.0, and this timer keeps republishing the last des_q plus the
        enable heartbeat in _publish_cmd.
        """
        with self._cmd_lock:
            if not self._have_target:
                return
            alpha = (time.monotonic() - self._ramp_t0) / self._ramp_duration
            alpha = min(max(alpha, 0.0), 1.0)
            pos = self._ramp_from + alpha * (self._ramp_to - self._ramp_from)
            self._last_cmd_pos = pos.copy()
            grip = self._gripper_target.copy()
        self._publish_cmd(pos, grip)

    @staticmethod
    def _gripper_trigger(effort: float) -> float:
        """Invert the v2 trigger->effort curve (>0 opens, <0 closes); clipped to [0, 1]."""
        if effort > 0:
            trigger = 0.10 * (1.0 - effort / 2.0)
        else:
            trigger = 0.10 - effort * 0.90 / 1.6
        return float(np.clip(trigger, 0.0, 1.0))

    def _publish_cmd(self, arm14: np.ndarray, grip2: np.ndarray):
        """Publish one des_q frame: both arms (clamped, velocity=0) + grippers + enable heartbeat."""
        timestamp = self.get_clock().now().to_msg()

        # enable heartbeat, ~50 Hz
        self._enable_counter += 1
        if self._enable_counter % 4 == 0:
            enable_msg = Float32()
            enable_msg.data = 1.0
            self.enable_pub.publish(enable_msg)

        # Left arm (position command).
        left_arm_msg = JointState()
        left_arm_msg.header.stamp = timestamp
        left_arm_msg.header.frame_id = 'left_arm'
        left_arm_msg.name = self.left_joint_names
        left_arm_msg.position = self._clamp_arm_cmd('left_arm', arm14[0:7]).tolist()
        left_arm_msg.velocity = np.zeros(7).tolist()
        left_arm_msg.effort = np.zeros(7).tolist()
        self.action_publishers['left_arm'].publish(left_arm_msg)

        # Right arm.
        right_arm_msg = JointState()
        right_arm_msg.header.stamp = timestamp
        right_arm_msg.header.frame_id = 'right_arm'
        right_arm_msg.name = self.right_joint_names
        right_arm_msg.position = self._clamp_arm_cmd('right_arm', arm14[7:14]).tolist()
        right_arm_msg.velocity = np.zeros(7).tolist()
        right_arm_msg.effort = np.zeros(7).tolist()
        self.action_publishers['right_arm'].publish(right_arm_msg)

        # grippers: effort -> [0, 1] trigger
        left_gripper_msg = Float32()
        left_gripper_msg.data = self._gripper_trigger(float(grip2[0]))
        self.action_publishers['left_gripper'].publish(left_gripper_msg)

        right_gripper_msg = Float32()
        right_gripper_msg.data = self._gripper_trigger(float(grip2[1]))
        self.action_publishers['right_gripper'].publish(right_gripper_msg)
