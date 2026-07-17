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
    ):
        super().__init__(node_name)

        self.logger = self.get_logger()
        self.lock = Lock()
        # Sensor data older than this (seconds) is treated as dead: video runs
        # ~45 fps and joint states ~100 Hz, so 1 s of silence means the source
        # is gone, not slow. get_observation then returns None instead of the
        # frozen last sample, so the policy never acts on dead sensors.
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

    def publish_action(self, actions: np.ndarray):
        """Publish 16-dimensional action to ROS topics.

        Publishes joint position commands directly to the platform's
        /api/<arm>/joint_cmd topics (the v2 platform runs its own low-level
        controller from positions) and gripper trigger values to
        /api/<side>_gripper/cmd. Arm commands are clamped first (see
        _clamp_arm_cmd); gripper triggers are clipped to [0, 1].

        Args:
            actions: 16-dim array [left_arm_pos(7), left_gripper_effort(1),
                                   right_arm_pos(7), right_gripper_effort(1)]
        """
        if actions.shape != (16,):
            self.logger.error(f"Expected 16-dim action, got shape {actions.shape}")
            return

        timestamp = self.get_clock().now().to_msg()

        # Enable FSM
        enable_msg = Float32()
        enable_msg.data = 1.0
        self.enable_pub.publish(enable_msg)

        # Left arm (position command)
        left_arm_msg = JointState()
        left_arm_msg.header.stamp = timestamp
        left_arm_msg.header.frame_id = 'left_arm'
        left_arm_msg.name = self.left_joint_names
        left_arm_msg.position = self._clamp_arm_cmd('left_arm', actions[0:7]).tolist()
        left_arm_msg.velocity = np.zeros(7).tolist()
        left_arm_msg.effort = np.zeros(7).tolist()
        self.action_publishers['left_arm'].publish(left_arm_msg)

        # Left gripper: convert the model's effort (Nm) to the platform's
        # [0, 1] trigger value — inverse of the v2 piecewise trigger→effort
        # curve (same for both arms; see teleavatar_v2_policy). Positive effort =
        # opening (trigger < 0.10), negative = closing (trigger > 0.10).
        # Clipped because the platform expects a 0~1 trigger.
        left_gripper_msg = Float32()
        effort_left = float(actions[7])
        if effort_left > 0:
            trigger_left = 0.10 * (1.0 - effort_left / 2.0)
        else:  # effort_left <= 0
            trigger_left = 0.10 - effort_left * 0.90 / 1.6
        left_gripper_msg.data = float(np.clip(trigger_left, 0.0, 1.0))
        self.action_publishers['left_gripper'].publish(left_gripper_msg)

        # Right arm (position command)
        right_arm_msg = JointState()
        right_arm_msg.header.stamp = timestamp
        right_arm_msg.header.frame_id = 'right_arm'
        right_arm_msg.name = self.right_joint_names
        right_arm_msg.position = self._clamp_arm_cmd('right_arm', actions[8:15]).tolist()
        right_arm_msg.velocity = np.zeros(7).tolist()
        right_arm_msg.effort = np.zeros(7).tolist()
        self.action_publishers['right_arm'].publish(right_arm_msg)

        # Right gripper: same v2 effort → trigger conversion as the left.
        right_gripper_msg = Float32()
        effort_right = float(actions[15])
        if effort_right > 0:
            trigger_right = 0.10 * (1.0 - effort_right / 2.0)
        else:  # effort_right <= 0
            trigger_right = 0.10 - effort_right * 0.90 / 1.6
        right_gripper_msg.data = float(np.clip(trigger_right, 0.0, 1.0))
        self.action_publishers['right_gripper'].publish(right_gripper_msg)
