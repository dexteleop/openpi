#!/usr/bin/env python3
"""
ROS2 interface wrapper for Teleavatar robot.
Handles subscribing to sensor topics and publishing actions.

Image decoding strategy:
- Subscribes directly to FFMPEGPacket (H.265) topics, bypassing the
  ffmpeg_image_transport republish node.
- Uses PyAV with hevc_cuvid (GPU) for decoding, falling back to CPU hevc.
- head_camera (the 2:1 stereo XR video on /xr_video_topic): GPU hw-resize
  2160×4320 → 224×448 during decode, then crop the left eye + rotate 180°
  → 224×224. This matches what TeleavatarInputs(rotate_head_camera=True)
  produces from the raw 2:1 stereo at training time, so train and deploy see
  the same head view.
- left_color / right_color: decoded as-is at 480×848.
"""

import logging
import pathlib
import time
from collections import deque
from threading import Lock
from typing import Dict, Optional

import av
import numpy as np
import rclpy
import yaml
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

_HZ_WINDOW = 60  # number of frames to average for the per-camera Hz estimate


def _make_codec(name: str, options: dict | None = None) -> av.CodecContext:
    """Create and open a PyAV codec context, trying GPU first then CPU."""
    gpu_codec = "hevc_cuvid"
    cpu_codec = "hevc"
    try:
        ctx = av.CodecContext.create(gpu_codec, "r")
        if options:
            ctx.options = options
        ctx.open()
        logging.info(f"[{name}] using {gpu_codec}" + (f" options={options}" if options else ""))
        return ctx
    except Exception as e:
        logging.warning(f"[{name}] {gpu_codec} unavailable ({e}), falling back to CPU")
        ctx = av.CodecContext.create(cpu_codec, "r")
        ctx.open()
        return ctx


class TeleavatarROS2Interface(Node):
    """Thread-safe ROS2 interface for Teleavatar robot sensors and actuators."""

    def __init__(self, node_name: str = "teleavatar_openpi_interface", sensor_timeout: float = 1.0):
        super().__init__(node_name)

        self.logger = self.get_logger()
        self.lock = Lock()
        # Sensor data older than this (seconds) is treated as dead: cameras
        # stream continuously and joint states arrive at ~100 Hz, so 1 s of
        # silence means the source is gone, not slow. get_observation then
        # returns None instead of the frozen last sample, so the policy never
        # acts on dead sensors.
        self.sensor_timeout = sensor_timeout

        # Outgoing arm commands are clamped to the arm_config.yml joint
        # limits before publishing (see _clamp_arm_cmd).
        arm_config = yaml.safe_load(
            open(pathlib.Path(__file__).resolve().parents[2] / "arm_config.yml")
        )
        self.joint_lower = {
            'left_arm': np.array(arm_config["arms"]["left_arm"]["lower"]),
            'right_arm': np.array(arm_config["arms"]["right_arm"]["lower"]),
        }
        self.joint_upper = {
            'left_arm': np.array(arm_config["arms"]["left_arm"]["upper"]),
            'right_arm': np.array(arm_config["arms"]["right_arm"]["upper"]),
        }

        # Storage for latest sensor data
        self.latest_images: Dict[str, np.ndarray] = {}
        self.latest_joint_states: Dict[str, JointState] = {}
        self.image_timestamps: Dict[str, float] = {}
        self.joint_timestamps: Dict[str, float] = {}

        # Per-camera receive-time / decode-latency tracking for Hz logging.
        self._image_recv_times: Dict[str, deque] = {
            "left_color": deque(maxlen=_HZ_WINDOW),
            "right_color": deque(maxlen=_HZ_WINDOW),
            "head_camera": deque(maxlen=_HZ_WINDOW),
        }
        self._image_latencies: Dict[str, deque] = {
            "left_color": deque(maxlen=_HZ_WINDOW),
            "right_color": deque(maxlen=_HZ_WINDOW),
            "head_camera": deque(maxlen=_HZ_WINDOW),
        }
        self._last_hz_log: float = time.time()
        self._hz_log_interval: float = 5.0  # log camera Hz/latency every N seconds

        self.left_joint_names = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']
        self.right_joint_names = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
        self.left_gripper_names = ['l_joint8']
        self.right_gripper_names = ['r_joint8']

        # PyAV codec contexts (one per camera). head_camera uses a GPU hw-resize
        # 2160×4320 → 224×448 during decode; left/right decode at native 480×848.
        self._codecs: Dict[str, av.CodecContext] = {
            "left_color": _make_codec("left_color"),
            "right_color": _make_codec("right_color"),
            "head_camera": _make_codec("head_camera", options={"resize": "448x224"}),
        }

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        self.logger.info("TeleavatarROS2Interface initialized (waiting for sensor data in background)")

    def _setup_subscribers(self):
        """Setup ROS2 subscribers for images and joint states."""
        # Image subscribers — subscribe directly to the H.265 (FFMPEGPacket)
        # topics and decode with PyAV (bypasses the republish node). head_camera
        # is the 2:1 stereo XR video on /xr_video_topic; the decode callback
        # crops the left eye and rotates it (see _ffmpeg_callback).
        self.create_subscription(
            FFMPEGPacket,
            '/left/color/image_raw/ffmpeg',
            lambda msg: self._ffmpeg_callback(msg, 'left_color'),
            10,
        )
        self.create_subscription(
            FFMPEGPacket,
            '/right/color/image_raw/ffmpeg',
            lambda msg: self._ffmpeg_callback(msg, 'right_color'),
            10,
        )
        self.create_subscription(
            FFMPEGPacket,
            '/xr_video_topic/ffmpeg',   # 2:1 stereo head camera
            lambda msg: self._ffmpeg_callback(msg, 'head_camera'),
            10,
        )

        # Joint state subscribers - explicit subscriptions
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
        """Setup ROS2 publishers for action commands."""
        self.action_publishers = {
            'left_arm': self.create_publisher(JointState, '/left_arm/model_joint_cmd', 10),
            'right_arm': self.create_publisher(JointState, '/right_arm/model_joint_cmd', 10),
            'left_gripper': self.create_publisher(Float32, '/api/left_gripper/cmd', 10),
            'right_gripper': self.create_publisher(Float32, '/api/right_gripper/cmd', 10),
        }
        self.enable_pub = self.create_publisher(Float32, '/api/fsm/enable', 10)
        self.logger.info("ROS2 publishers initialized")

    def _ffmpeg_callback(self, msg: FFMPEGPacket, camera_name: str):
        """Decode an H.265 FFMPEGPacket with PyAV (GPU hevc_cuvid)."""
        try:
            t0 = time.time()
            pkt = av.Packet(bytes(msg.data))
            pkt.pts = msg.pts
            pkt.dts = msg.pts  # ffmpeg_image_transport sometimes leaves dts unset

            for frame in self._codecs[camera_name].decode(pkt):
                if camera_name == "head_camera":
                    # GPU hevc_cuvid hw-resizes 2160×4320 → 224×448 during decode.
                    # Crop the left eye + rotate 180° (camera mounted upside-down)
                    # so the frame matches TeleavatarInputs(rotate_head_camera=True)
                    # on the raw 2:1 stereo at training time (same pixel mapping:
                    # rot180 then left-half). The 2:1-width guard also covers a
                    # CPU-fallback decode (no hw-resize → 2160×4320); the policy's
                    # ResizeImages downsizes whatever square crop we hand back.
                    img = frame.to_ndarray(format="rgb24")
                    h, w = img.shape[:2]
                    if w == 2 * h:
                        img = np.rot90(img, k=2)
                        img = np.ascontiguousarray(img[:, :h, :])
                else:
                    img = frame.to_ndarray(format="rgb24")  # 480×848×3

                now = time.time()
                with self.lock:
                    self.latest_images[camera_name] = img
                    self.image_timestamps[camera_name] = now
                    self._image_recv_times[camera_name].append(now)
                    self._image_latencies[camera_name].append((now - t0) * 1000)

                self._maybe_log_hz()
                break  # one packet → at most one output frame
        except Exception as e:
            self.logger.error(f"Failed to decode {camera_name}: {e}")

    def _maybe_log_hz(self):
        """Periodically log per-camera frame rate and decode latency."""
        now = time.time()
        if now - self._last_hz_log < self._hz_log_interval:
            return
        self._last_hz_log = now

        parts = []
        with self.lock:
            for name in self._image_recv_times:
                times = self._image_recv_times[name]
                lats = self._image_latencies[name]
                hz = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 else 0.0
                lat = float(np.mean(lats)) if lats else 0.0
                parts.append(f"{name}={hz:.1f}Hz/{lat:.1f}ms")

        self.logger.info(f"Cameras: {', '.join(parts)}")

    def _joint_state_callback(self, msg: JointState, joint_group: str):
        """Callback for joint state messages."""
        # self.logger.info(f"Received joint state from {joint_group} at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        with self.lock:
            self.latest_joint_states[joint_group] = msg
            self.joint_timestamps[joint_group] = time.time()

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for initial sensor data to arrive.

        NOTE: This should be called AFTER the ROS2 node starts spinning,
        otherwise callbacks will never be triggered!

        Returns:
            True if all data received, False if timeout
        """
        required_images = ['left_color', 'right_color', 'head_camera']
        required_joints = ['left_arm', 'right_arm']

        start_time = time.time()
        self.logger.info("Waiting for initial sensor data...")

        last_status_time = start_time
        while time.time() - start_time < timeout:
            with self.lock:
                images_ready = all(cam in self.latest_images for cam in required_images)
                joints_ready = all(joint in self.latest_joint_states for joint in required_joints)

                # Log progress every 2 seconds
                if time.time() - last_status_time > 2.0:
                    have_images = [cam for cam in required_images if cam in self.latest_images]
                    have_joints = [joint for joint in required_joints if joint in self.latest_joint_states]
                    self.logger.info(f"  Progress: images={have_images}, joints={have_joints}")
                    last_status_time = time.time()

                if images_ready and joints_ready:
                    self.logger.info("✓ All sensor data received!")
                    return True

            time.sleep(0.1)

        # Timeout - log what's missing
        with self.lock:
            missing_images = [cam for cam in required_images if cam not in self.latest_images]
            missing_joints = [joint for joint in required_joints if joint not in self.latest_joint_states]

        self.logger.error(
            f"✗ Timeout waiting for sensor data after {timeout}s. "
            f"Missing: images={missing_images}, joints={missing_joints}"
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
        required_images = ['left_color', 'right_color', 'head_camera']
        required_joints = ['left_arm', 'right_arm']

        with self.lock:
            dead = []
            for cam in required_images:
                stamp = self.image_timestamps.get(cam)
                if cam not in self.latest_images or stamp is None:
                    dead.append(f"camera:{cam}: not received")
                elif now - stamp > self.sensor_timeout:
                    dead.append(f"camera:{cam}: {now - stamp:.1f}s stale")
            for joint_group in required_joints:
                stamp = self.joint_timestamps.get(joint_group)
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

        with self.lock:
            # Build 48-dimensional state vector
            # Layout: [positions(16), velocities(16), efforts(16)]
            state_48d = np.zeros(48, dtype=np.float32)

            # Extract joint data
            left_arm = self.latest_joint_states['left_arm']
            right_arm = self.latest_joint_states['right_arm']
            # left_gripper = self.latest_joint_states['left_gripper']
            # right_gripper = self.latest_joint_states['right_gripper']

            # Positions (indices 0-15)
            state_48d[0:7] = self._extract_joint_field(left_arm, 'position', 7)
            # state_48d[7] = self._extract_joint_field(left_gripper, 'position', 1)[0]
            state_48d[8:15] = self._extract_joint_field(right_arm, 'position', 7)
            # state_48d[15] = self._extract_joint_field(right_gripper, 'position', 1)[0]

            # Velocities (indices 16-31)
            state_48d[16:23] = self._extract_joint_field(left_arm, 'velocity', 7)
            # state_48d[23] = self._extract_joint_field(left_gripper, 'velocity', 1)[0]
            state_48d[24:31] = self._extract_joint_field(right_arm, 'velocity', 7)
            # state_48d[31] = self._extract_joint_field(right_gripper, 'velocity', 1)[0]

            # Efforts (indices 32-47)
            state_48d[32:39] = self._extract_joint_field(left_arm, 'effort', 7)
            # state_48d[39] = self._extract_joint_field(left_gripper, 'effort', 1)[0]
            state_48d[40:47] = self._extract_joint_field(right_arm, 'effort', 7)
            # state_48d[47] = self._extract_joint_field(right_gripper, 'effort', 1)[0]

            return {
                'images': {
                    'left_color': self.latest_images['left_color'].copy(),
                    'right_color': self.latest_images['right_color'].copy(),
                    # head_camera is already the left eye, rotated 180° (cropped
                    # in _ffmpeg_callback; 224×224 on the GPU decode path).
                    'head_camera': self.latest_images['head_camera'].copy(),
                },
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

        Publishes position commands to model_joint_cmd topics for arms
        (clamped to the arm_config.yml joint limits first); a separate
        control node (arm_pd_controller) subscribes to these and computes
        velocity commands. Gripper triggers are clipped to [0, 1].

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

        # Left gripper (effort)
        # left_gripper_msg = JointState()
        # left_gripper_msg.header.stamp = timestamp
        # left_gripper_msg.header.frame_id = 'left_gripper'
        # left_gripper_msg.name = self.left_gripper_names
        # left_gripper_msg.position = [0.0]
        # left_gripper_msg.velocity = [0.0]
        # left_gripper_msg.effort = [float(actions[7])]
        left_gripper_msg = Float32()
        grip_value_left = float(actions[7])
        if grip_value_left > 0:
            data_left = 0.5 - grip_value_left / 7.0
        else:  # grip_value_left <= 0
            data_left = 0.5 - grip_value_left
        left_gripper_msg.data = float(np.clip(data_left, 0.0, 1.0))
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

        # Right gripper (effort)
        # right_gripper_msg = JointState()
        # right_gripper_msg.header.stamp = timestamp
        # right_gripper_msg.header.frame_id = 'right_gripper'
        # right_gripper_msg.name = self.right_gripper_names
        # right_gripper_msg.position = [0.0]
        # right_gripper_msg.velocity = [0.0]
        right_gripper_msg = Float32()
        grip_value_right = float(actions[15])
        if grip_value_right < 0:
            data_right = 0.5 + grip_value_right / 7.0
        else:  # grip_value_right >= 0
            data_right = 0.5 + grip_value_right
        right_gripper_msg.data = float(np.clip(data_right, 0.0, 1.0))
        self.action_publishers['right_gripper'].publish(right_gripper_msg)
