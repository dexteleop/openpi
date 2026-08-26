#!/usr/bin/env python3
"""
Environment wrapper for Teleavatar robot using openpi_client.runtime framework.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.teleavatar_v2 import ros2_interface


class TeleavatarEnvironment(_environment.Environment):
    """Environment for Teleavatar dual-arm robot."""

    def __init__(
        self,
        prompt: str = "pick a toy and put it in the basket using left gripper",
        control_frequency: float = 30.0,
        interp_frequency: float = 200.0,
        interpolate: bool = True,
        recovery_ramp_s: float = 0.75,
        outage_poll_hz: float = 100.0,
    ):
        """Initialize Teleavatar environment.

        Args:
            prompt: Default language instruction for the policy
            control_frequency: Rate (Hz) apply_action is called; sets the interp ramp duration.
            interp_frequency: Rate (Hz) the interface republishes interpolated des_q (~200 Hz).
            interpolate: Enable des_q interpolation (False = raw ZOH publish).
            recovery_ramp_s: Ramp duration for the first command after a sensor
                outage. The arm has been frozen, so traversing a large pose
                delta in the usual one control period would be a jerk.
            outage_poll_hz: Rate to re-check sensors while frozen.

        Note: Images are NOT resized here. All three views arrive from the
        RTP/H265 composite stream (rtp_video_interface) already split to single
        eyes (head 960×960, wrists 400×640; see
        ros2_interface._POLICY_TO_RTP_VIEW), matching the crops applied to
        the training videos.
        """
        self._prompt = prompt
        self._control_frequency = control_frequency
        self._interp_frequency = interp_frequency
        self._interpolate = interpolate
        self._recovery_ramp_s = recovery_ramp_s
        self._outage_poll_period = 1.0 / max(outage_poll_hz, 1.0)

        # Set by attach_agent(); used to drop the cached action chunk after an
        # outage so we never replay actions computed from pre-outage images.
        self._agent = None
        self._pending_recovery = False

        # Initialize ROS2 interface in a separate thread
        self._ros_interface: Optional[ros2_interface.TeleavatarROS2Interface] = None
        self._ros_thread: Optional[threading.Thread] = None
        self._init_ros2()

        logging.info(f"TeleavatarEnvironment initialized with prompt: '{prompt}'")

    def attach_agent(self, agent) -> None:
        """Register the agent so a sensor outage can invalidate its action chunk.

        ActionChunkBroker caches a whole chunk (16 steps here) and keeps
        serving it without re-querying the policy. After an outage those
        actions were computed from stale images, so they must be discarded.
        """
        self._agent = agent

    def _init_ros2(self):
        """Initialize ROS2 in a background thread and wait for initial sensor data.

        The executor runs in a daemon thread for the lifetime of the process;
        there is no teardown, so only one environment per process is supported
        (rclpy.init() cannot be called twice).
        """
        import rclpy

        # Event to signal when executor starts spinning
        spin_started = threading.Event()

        def ros_spin():
            rclpy.init()
            self._ros_interface = ros2_interface.TeleavatarROS2Interface(
                control_frequency=self._control_frequency,
                interp_frequency=self._interp_frequency,
                interpolate=self._interpolate,
            )

            # Spin in background
            executor = rclpy.executors.MultiThreadedExecutor()
            executor.add_node(self._ros_interface)

            # Signal that spinning is about to start
            spin_started.set()

            try:
                executor.spin()
            finally:
                executor.shutdown()
                self._ros_interface.destroy_node()
                rclpy.shutdown()

        self._ros_thread = threading.Thread(target=ros_spin, daemon=True)
        self._ros_thread.start()

        # Wait for ROS2 interface object to be created
        timeout = 10.0
        start_time = time.time()
        while self._ros_interface is None and time.time() - start_time < timeout:
            time.sleep(0.1)

        if self._ros_interface is None:
            raise RuntimeError("Failed to initialize ROS2 interface object within timeout")

        logging.info("ROS2 interface object created, waiting for executor to start spinning...")

        # Wait for executor to start spinning
        if not spin_started.wait(timeout=5.0):
            raise RuntimeError("ROS2 executor failed to start spinning")

        logging.info("ROS2 executor started, waiting for initial sensor data...")

        # Now wait for initial sensor data (callbacks can now be triggered)
        if not self._ros_interface.wait_for_initial_data(timeout=30.0):
            raise RuntimeError(
                "Failed to receive initial sensor data. "
                "Please check that the RTP video stream and ROS2 joint topics are up:\n"
                "  - S100 is pushing the RTP/H265 stream to this machine (default port 8890, payload 96)\n"
                "  - GStreamer H265 decode works (gst-inspect-1.0 nvh265dec)\n"
                "  - ROS_DOMAIN_ID matches the robot (export ROS_DOMAIN_ID=29) in this shell\n"
                "  ros2 topic list\n"
                "  ros2 topic echo /left_arm/joint_states --once"
            )

        logging.info("ROS2 interface initialized successfully with sensor data")

    @override
    def reset(self) -> None:
        """Reset the environment.

        For Teleavatar, this is a no-op as we don't have a reset mechanism.
        In a real deployment, you might want to move to a home position here.
        """
        logging.info("Environment reset called (no-op for Teleavatar)")

    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete.

        For Teleavatar, episodes never complete automatically - they must be
        terminated by the user (e.g., Ctrl+C).
        """
        return False

    @override
    def get_observation(self) -> dict:
        """Get current observation from robot sensors.

        Blocks until the sensors are fresh. If any sensor is missing or stale,
        this freezes the robot at its last commanded pose and waits for
        recovery rather than raising — see _wait_for_fresh_observation.

        Returns:
            Dictionary with keys:
                - 'state': 48-dim proprioceptive state
                - 'images': Dict of camera images in (H, W, C) format at ORIGINAL resolution
                - 'prompt': Language instruction

        Note: Image formats match the (cropped) training data:
            - left_color: RTP split `left_wrist_right_eye`, 400×640×3 (H,W,C)
            - right_color: RTP split `right_wrist_left_eye`, 400×640×3 (H,W,C)
            - head_camera: RTP split `head_left_eye`, 960×960×3 (H,W,C)
        All views are split from the RTP composite frame by the
        RTP video interface; the policy's _parse_image will handle any
        remaining format conversion if needed.
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        # Get raw observation from ROS2
        raw_obs = self._ros_interface.get_observation()
        if raw_obs is None:
            raw_obs = self._wait_for_fresh_observation()

        # Process images: keep original resolution AND keep (H, W, C) format
        # Policy's _parse_image will handle format conversion if needed
        # Return with the exact keys expected by teleavatar_v2_policy.py
        return {
            'observation/state': raw_obs['state'],
            'observation/images/left_color': image_tools.convert_to_uint8(raw_obs['images']['left_color']),
            'observation/images/right_color': image_tools.convert_to_uint8(raw_obs['images']['right_color']),
            'observation/images/head_camera': image_tools.convert_to_uint8(raw_obs['images']['head_camera']),
            'prompt': self._prompt,
        }

    def _wait_for_fresh_observation(self) -> dict:
        """Block until the sensors are live again, freezing the robot meanwhile.

        Freezing requires no action of its own: by not returning, the control
        loop stops calling apply_action, so the 200 Hz _interp_publish timer in
        ros2_interface keeps republishing the last des_q (alpha saturates at
        1.0) and keeps the 50 Hz /api/fsm/enable heartbeat alive. The arm holds
        position and stays enabled.

        This deliberately never raises. Exiting the process on a stale
        observation also kills the enable heartbeat, which is a worse outcome
        than waiting — a video outage is usually transient (the RTP watchdog
        restarts the pipeline if the fault is on our side).
        """
        outage_start = time.time()
        logging.error(
            "Sensors unavailable — freezing the robot at its last commanded pose and waiting. "
            "See the interface log above for which sensors are dead."
        )
        last_log = outage_start

        while True:
            time.sleep(self._outage_poll_period)
            raw_obs = self._ros_interface.get_observation()
            if raw_obs is not None:
                duration = time.time() - outage_start
                logging.warning(
                    "SENSOR_OUTAGE_END duration=%.2fs — discarding the cached action chunk "
                    "and easing back in over %.2fs",
                    duration,
                    self._recovery_ramp_s,
                )
                if self._agent is not None:
                    self._agent.reset()
                self._pending_recovery = True
                return raw_obs

            now = time.time()
            if now - last_log >= 5.0:
                logging.error("Still frozen: sensors unavailable for %.1fs", now - outage_start)
                last_log = now

    @override
    def apply_action(self, action: dict) -> None:
        """Apply action to the robot.

        Args:
            action: Dictionary containing 'actions' key with 16-dim action array
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        if 'actions' not in action:
            raise ValueError(f"Action dict must contain 'actions' key, got: {action.keys()}")

        actions = action['actions']
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=np.float32)

        # Ensure correct shape
        if actions.shape != (16,):
            raise ValueError(f"Expected 16-dim action, got shape {actions.shape}")

        # First command after an outage gets a longer ramp: the arm has been
        # frozen, so covering a large pose delta in one control period (33 ms)
        # would be a jerk.
        ramp_duration = self._recovery_ramp_s if self._pending_recovery else None
        self._pending_recovery = False

        # Publish to ROS2
        self._ros_interface.publish_action(actions, ramp_duration=ramp_duration)

    def set_prompt(self, prompt: str):
        """Update the language instruction prompt.

        Args:
            prompt: New language instruction
        """
        self._prompt = prompt
        logging.info(f"Updated prompt to: '{prompt}'")
