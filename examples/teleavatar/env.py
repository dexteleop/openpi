#!/usr/bin/env python3
"""
Environment wrapper for Teleavatar robot using openpi_client.runtime framework.
"""

import logging
import threading
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.teleavatar import ros2_interface


class TeleavatarEnvironment(_environment.Environment):
    """Environment for Teleavatar dual-arm robot."""

    def __init__(
        self,
        prompt: str = "pick a toy and put it in the basket using left gripper",
    ):
        """Initialize Teleavatar environment.

        Args:
            prompt: Default language instruction for the policy

        Note: Images are NOT resized here. Stereo cameras stay at 480×848; the
        head camera arrives already cropped to the left eye and rotated 180°
        from the ROS2 PyAV decode (224×224), matching the training head view.
        """
        self._prompt = prompt

        # Initialize ROS2 interface in a separate thread
        self._ros_interface: Optional[ros2_interface.TeleavatarROS2Interface] = None
        self._ros_thread: Optional[threading.Thread] = None
        self._init_ros2()

        logging.info(f"TeleavatarEnvironment initialized with prompt: '{prompt}'")

    def _init_ros2(self):
        """Initialize ROS2 in a background thread and wait for initial sensor data."""
        import rclpy
        import time

        # Event to signal when executor starts spinning
        spin_started = threading.Event()

        def ros_spin():
            rclpy.init()
            self._ros_interface = ros2_interface.TeleavatarROS2Interface()

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
                "Please check that ROS2 topics are publishing:\n"
                "  ros2 topic list\n"
                "  ros2 topic hz /left/color/image_raw\n"
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

        Returns:
            Dictionary with keys:
                - 'state': 48-dim proprioceptive state
                - 'images': Dict of camera images in (H, W, C) format at ORIGINAL resolution
                - 'prompt': Language instruction

        Note: Image formats match the training data:
            - left_color, right_color: 480×848×3 (H,W,C)
            - head_camera: 224×224×3 (H,W,C), already left-eye-cropped + rot180
              by the ROS2 PyAV decode callback.
        The policy's _parse_image will handle any format conversion if needed.
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        # Get raw observation from ROS2
        raw_obs = self._ros_interface.get_observation()
        if raw_obs is None:
            raise RuntimeError("Failed to get observation from ROS2 interface")

        # Process images: keep original resolution AND keep (H, W, C) format
        # Policy's _parse_image will handle format conversion if needed
        # Return with the exact keys expected by teleavatar_policy.py
        return {
            'observation/state': raw_obs['state'],
            'observation/images/left_color': image_tools.convert_to_uint8(raw_obs['images']['left_color']),
            'observation/images/right_color': image_tools.convert_to_uint8(raw_obs['images']['right_color']),
            'observation/images/head_camera': image_tools.convert_to_uint8(raw_obs['images']['head_camera']),
            'prompt': self._prompt,
        }

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

        # Publish to ROS2
        self._ros_interface.publish_action(actions)

    def set_prompt(self, prompt: str):
        """Update the language instruction prompt.

        Args:
            prompt: New language instruction
        """
        self._prompt = prompt
        logging.info(f"Updated prompt to: '{prompt}'")

    def __del__(self):
        """Cleanup when environment is destroyed."""
        if self._ros_thread is not None and self._ros_thread.is_alive():
            logging.info("Shutting down ROS2 thread...")
