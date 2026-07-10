#!/usr/bin/env python3
"""
Arm velocity controller node for Teleavatar.

Subscribes to:
  - /left_arm/model_joint_cmd (position commands from policy)
  - /left_arm/joint_states (actual joint states)
  - /right_arm/model_joint_cmd
  - /right_arm/joint_states

Publishes to:
  - /left_arm/joint_cmd (velocity commands at 100Hz)
  - /right_arm/joint_cmd (velocity commands at 100Hz)

Control law: v = kp * (des_q - state_q) + feedforward
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ArmVelocityController(Node):
    """PD controller with feedforward for arm velocity control."""

    def __init__(self):
        super().__init__('arm_velocity_controller')

        # Control parameters
        self.arm_ctrl_dt = 1.0 / 100.0  # 100 Hz control loop
        self.num_joints = 7

        # PD gains (feedback)
        self.kp_err_left = np.array([7.0, 7.0, 10.0, 10.0, 10.0, 8.0, 8.0])
        self.kp_err_right = np.array([7.0, 7.0, 10.0, 10.0, 10.0, 8.0, 8.0])
        self.joint_vel_limit = np.array([15.0, 15.0, 20.0, 20.0, 44.0, 33.0, 33.0])
        self.joint_vel_limit = np.array([15.0, 15.0, 20.0, 20.0, 44.0, 33.0, 33.0])

        # State storage - Left arm
        self.left_des_q = None
        self.left_state_q = None
        self.left_cmd_last_time = None  # Last time we received model_joint_cmd
        self.left_joint_names = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']

        # State storage - Right arm
        self.right_des_q = None
        self.right_state_q = None
        self.right_cmd_last_time = None  # Last time we received model_joint_cmd
        self.right_joint_names = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']

        # Timeout for model commands (seconds)
        self.cmd_timeout = 0.5  # If no command for 0.5s, stop publishing

        # Subscribers for model commands (from policy)
        self.create_subscription(
            JointState,
            '/left_arm/model_joint_cmd',
            self.left_model_cmd_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/right_arm/model_joint_cmd',
            self.right_model_cmd_callback,
            10
        )

        # Subscribers for joint states (from robot)
        self.create_subscription(
            JointState,
            '/left_arm/joint_states',
            self.left_state_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/right_arm/joint_states',
            self.right_state_callback,
            10
        )

        # Publishers for velocity commands
        self.left_cmd_pub = self.create_publisher(
            JointState,
            '/api/left_arm/joint_cmd',
            10
        )
        self.right_cmd_pub = self.create_publisher(
            JointState,
            '/api/right_arm/joint_cmd',
            10
        )

        # Control loop timer (100 Hz)
        self.create_timer(self.arm_ctrl_dt, self.control_loop)

        self.get_logger().info('Arm velocity controller initialized (100Hz)')

    def left_model_cmd_callback(self, msg: JointState):
        """Receive desired position from policy for left arm."""
        if len(msg.position) >= self.num_joints:
            self.left_des_q = np.array(msg.position[:self.num_joints])
            self.left_cmd_last_time = self.get_clock().now()

    def right_model_cmd_callback(self, msg: JointState):
        """Receive desired position from policy for right arm."""
        if len(msg.position) >= self.num_joints:
            self.right_des_q = np.array(msg.position[:self.num_joints])
            self.right_cmd_last_time = self.get_clock().now()

    def left_state_callback(self, msg: JointState):
        """Receive actual joint states for left arm."""
        if len(msg.position) >= self.num_joints:
            self.left_state_q = np.array(msg.position[:self.num_joints])

    def right_state_callback(self, msg: JointState):
        """Receive actual joint states for right arm."""
        if len(msg.position) >= self.num_joints:
            self.right_state_q = np.array(msg.position[:self.num_joints])

    def get_target_v(self, des_q, state_q, kp_err):
        """
        Compute target velocity using simple PD control.

        Args:
            des_q: Desired position
            state_q: Current position
            kp_err: Proportional gains

        Returns:
            target_velocity
        """
        vel_fb = kp_err * (des_q - state_q)
        vel_fb_clipped = np.clip(vel_fb, -0.3 * self.joint_vel_limit, 0.3 * self.joint_vel_limit)
        return vel_fb_clipped

    def control_loop(self):
        """Main control loop running at 100 Hz."""
        now = self.get_clock().now()
        timestamp = now.to_msg()

        # Control left arm
        if self.left_des_q is not None and self.left_state_q is not None:
            # Check if command is still fresh
            if self.left_cmd_last_time is not None:
                time_since_cmd = (now - self.left_cmd_last_time).nanoseconds / 1e9
                if time_since_cmd > self.cmd_timeout:
                    # Command timeout - stop publishing
                    self.get_logger().warn(
                        f'Left arm model_joint_cmd timeout ({time_since_cmd:.2f}s), stopping control',
                        throttle_duration_sec=2.0
                    )
                    self.left_des_q = None
                    return

            left_vel = self.get_target_v(
                self.left_des_q,
                self.left_state_q,
                self.kp_err_left
            )

            # Publish left arm command (position + velocity)
            left_msg = JointState()
            left_msg.header.stamp = timestamp
            left_msg.name = self.left_joint_names
            left_msg.header.frame_id = 'left_arm'
            left_msg.position = self.left_des_q.tolist()
            left_msg.velocity = left_vel.tolist()
            left_msg.effort = np.zeros(self.num_joints).tolist()
            self.left_cmd_pub.publish(left_msg)

        # Control right arm
        if self.right_des_q is not None and self.right_state_q is not None:
            # Check if command is still fresh
            if self.right_cmd_last_time is not None:
                time_since_cmd = (now - self.right_cmd_last_time).nanoseconds / 1e9
                if time_since_cmd > self.cmd_timeout:
                    # Command timeout - stop publishing
                    self.get_logger().warn(
                        f'Right arm model_joint_cmd timeout ({time_since_cmd:.2f}s), stopping control',
                        throttle_duration_sec=2.0
                    )
                    self.right_des_q = None
                    return

            right_vel = self.get_target_v(
                self.right_des_q,
                self.right_state_q,
                self.kp_err_right
            )

            # Publish right arm command (position + velocity)
            right_msg = JointState()
            right_msg.header.stamp = timestamp
            right_msg.header.frame_id = 'right_arm'
            right_msg.name = self.right_joint_names
            right_msg.position = self.right_des_q.tolist()
            right_msg.velocity = right_vel.tolist()
            right_msg.effort = np.zeros(self.num_joints).tolist()
            self.right_cmd_pub.publish(right_msg)


def main(args=None):
    rclpy.init(args=args)
    controller = ArmVelocityController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
