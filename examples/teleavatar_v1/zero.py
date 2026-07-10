#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.clock import Clock
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32
import numpy as np
import yaml
robot_config = yaml.safe_load(open("arm_config.yml"))
class JointInterpolator(Node):
    def __init__(self, namespace="right_arm", down=True, timer_on=False):
        super().__init__(f"{namespace}_joint_interpolator")
        self.namespace = namespace
        self.down = down
        self.subscription = self.create_subscription(
            JointState,
            f"/{namespace}/joint_states",
            self.joint_state_callback,
            10,
        )
        self.publisher = self.create_publisher(
            JointState, f"/api/{namespace}/joint_cmd", 10
        )
        # Enable signal
        if timer_on == True:
            self.enable_pub = self.create_publisher(Float32, '/api/fsm/enable', 10)
            self.create_timer(0.02, self.enable_callback)
        self.err_publisher = self.create_publisher(
            JointState, f"/{namespace}/traj_error", 10
        )
        self.current_pos = None
        self.start_time = None
        self.joint_names = []
        self.duration = 5.0  # seconds
        self.target_positions = []
        self.position_tolerance = 0.05  
        self.convergence_duration = 0.5  
        self.convergence_start_time = None
        self.is_converged = False
        num_activate = 7
        self.enable_flags = [False] * (7 - num_activate) + [True] * num_activate
        self.upper = np.array(robot_config["arms"][namespace]["upper"])
        self.lower = np.array(robot_config["arms"][namespace]["lower"])
        # Timer to run at 100 Hz for publishing commands
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100 Hz
        if self.down:
            joint_2_down_pos = 1.5 if self.namespace == "left" else -1.5
            joint_4_down_pos = 0.0
        else:
            joint_2_down_pos, joint_4_down_pos = 0, 0
        # 根据namespace设置不同的目标位置
        if self.namespace == "right_arm":
            # 右臂目标位置
            self.target_positions = [
                -0.50,  # r_joint1
                -0.92,  # r_joint2
                0.52,  # r_joint3
                -1.28,    # r_joint4
                0.32,  # r_joint5
                0.55, # r_joint6
                -0.52   # r_joint7
            ]
        else:  # left
            # 左臂目标位置
            self.target_positions = [
                0.41,  # l_joint1
                1.16,  # l_joint2
                -0.47,  # l_joint3
                0.90,    # l_joint4
                0.23,  # l_joint5
                -0.15, # l_joint6
                0.60   # l_joint7
            ]
        self.get_logger().info(
            f"Joint interpolator node initialized for {namespace}. Waiting for initial joint states..."
        )
        self.get_logger().info(f"Position tolerance: {self.position_tolerance} rad")
        self.get_logger().info(f"Convergence duration: {self.convergence_duration} sec")
    def enable_callback(self):
        msg = Float32()
        msg.data = 1.0
        self.enable_pub.publish(msg)
    def joint_state_callback(self, msg):
        self.joint_names = msg.name
        self.current_pos = np.array(msg.position)
        self.current_vel = np.array(msg.velocity)
    def check_convergence(self, current_position, target_position):
        enabled_indices = [i for i, enabled in enumerate(self.enable_flags) if enabled]
        position_errors = np.abs(np.array(current_position) - np.array(target_position))
        enabled_errors = position_errors[enabled_indices]
        max_error = np.max(enabled_errors)
        return max_error < self.position_tolerance
    def timer_callback(self):
        if self.current_pos is None or self.start_time is None:
            self.get_logger().warn(
                "Current position not set yet, skipping timer callback."
            )
            self.start_time = self.get_clock().now()
            return
        current_time = self.get_clock().now()
        elapsed_time = (
            current_time - self.start_time
        ).nanoseconds / 1e9  # Convert to seconds
        cmd_msg = JointState()
        cmd_msg.header.stamp = current_time.to_msg()
        cmd_msg.name = self.joint_names
        if elapsed_time >= self.duration:
            # Publish target positions (all zeros) once duration is reached
            cmd_msg.position = self.target_positions
        else:
            # Linear interpolation between initial and target positions
            alpha = elapsed_time / self.duration
            interpolated_positions = [
                self.current_pos[i] * (1.0 - alpha) + self.target_positions[i] * alpha
                for i in range(len(self.current_pos))
            ]
            cmd_msg.position = interpolated_positions
        cmd_msg.velocity = [0.0] * 7
        des_position = np.array(cmd_msg.position)
        kp = np.array([ 7, 7, 10, 10, 10, 8, 8 ])
        vel_fb = kp * (des_position - self.current_pos)
        for i in range(len(self.enable_flags)):
            if self.enable_flags[i]:
                cmd_msg.position[i] = self.current_pos[i]
                cmd_msg.velocity[i] = vel_fb[i]
        cmd_msg.effort = [0.0] * 7
        self.publisher.publish(cmd_msg)
        msg = JointState()
        msg.header.stamp = current_time.to_msg()
        msg.name = self.joint_names
        msg.position = [
            self.current_pos[i] - cmd_msg.position[i]
            for i in range(len(self.current_pos))
        ]
        self.err_publisher.publish(msg)
        if elapsed_time >= self.duration: 
            target_pos = np.array(self.target_positions)
            if self.check_convergence(self.current_pos, target_pos):
                if self.convergence_start_time is None:
                    self.convergence_start_time = current_time
                    self.get_logger().info(
                        f"{self.namespace} arm: Position converged, starting convergence timer..."
                    )
                else:
                    convergence_elapsed = (
                        current_time - self.convergence_start_time
                    ).nanoseconds / 1e9
                    if convergence_elapsed >= self.convergence_duration:
                        if not self.is_converged:
                            self.get_logger().info(
                                f"{self.namespace} arm: Convergence achieved! Stopping control."
                            )
                            self.is_converged = True
            else:
                if self.convergence_start_time is not None:
                    self.get_logger().info(
                        f"{self.namespace} arm: Position diverged, resetting convergence timer..."
                    )
                    self.convergence_start_time = None
def main(args=None):
    rclpy.init(args=args)
    joint_interpolator = JointInterpolator()
    rclpy.spin(joint_interpolator)
    joint_interpolator.destroy_node()
    rclpy.shutdown()
def main_multithreaded(args=None):
    rclpy.init(args=args)
    left_node = JointInterpolator(namespace="left_arm", down=True, timer_on=True)  # left arm
    right_node = JointInterpolator(namespace="right_arm", down=True, timer_on=True)  # right arm
    executor = MultiThreadedExecutor()
    executor.add_node(left_node)
    executor.add_node(right_node)
    try:
        print("Starting both arm nodes with MultiThreadedExecutor...")
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if left_node.is_converged and right_node.is_converged:
                print("Both arms converged! Shutting down...")
                break
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        left_node.destroy_node()
        right_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
if __name__ == "__main__":
    main_multithreaded()