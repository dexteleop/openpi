#!/usr/bin/env python3
"""
Reset the Teleavatar robot to a recorded pose from a LeRobot dataset —
essentially the ramp phase of replay_episode.py as a standalone tool
(zero.py with the target pose read from an episode instead of hardcoded).

Ramps both arms from their CURRENT joint positions to the chosen frame of
the chosen episode over --ramp-s seconds, publishes the matching gripper
triggers, then holds the target until the measured positions converge
(like zero.py) and exits.

Usage (client conda env, robot enabled):
    python examples/teleavatar_v2/zero_from_episode.py --dataset <dataset_path> --episode 0
    python examples/teleavatar_v2/zero_from_episode.py --dataset <dataset_path> --episode 12 --frame -1   # last frame
    python examples/teleavatar_v2/zero_from_episode.py --dataset <dataset_path> --episode 0 --dry-run     # no ROS2 needed
"""

import argparse
import pathlib
import time

import numpy as np

from replay_episode import load_episode


def main():
    parser = argparse.ArgumentParser(description="Move the Teleavatar to a recorded episode pose")
    parser.add_argument("--dataset", type=pathlib.Path, required=True,
                        help="LeRobot dataset root (contains meta/ and data/)")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to take the pose from")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index of the target pose (negative counts from the end, e.g. -1 = last)")
    parser.add_argument("--source", choices=["action", "state"], default="action",
                        help="Take the pose from recorded commands (action) or measured trajectory (observation.state)")
    parser.add_argument("--ramp-s", type=float, default=5.0, help="Seconds to ramp from current pose to the target")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Per-joint convergence tolerance (rad)")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Max seconds to wait for convergence after the ramp")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Only print the target pose, publish nothing")
    args = parser.parse_args()

    left_pos, right_pos, left_trigger, right_trigger, _fps = load_episode(args.dataset, args.episode, args.source)
    target_left = left_pos[args.frame]
    target_right = right_pos[args.frame]
    target_left_trigger = float(left_trigger[args.frame])
    target_right_trigger = float(right_trigger[args.frame])

    frame_idx = args.frame if args.frame >= 0 else len(left_pos) + args.frame
    print(f"Target: episode {args.episode} frame {frame_idx}/{len(left_pos) - 1} ({args.source}) from {args.dataset}")
    print(f"left arm  = {np.round(target_left, 3).tolist()}")
    print(f"right arm = {np.round(target_right, 3).tolist()}")
    print(f"grippers  = left {target_left_trigger:.3f}, right {target_right_trigger:.3f} (trigger)")
    if args.dry_run:
        return

    if not args.yes:
        input(f"Robot will ramp to this pose over {args.ramp_s:.1f}s. Press Enter to start (Ctrl+C to abort)... ")

    # ROS2 imports are deferred so --dry-run works without a ROS2 environment.
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32

    rclpy.init()
    node = Node("teleavatar_zero_from_episode")
    pubs = {
        "left_arm": node.create_publisher(JointState, "/api/left_arm/joint_cmd", 10),
        "right_arm": node.create_publisher(JointState, "/api/right_arm/joint_cmd", 10),
        "left_gripper": node.create_publisher(Float32, "/api/left_gripper/cmd", 10),
        "right_gripper": node.create_publisher(Float32, "/api/right_gripper/cmd", 10),
        "enable": node.create_publisher(Float32, "/api/fsm/enable", 10),
    }
    joint_names = {
        "left_arm": ["l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5", "l_joint6", "l_joint7"],
        "right_arm": ["r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6", "r_joint7"],
    }
    current = {}
    node.create_subscription(JointState, "/left_arm/joint_states",
                             lambda m: current.update(left_arm=np.array(m.position[:7])), 10)
    node.create_subscription(JointState, "/right_arm/joint_states",
                             lambda m: current.update(right_arm=np.array(m.position[:7])), 10)

    def publish_frame(lp, rp):
        enable = Float32()
        enable.data = 1.0
        pubs["enable"].publish(enable)
        stamp = node.get_clock().now().to_msg()
        for arm, positions in (("left_arm", lp), ("right_arm", rp)):
            msg = JointState()
            msg.header.stamp = stamp
            msg.header.frame_id = arm
            msg.name = joint_names[arm]
            msg.position = [float(p) for p in positions]
            msg.velocity = [0.0] * 7
            msg.effort = [0.0] * 7
            pubs[arm].publish(msg)
        for grip, value in (("left_gripper", target_left_trigger), ("right_gripper", target_right_trigger)):
            msg = Float32()
            msg.data = value
            pubs[grip].publish(msg)

    try:
        node.get_logger().info("Waiting for joint states...")
        deadline = time.monotonic() + 10.0
        while len(current) < 2:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for /left_arm/joint_states and /right_arm/joint_states")

        # Ramp from the current pose to the target (100 Hz, like zero.py).
        node.get_logger().info(f"Ramping to target over {args.ramp_s:.1f}s...")
        start_left, start_right = current["left_arm"].copy(), current["right_arm"].copy()
        t0 = time.monotonic()
        while (elapsed := time.monotonic() - t0) < args.ramp_s:
            alpha = elapsed / args.ramp_s
            publish_frame(start_left * (1 - alpha) + target_left * alpha,
                          start_right * (1 - alpha) + target_right * alpha)
            rclpy.spin_once(node, timeout_sec=0)
            time.sleep(0.01)

        # Hold the target and wait for the measured positions to converge.
        node.get_logger().info(f"Holding target, waiting for convergence (<{args.tolerance} rad)...")
        converged_since = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.timeout_s:
            publish_frame(target_left, target_right)
            rclpy.spin_once(node, timeout_sec=0)
            err = max(np.abs(current["left_arm"] - target_left).max(),
                      np.abs(current["right_arm"] - target_right).max())
            if err < args.tolerance:
                converged_since = converged_since or time.monotonic()
                if time.monotonic() - converged_since >= 0.5:  # stable for 0.5s, like zero.py
                    node.get_logger().info(f"✓ Converged (max error {err:.3f} rad).")
                    return
            else:
                converged_since = None
            time.sleep(0.01)

        node.get_logger().warning(
            f"Did not converge within {args.timeout_s:.0f}s (max error "
            f"{max(np.abs(current['left_arm'] - target_left).max(), np.abs(current['right_arm'] - target_right).max()):.3f} rad); "
            "robot holds the last commanded pose."
        )
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted — stopped publishing; robot holds the last commanded pose.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
