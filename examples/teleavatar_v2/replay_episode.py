#!/usr/bin/env python3
"""
Replay one recorded LeRobot episode on the Teleavatar robot (v2 platform).

Reads the episode's action (or observation.state) sequence straight from the
dataset parquet and publishes it at the recorded fps using the same command
conventions as ros2_interface.py:
- arm joint positions → /api/{left,right}_arm/joint_cmd (JointState.position;
  the v2 platform runs its own low-level controller, velocity/effort left 0)
- gripper efforts → converted to [0,1] trigger values (v2 piecewise curve)
  → /api/{left,right}_gripper/cmd (Float32)
- FSM enable → /api/fsm/enable = 1.0 every tick

Safety: before replaying, the script linearly ramps from the CURRENT joint
positions to the episode's first frame over --ramp-s seconds (like zero.py),
so the arms never jump. Ctrl+C stops publishing immediately.

Usage (client conda env, robot enabled; run zero.py first if you want a
known home pose):
    python examples/teleavatar_v2/replay_episode.py --dataset <dataset_path> --episode 0
    python examples/teleavatar_v2/replay_episode.py --dataset <dataset_path> --episode 12 --speed 0.5
    python examples/teleavatar_v2/replay_episode.py --dataset <dataset_path> --episode 0 --dry-run   # no ROS2 needed

Needs pandas + pyarrow for parquet reading (pip install pandas pyarrow).
"""

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd


def gripper_effort_to_trigger(effort: np.ndarray) -> np.ndarray:
    """Recorded gripper effort (Nm) → platform [0,1] trigger value.

    Inverse of the v2 piecewise trigger→effort curve (same for both arms),
    identical to the conversion in ros2_interface.publish_action. Clipped
    because the platform expects a 0~1 trigger.
    """
    trigger = np.where(effort > 0, 0.10 * (1.0 - effort / 2.0), 0.10 - effort * 0.90 / 1.6)
    return np.clip(trigger, 0.0, 1.0)


def load_episode(dataset_root: pathlib.Path, episode: int, source: str):
    """Load one episode's command sequence from the LeRobot dataset.

    Returns (left_pos[T,7], right_pos[T,7], left_trigger[T], right_trigger[T], fps).
    Vector layout: [positions(16), velocities(16), efforts(16), ...] with each
    16-block ordered [left arm 1-7, left gripper, right arm 1-7, right gripper].
    The tail varies by dataset version (48-dim v1: nothing; 62-dim: +ee_pose(14);
    72-dim: +ee_pose(14)+chassis(10)) — the indices used here are identical in all.
    """
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    chunk = episode // info["chunks_size"]
    parquet_path = dataset_root / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode file not found: {parquet_path}")

    column = "action" if source == "action" else "observation.state"
    df = pd.read_parquet(parquet_path, columns=[column])
    data = np.stack(df[column].to_numpy()).astype(np.float32)  # [T, 62]

    left_pos = data[:, 0:7]
    right_pos = data[:, 8:15]
    left_trigger = gripper_effort_to_trigger(data[:, 39])
    right_trigger = gripper_effort_to_trigger(data[:, 47])
    return left_pos, right_pos, left_trigger, right_trigger, fps


def print_summary(left_pos, right_pos, left_trigger, right_trigger, fps, speed):
    frames = len(left_pos)
    print(f"frames={frames}  fps={fps:g}  speed={speed:g}x  duration={frames / (fps * speed):.1f}s")
    print(f"left arm  first={np.round(left_pos[0], 3).tolist()}")
    print(f"          last ={np.round(left_pos[-1], 3).tolist()}")
    print(f"right arm first={np.round(right_pos[0], 3).tolist()}")
    print(f"          last ={np.round(right_pos[-1], 3).tolist()}")
    print(f"left  trigger range=[{left_trigger.min():.3f}, {left_trigger.max():.3f}]")
    print(f"right trigger range=[{right_trigger.min():.3f}, {right_trigger.max():.3f}]")
    step = np.abs(np.diff(np.concatenate([left_pos, right_pos], axis=1), axis=0)).max()
    print(f"max per-frame joint step={step:.4f} rad")


def main():
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode on the Teleavatar robot")
    parser.add_argument("--dataset", type=pathlib.Path, required=True,
                        help="LeRobot dataset root (contains meta/ and data/)")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to replay")
    parser.add_argument("--source", choices=["action", "state"], default="action",
                        help="Replay recorded commands (action) or measured trajectory (observation.state)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (0.5 = half speed)")
    parser.add_argument("--ramp-s", type=float, default=5.0,
                        help="Seconds to ramp from current pose to the first frame before replaying")
    parser.add_argument("--start", type=int, default=0, help="First frame to replay")
    parser.add_argument("--end", type=int, default=None, help="Frame to stop at (exclusive)")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Only print the episode summary, publish nothing")
    args = parser.parse_args()

    left_pos, right_pos, left_trigger, right_trigger, fps = load_episode(args.dataset, args.episode, args.source)
    sl = slice(args.start, args.end)
    left_pos, right_pos = left_pos[sl], right_pos[sl]
    left_trigger, right_trigger = left_trigger[sl], right_trigger[sl]
    if len(left_pos) == 0:
        raise ValueError("Selected frame range is empty")

    print(f"Episode {args.episode} ({args.source}) from {args.dataset}, frames [{args.start}:{args.end or ''}]")
    print_summary(left_pos, right_pos, left_trigger, right_trigger, fps, args.speed)
    if args.dry_run:
        return

    if not args.yes:
        input("Robot will ramp to the first frame and replay. Press Enter to start (Ctrl+C to abort)... ")

    # ROS2 imports are deferred so --dry-run works without a ROS2 environment.
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32

    rclpy.init()
    node = Node("teleavatar_episode_replayer")
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

    def publish_frame(lp, rp, lt, rt):
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
        for grip, value in (("left_gripper", lt), ("right_gripper", rt)):
            msg = Float32()
            msg.data = float(value)
            pubs[grip].publish(msg)

    try:
        # Wait for current joint states (needed for the safety ramp).
        node.get_logger().info("Waiting for joint states...")
        deadline = time.monotonic() + 10.0
        while len(current) < 2:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for /left_arm/joint_states and /right_arm/joint_states")

        # Ramp from the current pose to the first frame (100 Hz, like zero.py).
        node.get_logger().info(f"Ramping to first frame over {args.ramp_s:.1f}s...")
        start_left, start_right = current["left_arm"].copy(), current["right_arm"].copy()
        t0 = time.monotonic()
        while (elapsed := time.monotonic() - t0) < args.ramp_s:
            alpha = elapsed / args.ramp_s
            publish_frame(start_left * (1 - alpha) + left_pos[0] * alpha,
                          start_right * (1 - alpha) + right_pos[0] * alpha,
                          left_trigger[0], right_trigger[0])
            rclpy.spin_once(node, timeout_sec=0)
            time.sleep(0.01)

        # Replay with absolute-time pacing (no drift accumulation).
        frames = len(left_pos)
        period = 1.0 / (fps * args.speed)
        node.get_logger().info(f"Replaying {frames} frames at {fps * args.speed:g} Hz...")
        t0 = time.monotonic()
        for i in range(frames):
            publish_frame(left_pos[i], right_pos[i], left_trigger[i], right_trigger[i])
            if i % 100 == 0:
                node.get_logger().info(f"  frame {i}/{frames}")
            sleep = t0 + (i + 1) * period - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
        node.get_logger().info("Replay finished (robot holds the last commanded pose).")
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted — stopped publishing; robot holds the last commanded pose.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
