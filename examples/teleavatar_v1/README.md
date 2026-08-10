# TeleAvatar V1 — Deployment & Data Format

TeleAvatar V1 at a glance: cameras are published as ROS2 `FFMPEGPacket` topics (upside-down stereo head camera + two mono wrist cameras, decoded with PyAV); arm control requires a 100 Hz PD velocity relay ([`arm_pd_controller.py`](arm_pd_controller.py)); grippers use an asymmetric ±7 linear mapping.

- **Training configs**: `pi0_teleavatar_v1` / `pi05_teleavatar_v1` / `pi0_teleavatar_v1_low_mem_finetune`
- **Data config class**: `LeRobotTeleavatarV1DataConfig`
- **Policy module**: [`teleavatar_v1_policy.py`](../../src/openpi/policies/teleavatar_v1_policy.py)
- **Dataset conversion**: [rosbag_to_dataset_TA1](https://github.com/dexteleop/rosbag_to_dataset_TA1)

For installation, the fine-tuning workflow, and the shared data format, see the [main README](../../README.md).

## 📑 Table of Contents

- [Client Dependencies](#-client-dependencies)
- [Robot-Side Setup](#-robot-side-setup)
  - [1. Switch the Robot to API Mode](#1-switch-the-robot-to-api-mode)
  - [2. Topic Bridging & ROS Domain](#2-topic-bridging--ros-domain)
- [Deployment Data Flow](#-deployment-data-flow)
- [Deployment Steps](#-deployment-steps)
  - [1. Zero the Robot (Client)](#1-zero-the-robot-client)
  - [2. Start the Policy Server (Server)](#2-start-the-policy-server-server)
  - [3. Start the Low-Level Arm Controller (Client)](#3-start-the-low-level-arm-controller-client)
  - [4. Run the Robot Client (Client)](#4-run-the-robot-client-client)
- [ROS2 Topics](#-ros2-topics)
- [Camera Format](#-camera-format)
- [Gripper Mapping](#-gripper-mapping)

## 📦 Client Dependencies

Camera decoding uses PyAV (`av`) to subscribe to the `FFMPEGPacket` topics. The dependency is already included in `environment.yml` — no extra installation needed. An NVIDIA GPU on the client is recommended to enable `hevc_cuvid` hardware decoding; otherwise decoding falls back to the CPU.

## 🔌 Robot-Side Setup

Before deploying, the robot must be switched out of VR/teleoperation mode. This setup is one-time (the configuration is persisted and applied on reboot). See the TA1 user manual for details.

### 1. Switch the Robot to API Mode

Edit `/root/system_config.yaml` on the robot's main controller (RK3588):

```yaml
robot:
  meta_mode: 1            # 0 = VR/teleop, 1 = API
  left:
    enable: true
    arm_control_mode: 1   # 0 = end-effector, 1 = joint; the policy sends joint commands
  right:
    enable: true
    arm_control_mode: 1
```

The remaining fields (`enable_collision_check`, `chassis`, `adjust_pose`, etc.) can stay at their defaults — see the TA1 user manual. Then power-cycle the robot to apply the change.

> When switching back to VR mode, `arm_control_mode` must be set back to end-effector (`0`) so that IK is available.

### 2. Topic Bridging & ROS Domain

TA1 connects the robot and the client host through a built-in bridge service: state topics (joint states, cameras) are bridged **out** of the robot, and `/api` control topics are bridged **in**. Run the client with the default **`ROS_DOMAIN_ID=0`** — make sure the variable is not exported to another value in your shell (e.g. `29` left over from a TeleAvatar V2 setup).

Verify from the client host:

```bash
ros2 topic list
ros2 topic echo /right_arm/joint_states
```

## 🔄 Deployment Data Flow

```
cameras/joints (ROS2)  →  ros2_interface  →  main.py (env)  ──WebSocket──▶  serve_policy (policy)
                                                  ▲                                 │
                                                  └───────── 16-dim actions ◀───────┘
                                                  │
                                                  ▼
                              /left_arm/model_joint_cmd, /right_arm/model_joint_cmd
                              /api/left_gripper/cmd,     /api/right_gripper/cmd
                                                  │
                                                  ▼
                      arm_pd_controller (100 Hz PD)  →  /api/<arm>/joint_cmd (velocity commands)
```

> `serve_policy` runs on the **server** (uv environment); `zero.py`, `arm_pd_controller.py`, and `main.py` run on the **client** (conda + ROS2 environment). Topics between the robot and the client go through the built-in bridge service (see [Robot-Side Setup](#-robot-side-setup)).

## 🚀 Deployment Steps

The steps below assume [Robot-Side Setup](#-robot-side-setup) is complete: the robot is in API mode with joint control, and the client shell uses the default `ROS_DOMAIN_ID=0`.

### 1. Zero the Robot (Client)

```bash
python examples/teleavatar_v1/zero.py
```

Interpolates from the current joint positions to a preset home pose over 5 seconds (the target pose is hardcoded in the script; joint limits are read from `arm_config.yml` at the repository root), publishing at 100 Hz to `/api/{left,right}_arm/joint_cmd` and setting `/api/fsm/enable=1` to enable the robot. It exits automatically once both arms converge.

> Zeroing must finish **before** starting `arm_pd_controller` / `main.py` — they all publish commands to `/api/{arm}/joint_cmd` and would conflict if run simultaneously.

### 2. Start the Policy Server (Server)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_v1 \
    --policy.dir=checkpoints/pi0_teleavatar_v1/my_experiment/20000
```

`--policy.config` must match the training config of the checkpoint (the two generations differ in image cropping and gripper conversion; a mismatch fails silently). `--policy.dir` must contain `assets/<dataset_name>/norm_stats.json` (packaged automatically at training time). Default port: `8000`.

### 3. Start the Low-Level Arm Controller (Client)

The V1 platform's `/api/<arm>/joint_cmd` accepts **velocity commands**, so the target joint positions produced by the policy must be converted by a 100 Hz PD controller:

```bash
python examples/teleavatar_v1/arm_pd_controller.py
```

This node subscribes to `/{left,right}_arm/model_joint_cmd` and `/{left,right}_arm/joint_states`, computes velocities as `v = kp * (q_des - q_state)` (with a 0.5 s command timeout), and publishes to `/api/{left,right}_arm/joint_cmd`. Pass `--safe` to clip the commanded velocities to ±0.3 × the platform velocity limits.

### 4. Run the Robot Client (Client)

```bash
python examples/teleavatar_v1/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --prompt "stack the three blocks"
```

Common flags (see `Args` in `main.py`): `--control-frequency` (control loop rate), `--open-loop-horizon` (steps executed before re-querying the policy), `--prompt` (language instruction). The client performs action chunking via `ActionChunkBroker`.

## 📡 ROS2 Topics

**Subscribed (observations)**

| Topic                             | Type           | Purpose                                  |
| --------------------------------- | -------------- | ---------------------------------------- |
| `/xr_video_topic/ffmpeg`          | `FFMPEGPacket` | Head 2:1 stereo (H.265), main view       |
| `/left/color/image_raw/ffmpeg`    | `FFMPEGPacket` | Left wrist camera (H.265)                |
| `/right/color/image_raw/ffmpeg`   | `FFMPEGPacket` | Right wrist camera (H.265)               |
| `/left_arm/joint_states`          | `JointState`   | Left arm joint states                    |
| `/right_arm/joint_states`         | `JointState`   | Right arm joint states                   |

> All three camera feeds are H.265 streams; the client subscribes to the `FFMPEGPacket` topics and decodes with PyAV. The head camera frame is cropped to the left eye and rotated 180°, consistent with `rotate_head_camera=True` at training time.

**Published (actions)**

| Topic                         | Type         | Publisher           | Purpose                            |
| ----------------------------- | ------------ | ------------------- | ---------------------------------- |
| `/left_arm/model_joint_cmd`   | `JointState` | ros2_interface      | Left arm target joint positions    |
| `/right_arm/model_joint_cmd`  | `JointState` | ros2_interface      | Right arm target joint positions   |
| `/api/left_gripper/cmd`       | `Float32`    | ros2_interface      | Left gripper command               |
| `/api/right_gripper/cmd`      | `Float32`    | ros2_interface      | Right gripper command              |
| `/api/fsm/enable`             | `Float32`    | ros2_interface      | Enable FSM                         |
| `/api/left_arm/joint_cmd`     | `JointState` | arm_pd_controller   | Left arm velocity commands (100 Hz) |
| `/api/right_arm/joint_cmd`    | `JointState` | arm_pd_controller   | Right arm velocity commands (100 Hz) |

## 📷 Camera Format

Only the head camera is stereo (mounted upside-down); the wrist cameras are mono:

| Dataset key     | Raw resolution                       | Model input key     | Processing at training time (`TeleavatarInputs`) |
| --------------- | ------------------------------------ | ------------------- | ------------------------------------------------ |
| `head_camera`   | 4320×2160 (2:1 stereo, upside-down)  | `base_0_rgb`        | Rotate 180°, then crop the left eye → 2160×2160  |
| `left_color`    | 848×480 (mono)                       | `left_wrist_0_rgb`  | Used as-is                                       |
| `right_color`   | 848×480 (mono)                       | `right_wrist_0_rgb` | Used as-is                                       |

> The upside-down head camera requires `rotate_head_camera=True` (the default of `LeRobotTeleavatarV1DataConfig`); **training and inference must be consistent**. The crop logic checks the aspect ratio, so already-cropped square frames and 848×480 mono frames are skipped automatically.

## 🤏 Gripper Mapping

The V1 trigger↔effort mapping is a **left/right asymmetric** ±7 linear scaling; see the conversion functions in [`teleavatar_v1_policy.py`](../../src/openpi/policies/teleavatar_v1_policy.py). The conversion happens before dataset norm-stats normalization — if you change it, you must re-run `scripts/compute_norm_stats.py`.
