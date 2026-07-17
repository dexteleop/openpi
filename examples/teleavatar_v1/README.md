# TeleAvatar V1 — Deployment & Data Format

TeleAvatar V1 at a glance: cameras are published as ROS2 `FFMPEGPacket` topics (upside-down stereo head camera + two mono wrist cameras, decoded with PyAV); arm control requires a 100 Hz PD velocity relay ([`arm_pd_controller.py`](arm_pd_controller.py)); grippers use an asymmetric ±7 linear mapping.

- **Training configs**: `pi0_teleavatar_v1` / `pi05_teleavatar_v1` / `pi0_teleavatar_v1_low_mem_finetune`
- **Data config class**: `LeRobotTeleavatarV1DataConfig`
- **Policy module**: [`teleavatar_v1_policy.py`](../../src/openpi/policies/teleavatar_v1_policy.py)
- **Dataset conversion**: [rosbag_to_dataset_TA1](https://github.com/dexteleop/rosbag_to_dataset_TA1)

For installation, the fine-tuning workflow, and the shared data format, see the [main README](../../README.md).

## 📑 Table of Contents

- [Client Dependencies](#-client-dependencies)
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

> `serve_policy` runs on the **server** (uv environment); `zero.py`, `arm_pd_controller.py`, and `main.py` run on the **client** (conda + ROS2 environment).

## 🚀 Deployment Steps

### 1. Zero the Robot (Client)

```bash
python examples/teleavatar_v1/zero.py
```

Interpolates from the current joint positions to a preset home pose over 5 seconds (the target pose is hardcoded in the script; joint limits are read from `arm_config.yml` in the working directory, so run it from the repository root), publishing at 100 Hz to `/api/{left,right}_arm/joint_cmd` and setting `/api/fsm/enable=1` to enable the robot. It exits automatically once both arms converge.

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

This node subscribes to `/{left,right}_arm/model_joint_cmd` and `/{left,right}_arm/joint_states`, computes velocities as `v = kp * (q_des - q_state)` (with clipping and a 0.5 s command timeout), and publishes to `/api/{left,right}_arm/joint_cmd`.

### 4. Run the Robot Client (Client)

```bash
python examples/teleavatar_v1/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --prompt "stack the three blocks"
```

Common flags (see `Args` in `main.py`): `--control-frequency` (control loop rate), `--action-horizon` (steps per action chunk), `--open-loop-horizon` (steps executed before re-querying the policy), `--prompt` (language instruction). The client performs action chunking via `ActionChunkBroker`.

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
