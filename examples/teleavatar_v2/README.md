# TeleAvatar V2 — Deployment & Data Format

TeleAvatar V2 at a glance: all three cameras are stereo and are tiled into **a single RTP/H.265 video stream** (decoded with GStreamer — no ROS2 camera topics); the platform has built-in low-level control, so position commands are published directly to `/api/<arm>/joint_cmd` (**no** PD relay needed); grippers use a left/right symmetric piecewise force curve.

- **Training configs**: `pi0_teleavatar_v2` / `pi05_teleavatar_v2` / `pi0_teleavatar_v2_low_mem_finetune`
- **Data config class**: `LeRobotTeleavatarV2DataConfig`
- **Policy module**: [`teleavatar_v2_policy.py`](../../src/openpi/policies/teleavatar_v2_policy.py)
- **Dataset conversion**: [rosbag_to_dataset_TA2](https://github.com/dexteleop/rosbag_to_dataset_TA2)

For installation, the fine-tuning workflow, and the shared data format, see the [main README](../../README.md).

## 📑 Table of Contents

- [Client Dependencies](#-client-dependencies)
- [Deployment Data Flow](#-deployment-data-flow)
- [Deployment Steps](#-deployment-steps)
  - [1. Zero the Robot (Client)](#1-zero-the-robot-client)
  - [2. Start the Policy Server (Server)](#2-start-the-policy-server-server)
  - [3. Run the Robot Client (Client)](#3-run-the-robot-client-client)
- [Topics & Video Stream](#-topics--video-stream)
- [Camera Format](#-camera-format)
- [Gripper Mapping](#-gripper-mapping)
- [Additional Tools](#-additional-tools)

## 📦 Client Dependencies

Camera images arrive as an RTP/H.265 video stream (receiver implemented in [`rtp_video_interface.py`](rtp_video_interface.py)), which requires GStreamer and PyGObject. The current `environment.yml` already includes these dependencies; if your client environment was created from an older `environment.yml`, install them into the conda environment:

```bash
conda install -c conda-forge pygobject gst-python gstreamer \
    gst-plugins-base gst-plugins-good gst-plugins-bad gst-libav
```

Environment self-check (`nvidia-smi` must work and all GStreamer plugins must be present):

```bash
nvidia-smi
for e in rtph265depay h265parse nvh265dec videoconvert appsink; do
  gst-inspect-1.0 "$e" >/dev/null && echo "$e OK" || echo "$e MISSING"
done
```

## 🔄 Deployment Data Flow

```
cameras (RTP/H.265 stream) ┐
                           ├→  ros2_interface  →  main.py (env)  ──WebSocket──▶  serve_policy (policy)
joints (ROS2)              ┘                          ▲                                 │
                                                      └───────── 16-dim actions ◀───────┘
                                                      │
                                                      ▼
                                  /api/left_arm/joint_cmd,  /api/right_arm/joint_cmd  (position commands)
                                  /api/left_gripper/cmd,    /api/right_gripper/cmd    (trigger commands)
```

> `serve_policy` runs on the **server** (uv environment); `zero.py` and `main.py` run on the **client** (conda + ROS2 environment). Do **not** run V1's `arm_pd_controller.py` on V2 — it publishes to the same topics as V2's `ros2_interface` and would conflict.

## 🚀 Deployment Steps

### 1. Zero the Robot (Client)

```bash
python examples/teleavatar_v2/zero.py
```

Interpolates from the current joint positions to a preset home pose over 5 seconds (the target pose is hardcoded in the script; joint limits are read from `arm_config.yml` in the working directory, so run it from the repository root), publishing at 100 Hz to `/api/{left,right}_arm/joint_cmd` and setting `/api/fsm/enable=1` to enable the robot. It exits automatically once both arms converge.

> Zeroing must finish **before** starting `main.py` — both publish commands to `/api/{arm}/joint_cmd` and would conflict if run simultaneously. Alternatively, use [`zero_from_episode.py`](zero_from_episode.py) to reset to the starting pose of a dataset episode (see [Additional Tools](#-additional-tools)).

### 2. Start the Policy Server (Server)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_v2 \
    --policy.dir=checkpoints/pi0_teleavatar_v2/my_experiment/20000
```

`--policy.config` must match the training config of the checkpoint (the two generations differ in image cropping and gripper conversion; a mismatch fails silently). `--policy.dir` must contain `assets/<dataset_name>/norm_stats.json` (packaged automatically at training time). Default port: `8000`.

### 3. Run the Robot Client (Client)

```bash
python examples/teleavatar_v2/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --prompt "stack the three blocks"
```

Common flags (see `Args` in `main.py`): `--control-frequency` (control loop rate), `--action-horizon` (steps per action chunk), `--open-loop-horizon` (steps executed before re-querying the policy), `--prompt` (language instruction). The client performs action chunking via `ActionChunkBroker`.

## 📡 Topics & Video Stream

Topics used by `ros2_interface.py`:

**Subscribed (observations)**

| Topic                             | Type           | Purpose                    |
| --------------------------------- | -------------- | -------------------------- |
| `/left_arm/joint_states`          | `JointState`   | Left arm joint states      |
| `/right_arm/joint_states`         | `JointState`   | Right arm joint states     |

**Cameras (RTP/H.265 video stream, not ROS2)**

The S100 downsamples the head (2 eyes) + both wrist (2 eyes each) cameras, aligns them by exposure time, and tiles them into a single 2720×1280 frame, which is H.265-encoded and streamed over RTP (default port 8890, payload 96, ~45 fps). On the client, `RTPH265VideoInterface` in [`rtp_video_interface.py`](rtp_video_interface.py) decodes it with GStreamer (`nvh265dec`) and crops six fixed sub-views; `ros2_interface` maps three of them to policy inputs (this mapping has been verified on the real robot and matches the inner-eye crops used at training time):

| Policy observation key | RTP sub-view            | Size (H×W) | Description                    |
| ---------------------- | ----------------------- | ---------- | ------------------------------ |
| `head_camera`          | `head_left_eye`         | 960×960    | Head left eye                  |
| `left_color`           | `left_wrist_right_eye`  | 400×640    | Left wrist right eye (inner)   |
| `right_color`          | `right_wrist_left_eye`  | 400×640    | Right wrist left eye (inner)   |

> If the sender's tiling layout changes, update the crop coordinates in `TELEAVATAR_SPLIT_REGIONS` inside the interface accordingly; use [`test.py`](test.py) to test stream reception.

**Published (actions)**

| Topic                         | Type         | Publisher        | Purpose                            |
| ----------------------------- | ------------ | ---------------- | ---------------------------------- |
| `/api/left_arm/joint_cmd`     | `JointState` | ros2_interface   | Left arm target joint positions    |
| `/api/right_arm/joint_cmd`    | `JointState` | ros2_interface   | Right arm target joint positions   |
| `/api/left_gripper/cmd`       | `Float32`    | ros2_interface   | Left gripper trigger command       |
| `/api/right_gripper/cmd`      | `Float32`    | ros2_interface   | Right gripper trigger command      |
| `/api/fsm/enable`             | `Float32`    | ros2_interface   | Enable FSM                         |

## 📷 Camera Format

All three cameras are **side-by-side stereo**; a single eye is cropped from each before being fed to the model:

| Dataset key     | Raw resolution (dataset videos) | Model input key     | Processing at training time (`TeleavatarInputs`)      |
| --------------- | ------------------------------- | ------------------- | ------------------------------------------------------ |
| `head_camera`   | 3840×1920 (2:1 stereo)          | `base_0_rgb`        | Crop the left eye → 1920×1920 square main view          |
| `left_color`    | 2560×800 (stereo)               | `left_wrist_0_rgb`  | Crop the **right eye** → 1280×800 (inner, facing the table center) |
| `right_color`   | 2560×800 (stereo)               | `right_wrist_0_rgb` | Crop the **left eye** → 1280×800 (inner, facing the table center)  |

> The V2 head camera is mounted **right-side-up**: `rotate_head_camera=False` (the default of `LeRobotTeleavatarV2DataConfig`); **training and inference must be consistent**. The crop logic uses a `width >= 2 * height` aspect-ratio check: raw stereo frames (head 2:1, wrists 3.2:1) are cropped, while already-cropped single-eye frames (1:1, 1.6:1) are skipped automatically. At inference, the client takes the pre-split single-eye views directly from the RTP stream (head 960×960, wrists 400×640 — half the resolution of the dataset videos), which hit the skip branch; the policy side then resizes everything to 224×224.

## 🤏 Gripper Mapping

The platform accepts a trigger value in `[0, 1]` and maps it to an effort (Nm) through a piecewise linear curve (left/right symmetric; positive = opening, negative = closing, zero crossing at trigger = 0.10):

```
trigger <  0.10:  effort = +2.0 × (1 − trigger / 0.10)     # opening segment: +2.0 → 0
trigger >= 0.10:  effort = −1.6 × (trigger − 0.10) / 0.90  # closing segment:  0 → −1.6
```

In practice: fully open → 0.0, grasping → 0.6–1.0 (1.0 is the rated maximum closing force); avoid the 0.10–0.30 range, which sits near the zero crossing where the force is minimal. At training time, `TeleavatarInputs` applies the inverse of this curve to convert the dataset's gripper efforts to trigger values, and the model output is converted back after inference. The conversion happens before dataset norm-stats normalization — if you change it, you must re-run `scripts/compute_norm_stats.py`.

## 🧰 Additional Tools

- [`replay_episode.py`](replay_episode.py): Replay a dataset episode on the real robot (first eases from the current pose to frame 0, then replays at the recorded frame rate; `--dry-run` needs no ROS2 and only prints a trajectory summary):

  ```bash
  python examples/teleavatar_v2/replay_episode.py --dataset <dataset_path> --episode 0 [--speed 0.5]
  ```

- [`zero_from_episode.py`](zero_from_episode.py): Ease both arms to the pose of a given frame of an episode (frame 0 by default, `--frame -1` for the last frame), exiting once converged:

  ```bash
  python examples/teleavatar_v2/zero_from_episode.py --dataset <dataset_path> --episode 0 [--frame -1]
  ```

- [`test.py`](test.py): RTP reception / decoding / six-way split-and-save test, for checking the video pipeline before deployment.
