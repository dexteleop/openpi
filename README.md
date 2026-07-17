# openpi for TeleAvatar Dual-Arm Robots

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

This repository is based on [openpi](https://github.com/Physical-Intelligence/openpi) and targets the TeleAvatar dual-arm robots (TeleAvatar V1 and TeleAvatar V2). It provides open-source base VLA models, the fine-tuning pipeline, and the complete ROS2 stack for deploying policies on the real robots.

> 📖 For the general upstream openpi documentation (PyTorch support, full model list, troubleshooting, etc.), see [README_openpi.md](README_openpi.md).

The following base VLA model weights are provided for fine-tuning:

| Base model | Usage       | Description                                                          | Checkpoint path                            |
| ---------- | ----------- | -------------------------------------------------------------------- | ------------------------------------------ |
| π₀         | Fine-tuning | [π₀ base model](https://www.physicalintelligence.company/blog/pi0)   | `gs://openpi-assets/checkpoints/pi0_base`  |
| π₀.₅       | Fine-tuning | [π₀.₅ base model](https://www.physicalintelligence.company/blog/pi05) | `gs://openpi-assets/checkpoints/pi05_base` |

## 📑 Table of Contents

- [Robot Generations](#-robot-generations)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
  - [Server Environment (uv)](#server-environment-uv)
  - [Client Environment (conda + ROS2)](#client-environment-conda--ros2)
- [Fine-Tuning](#-fine-tuning)
  - [1. Download the Base Model](#1-download-the-base-model)
  - [2. Convert Teleoperation Data to a LeRobot Dataset](#2-convert-teleoperation-data-to-a-lerobot-dataset)
  - [3. Choose and Configure a Training Config](#3-choose-and-configure-a-training-config)
  - [4. Compute Normalization Statistics](#4-compute-normalization-statistics)
  - [5. Launch Training](#5-launch-training)
- [Real-Robot Deployment](#-real-robot-deployment)
- [Data Format (Shared)](#-data-format-shared)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🤖 Robot Generations

This repository supports both TeleAvatar generations side by side; the two codepaths are **fully separated** (mirroring the upstream aloha/droid multi-robot layout). This document covers only what the two generations **share** (installation, fine-tuning workflow, common data format). For each generation's **deployment procedure, topics, camera formats, and gripper mapping**, see its own README:

- **TeleAvatar V1**: [examples/teleavatar_v1/README.md](examples/teleavatar_v1/README.md)
- **TeleAvatar V2**: [examples/teleavatar_v2/README.md](examples/teleavatar_v2/README.md)

| | TeleAvatar V1 | TeleAvatar V2 |
| --- | --- | --- |
| Training configs | `pi0_teleavatar_v1` / `pi05_teleavatar_v1` / `pi0_teleavatar_v1_low_mem_finetune` | `pi0_teleavatar_v2` / `pi05_teleavatar_v2` / `pi0_teleavatar_v2_low_mem_finetune` |
| Policy module | [`teleavatar_v1_policy.py`](src/openpi/policies/teleavatar_v1_policy.py) | [`teleavatar_v2_policy.py`](src/openpi/policies/teleavatar_v2_policy.py) |
| Client | [`examples/teleavatar_v1/`](examples/teleavatar_v1/) | [`examples/teleavatar_v2/`](examples/teleavatar_v2/) |
| Dataset conversion | [rosbag_to_dataset_TA1](https://github.com/dexteleop/rosbag_to_dataset_TA1) | [rosbag_to_dataset_TA2](https://github.com/dexteleop/rosbag_to_dataset_TA2) |
| Cameras | ROS2 FFMPEGPacket topics (upside-down stereo head + mono wrists, PyAV decoding) | Single RTP/H.265 combined stream (three stereo cameras, one eye each, GStreamer decoding) |
| Arm control | `model_joint_cmd` + `arm_pd_controller` (100 Hz PD velocity relay) | Position commands published directly to `/api/<arm>/joint_cmd` |
| Gripper mapping | Asymmetric ±7 linear curve | Symmetric piecewise curve (zero crossing at trigger = 0.10) |

## 📋 System Requirements

Running the models in this repository requires an NVIDIA GPU with at least the following specs. The training script does not currently support multi-node training.

| Mode                | Required VRAM | Example GPU        |
| ------------------- | ------------- | ------------------ |
| Inference           | > 8 GB        | RTX 4090           |
| Fine-tuning (LoRA)  | > 22 GB       | RTX 4090 / A100    |
| Fine-tuning (full)  | > 70 GB       | A100 (80GB) / H100 |

The client-side software dependencies for real-robot deployment (ROS2, etc.) are installed together via [`environment.yml`](environment.yml) — see [Installation](#-installation). An NVIDIA GPU is recommended on the client for camera decoding; the decoding dependencies differ between the two generations, see the per-generation READMEs.

## 🔧 Installation

When cloning this repository, make sure to fetch the submodules:

```bash
git clone --recurse-submodules https://github.com/dexteleop/openpi.git
```

The project uses **two separate environments**, which may live on different (or the same) machines:

| Environment | Purpose                                                        | Contains            | Installed with          |
| ----------- | -------------------------------------------------------------- | ------------------- | ----------------------- |
| Server      | Training, norm stats, policy inference service (JAX/GPU)       | openpi itself       | `uv`                    |
| Client      | Real-robot deployment: collect observations, send actions (ROS2) | ROS2 + client code | `conda environment.yml` |

> Training and norm-stats computation only need the server environment. For real-robot deployment, the server runs the policy service and the client talks to it over WebSocket.

### Server Environment (uv)

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or, if you don't have curl, use wget
wget -qO- https://astral.sh/uv/install.sh | sh
```

Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Note: `GIT_LFS_SKIP_SMUDGE=1` is required to pull LeRobot as a dependency.

### Client Environment (conda + ROS2)

The robot side needs ROS2. This repository ships a conda environment file [`environment.yml`](environment.yml) (robostack-based ROS2 Humble, including `rclpy`, `ffmpeg_image_transport_msgs`, `av`, etc.). Create and activate it with conda/mamba:

```bash
conda env create -n teleavatar_client -f environment.yml
conda activate teleavatar_client
```

The client scripts depend on this repository's `openpi-client` subpackage; install it inside the same environment:

```bash
pip install -e packages/openpi-client
```

The camera decoding dependencies (PyAV for V1, GStreamer/PyGObject for V2) are already installed via `environment.yml`. Self-check instructions and the extra install commands for environments created from an older `environment.yml` are in the "Client Dependencies" section of each generation's README.

## 🚀 Fine-Tuning

### 1. Download the Base Model

Place `pi0_base` (or `pi05_base`) under `~/.cache/openpi/openpi-assets/checkpoints/` ahead of time. Alternatively, the `weight_loader` can pull it directly from `gs://openpi-assets/...` at training time.

### 2. Convert Teleoperation Data to a LeRobot Dataset

Each generation has its own conversion toolkit for turning rosbags into LeRobot datasets:

- **TeleAvatar V1**: [rosbag_to_dataset_TA1](https://github.com/dexteleop/rosbag_to_dataset_TA1)
- **TeleAvatar V2**: [rosbag_to_dataset_TA2](https://github.com/dexteleop/rosbag_to_dataset_TA2)

The converted dataset must follow the state layout and `action` sequence conventions described in [Data Format (Shared)](#-data-format-shared). Camera keys and resolutions are generation-specific — see the per-generation READMEs.

### 3. Choose and Configure a Training Config

[`src/openpi/training/config.py`](src/openpi/training/config.py) pre-defines 3 training configs per generation; choose based on your VRAM and needs:

| Config (V1 / V2)                   | Base model | Fine-tuning | batch_size | Notes                              |
| ---------------------------------- | ---------- | ----------- | ---------- | ---------------------------------- |
| `pi0_teleavatar_v1` / `pi0_teleavatar_v2`   | π₀        | Full        | 64         | Default choice                     |
| `pi05_teleavatar_v1` / `pi05_teleavatar_v2` | π₀.₅      | Full        | 64         | Cosine LR schedule + EMA           |
| `pi0_teleavatar_v1_low_mem_finetune` / `pi0_teleavatar_v2_low_mem_finetune` | π₀ | LoRA | 16 | Low VRAM: frozen backbone, EMA off |

Edit the corresponding `TrainConfig` to adjust training parameters. **The one field you MUST edit is `repo_id`** — point it at your converted LeRobot dataset; everything else has working defaults.

**TeleAvatar V1 example** (`pi0_teleavatar_v1`):

```python
TrainConfig(
    name="pi0_teleavatar_v1",
    model=pi0_config.Pi0Config(
        action_dim=32,      # keep 32 to match the pi0_base pretrained weights
        action_horizon=30,  # predict 30 action steps per inference
    ),
    data=LeRobotTeleavatarV1DataConfig(
        repo_id="/data/lerobot/teleavatar_v1_pick_place",  # REQUIRED: your local dataset path
        base_config=DataConfig(
            prompt_from_task=True,   # read language instructions from the LeRobot task
            action_sequence_keys=("action",),
        ),
        use_delta_joint_actions=False,  # absolute joint positions (not deltas)
        rotate_head_camera=True,        # V1 head camera is mounted upside-down
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    batch_size=64,
    num_train_steps=20_000,
),
```

**TeleAvatar V2 example** (`pi0_teleavatar_v2`):

```python
TrainConfig(
    name="pi0_teleavatar_v2",
    model=pi0_config.Pi0Config(
        action_dim=32,      # keep 32 to match the pi0_base pretrained weights
        action_horizon=30,  # predict 30 action steps per inference
    ),
    data=LeRobotTeleavatarV2DataConfig(
        repo_id="/data/lerobot/teleavatar_v2_pick_place",  # REQUIRED: your local dataset path
        base_config=DataConfig(
            prompt_from_task=True,   # read language instructions from the LeRobot task
            action_sequence_keys=("action",),
        ),
        use_delta_joint_actions=False,  # absolute joint positions (not deltas)
        rotate_head_camera=False,       # V2 head camera is mounted right-side-up
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    batch_size=64,
    num_train_steps=20_000,
),
```

**Key fields**:

- `repo_id`: Path to your local LeRobot dataset. **This is the required change.**
- `prompt_from_task`: When `True`, the language instruction is injected from the dataset's `meta.tasks` by `task_index`; when `False`, every sample uses a fixed placeholder instruction and the language channel is effectively disabled.
- `rotate_head_camera`: Whether to rotate the head camera 180° before the left-eye crop; must match the camera orientation at data-collection time (V1 upside-down → `True`, V2 right-side-up → `False`; both are the defaults of the corresponding data config class).
- `use_delta_joint_actions`: Whether to use delta actions for joint positions (grippers always stay absolute). Default `False`.

Checkpoint location and training behavior can be further controlled via `checkpoint_base_dir`, `overwrite`, `resume`, `wandb_enabled`, etc.

### 4. Compute Normalization Statistics

Before launching training, compute the normalization statistics of the training data. Run the script with the config name you just configured:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_teleavatar_v2
```

Where `norm_stats.json` is written depends on how `repo_id` is set:

- **`repo_id` is an absolute dataset path** (the recommended usage in this repository): stats are written into the **dataset directory itself** (`<dataset>/norm_stats.json`), next to the data; training reads them from there at startup.
- **`repo_id` is a bare name** (used together with `HF_LEROBOT_HOME`): stats are written to `./assets/<config_name>/<repo_id>/`, relative to **the working directory the command was run from** — training must be launched from the same directory to find them.

Either way, when training saves a checkpoint it copies the norm stats into the checkpoint's `<step>/assets/<dataset_name>/norm_stats.json`; inference reads them from inside the checkpoint, so a checkpoint copied to another machine works as-is. If the training startup log shows `Norm stats not found in ..., skipping.`, the stats were not found and training **silently continues without normalization** — make sure the log shows `Loaded norm stats from ...` before proceeding.

### 5. Launch Training

Start training with the following command (for V1, substitute the corresponding `*_v1` config name):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar_v2 --exp-name=my_experiment
```

- `--exp-name`: Experiment name, used to distinguish checkpoint save paths across different runs. With the command above, weights are saved under `<checkpoint_base_dir>/pi0_teleavatar_v2/my_experiment/<step>`.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: Lets JAX use up to 90% of GPU memory (default 75%) to maximize VRAM utilization.

Training can be monitored on the Weights & Biases dashboard (requires `wandb_enabled=True`).

## 🎮 Real-Robot Deployment

Deployment uses a **policy server + robot client** architecture: the policy server (uv environment) loads the checkpoint and serves inference over WebSocket; the robot side (conda + ROS2 environment) collects observations, requests actions from the server, and issues commands. The camera pipelines and arm control differ between the two generations — **see each generation's README for the concrete deployment steps, topic list, and troubleshooting**:

- **TeleAvatar V1**: [examples/teleavatar_v1/README.md](examples/teleavatar_v1/README.md)
- **TeleAvatar V2**: [examples/teleavatar_v2/README.md](examples/teleavatar_v2/README.md)

> General caveat: at serve time, `--policy.config` must match the training config of the checkpoint — the two generations differ in image cropping and gripper conversion, and using the wrong generation fails silently.

## 📊 Data Format (Shared)

The conventions below apply to both generations; camera formats and gripper curves differ per generation, see the per-generation READMEs. The implementations live in [`teleavatar_v1_policy.py`](src/openpi/policies/teleavatar_v1_policy.py) and [`teleavatar_v2_policy.py`](src/openpi/policies/teleavatar_v2_policy.py).

### State (`observation/state`)

The proprioceptive state stored in the dataset uses a 48-dim base layout of `[positions (16), velocities (16), efforts (16)]` (identical across both generations); arbitrary extra fields (e.g. end-effector poses, chassis motors) may be appended after it. The model reads fixed indices only, so appended fields are ignored automatically. Each 16-dim block shares the same internal layout:

```
[left arm joints 1-7, left gripper, right arm joints 1-7, right gripper]
```

The model actually uses only the **14 joint-position dims**: left arm positions (7) + right arm positions (7), whose indices are identical in all layouts.

### Action (16-dim)

The `action` sequence in the dataset shares the state layout; 16 dims are extracted at training time. The model outputs 16-dim actions:

```
[left arm joint positions (7), left gripper effort (1), right arm joint positions (7), right gripper effort (1)]
```

Grippers are **force-controlled**: the platform accepts a trigger value in `[0, 1]` and maps it to an effort (Nm) through a curve that **differs between the two generations** (see the per-generation READMEs). At training time, `TeleavatarInputs` applies the inverse of the corresponding curve to convert the dataset's gripper efforts to trigger values; at inference, `TeleavatarOutputs` converts the model output back to efforts, which `ros2_interface` finally converts to trigger values for publishing. **Note**: this conversion happens before dataset norm-stats normalization — if you change this logic, you must re-run `scripts/compute_norm_stats.py`.

## 📄 License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

Built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence (Apache 2.0).
