# openpi（灵御双臂机器人）

> 📖 上游通用 openpi 文档（PyTorch 支持、模型清单、排错等）见 [README_openpi.md](README_openpi.md)。

该代码仓库基于 [openpi](https://github.com/Physical-Intelligence/openpi)，面向灵御（Teleavatar）双臂机器人，包含开源基础模型、用于微调训练的程序，以及在真机上通过 ROS2 部署推理的完整链路。

提供以下基础 VLA 模型权重用于微调训练：

| 基础模型   | 用途 | 描述                                                                       | 检查点路径                                  |
| ---------- | ---- | -------------------------------------------------------------------------- | ------------------------------------------- |
| $\pi_0$    | 微调 | [$\pi_0$ 基础模型](https://www.physicalintelligence.company/blog/pi0)      | `gs://openpi-assets/checkpoints/pi0_base`   |
| $\pi_{05}$ | 微调 | [$\pi_{0.5}$ 基础模型](https://www.physicalintelligence.company/blog/pi05) | `gs://openpi-assets/checkpoints/pi05_base`  |


## 系统要求

要运行本仓库中的模型，需要配备至少以下规格的 NVIDIA GPU。当前训练脚本尚不支持多节点训练。

| 模式             | 所需内存 | 示例 GPU           |
| ---------------- | -------- | ------------------ |
| 推理             | > 8 GB   | RTX 4090           |
| 微调（LoRA）     | > 22 GB  | RTX 4090 / A100    |
| 微调（完整）     | > 70 GB  | A100 (80GB) / H100 |

真机部署还需要：

- 已安装 ROS2（rclpy）的运行环境，以及 `ffmpeg_image_transport_msgs` 消息包；
- 用于解码相机 H.265 码流的 PyAV，推荐带 NVIDIA 硬件解码（`hevc_cuvid`），否则会回退到 CPU 解码。


## 环境安装

克隆本仓库时，请确保更新子模块。

```bash
git clone https://github.com/zhou-yh19/openpi.git
```

使用 [uv](https://docs.astral.sh/uv/) 来管理 Python 依赖。
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者如果没有 curl，可以使用 wget
wget -qO- https://astral.sh/uv/install.sh | sh
```

安装 uv 后，运行以下命令来设置环境。

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：`GIT_LFS_SKIP_SMUDGE=1` 是为了拉取 LeRobot 作为依赖项所必需的。


## 数据格式说明

理解灵御机器人的观测/动作约定，对配置训练和部署都很重要。相关实现位于
[`src/openpi/policies/teleavatar_policy.py`](src/openpi/policies/teleavatar_policy.py)。

### 相机

机器人提供 3 路相机，在送入模型前会映射到 $\pi_0$ 的三个图像输入：

| 数据集 / ROS 来源              | 原始分辨率                  | 模型输入键          | 处理方式                                       |
| ------------------------------ | --------------------------- | ------------------- | ---------------------------------------------- |
| `head_camera`（`/xr_video_topic`） | 2:1 双目立体（如 2160×4320） | `base_0_rgb`        | 先旋转 180°，再裁剪左眼 → 方形主视角           |
| `left_color`（`/left/...`）     | 480×848                     | `left_wrist_0_rgb`  | 原样使用                                        |
| `right_color`（`/right/...`）   | 480×848                     | `right_wrist_0_rgb` | 原样使用                                        |

> 头部相机在官方发布的机器人上为**倒装**，因此训练数据和推理数据都是倒着的，需要旋转 180°
> 后再裁剪左眼。该行为由配置项 `rotate_head_camera` 控制，**训练和推理必须保持一致**。
> 裁剪逻辑带有 `width == 2 * height` 的形状判断：若帧已经是方形（即在推理端已被裁剪），则自动跳过，
> 不会重复处理。

### 状态（observation/state，48 维）

数据集中存储 48 维本体感觉状态，布局为 `[位置(16), 速度(16), 力矩(16)]`，每个 16 维块内部布局一致：

```
[左臂关节 1-7, 左夹爪, 右臂关节 1-7, 右夹爪]
```

模型实际只使用其中的 **14 维关节位置**：左臂位置(7) + 右臂位置(7)。

### 动作（action，16 维）

模型输出 16 维动作：

```
[左臂关节位置(7), 左夹爪力矩(1), 右臂关节位置(7), 右夹爪力矩(1)]
```

夹爪力矩在进入模型前会被归一化到 `[0, 1]` 控制器区间（`TeleavatarInputs`），推理输出后再反归一化为力矩
（`TeleavatarOutputs`）。**注意**：由于该归一化发生在数据集 norm_stats 归一化之前，若修改该逻辑，
必须重新运行 `scripts/compute_norm_stats.py`。


## 模型微调

### 1. 下载 base_model

提前将 `pi0_base`（或 `pi05_base`）放入到 `~/.cache/openpi/openpi-assets/checkpoints/` 目录中。
训练时也可由 `weight_loader` 直接从 `gs://openpi-assets/...` 拉取。


### 2. 把遥操作数据转换为 LeRobot 数据集

使用代码库 [rosbag_to_lerobot](https://github.com/dexteleop/rosbag_to_lerobot) 将 rosbag 转换为 LeRobot 数据集。
转换后的数据集需包含上文 [数据格式说明](#数据格式说明) 中描述的相机键、48 维状态和 `action` 序列。


### 3. 计算训练集归一化参数

在开始训练之前，需要计算训练数据的归一化统计信息。使用训练配置名称运行以下脚本。

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_teleavatar
```

生成的 `norm_stats.json` 会写入 `./assets/<config_name>/<repo_id>/` 目录下
（例如 `./assets/pi0_teleavatar/<your_dataset>/norm_stats.json`）。训练时会自动读取，
并在保存检查点时一并打包进检查点的 `assets/<repo_id>/` 目录，供推理时使用。


### 4. 选择并配置训练参数

本仓库在 [`src/openpi/training/config.py`](src/openpi/training/config.py) 中预置了 3 个灵御训练配置，
可根据显存和需求选择：

| 配置名                          | 基础模型 | 微调方式 | batch_size | 备注                              |
| ------------------------------- | -------- | -------- | ---------- | --------------------------------- |
| `pi0_teleavatar`                | pi0      | 完整微调 | 64         | 通用首选                          |
| `pi05_teleavatar`               | pi05     | 完整微调 | 64         | 余弦学习率 + EMA                  |
| `pi0_teleavatar_low_mem_finetune` | pi0    | LoRA     | 16         | 低显存，冻结主干、关闭 EMA        |

编辑对应的 `TrainConfig` 来调整训练参数。以 `pi0_teleavatar` 为例：

```python
TrainConfig(
    name="pi0_teleavatar",
    model=pi0_config.Pi0Config(
        action_dim=32,      # 保持 32 以匹配 pi0_base 预训练权重
        action_horizon=30,  # 每次预测 30 步动作
    ),
    data=LeRobotTeleavatarDataConfig(
        repo_id="path-to-dataset",      # 替换为本地 LeRobot 数据集路径
        base_config=DataConfig(
            prompt_from_task=True,       # 从 LeRobot task 中读取语言指令
            action_sequence_keys=("action",),
        ),
        use_delta_joint_actions=False,   # 使用绝对关节位置（非增量）
        rotate_head_camera=True,         # 官方机器人头部相机倒装，需旋转 180°
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    batch_size=64,
    num_train_steps=20_000,
),
```

关键参数说明：

- `repo_id`：本地 LeRobot 数据集路径，**必须修改**。
- `prompt_from_task`：为 `True` 时，从数据集 `meta.tasks` 中按 `task_index` 注入语言指令；
  为 `False` 时所有样本将使用固定占位指令，语言通道失效。
- `rotate_head_camera`：是否在裁剪左眼前旋转 180°，需与数据采集时的相机朝向一致（官方机器人为 `True`）。
- `use_delta_joint_actions`：是否对关节位置使用增量动作（夹爪始终为绝对值），默认 `False`。

可通过 `checkpoint_base_dir`、`overwrite`、`resume`、`wandb_enabled` 等字段控制检查点保存位置与训练行为。


### 5. 运行训练脚本

现在可以使用以下命令启动训练。

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar --exp-name=my_experiment
```

- `--exp-name`：实验名称，用于区分不同设置下微调后的权重保存路径。按上面命令微调后，
  权重保存在 `<checkpoint_base_dir>/pi0_teleavatar/my_experiment/<step>`。
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`：允许 JAX 使用高达 90% 的 GPU 内存（默认 75%），以最大化显存利用。

训练过程可在 Weights & Biases 仪表板上监控（需 `wandb_enabled=True`）。


## 推理部署（真机）

部署采用 **策略服务端 + 机器人客户端** 的架构：策略服务端加载检查点并通过 WebSocket 提供推理；
机器人端通过 ROS2 采集观测、向服务端请求动作，并下发到底层控制器。

整体数据流：

```
相机/关节 (ROS2)  →  ros2_interface  →  main.py(env)  ──WebSocket──▶  serve_policy(策略)
                                              ▲                              │
                                              └──────── 16 维动作 ◀──────────┘
                                              │
                                              ▼
                          /left_arm/model_joint_cmd, /right_arm/model_joint_cmd
                          /api/left_gripper/cmd,      /api/right_gripper/cmd
                                              │
                                              ▼
                  arm_pd_controller (100Hz PD)  →  /api/<arm>/joint_cmd (速度指令)
```

### 1. 启动策略服务端

在一个终端中加载微调后的检查点并启动 WebSocket 策略服务端：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar \
    --policy.dir=checkpoints/pi0_teleavatar/my_experiment/20000
```

- `--policy.config`：训练时使用的配置名（如 `pi0_teleavatar` / `pi05_teleavatar`）。
- `--policy.dir`：检查点目录，其中需包含 `assets/<repo_id>/norm_stats.json`（训练时自动打包）。
- 默认监听端口为 `8000`，可用 `--port` 修改。

### 2. 启动底层臂控制器

策略输出的是目标**关节位置**，需要由一个 100Hz 的 PD 控制器转换为底层速度指令。
在机器人端启动 [`arm_pd_controller.py`](examples/teleavatar/arm_pd_controller.py)：

```bash
python examples/teleavatar/arm_pd_controller.py
```

该节点订阅 `/{left,right}_arm/model_joint_cmd` 与 `/{left,right}_arm/joint_states`，
按 `v = kp * (q_des - q_state)` 计算速度（带限幅与 0.5s 指令超时保护），
发布到 `/api/{left,right}_arm/joint_cmd`。

### 3. 运行机器人客户端

在另一个终端启动机器人控制主程序 [`main.py`](examples/teleavatar/main.py)：

```bash
python examples/teleavatar/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --prompt "stack the three blocks"
```

常用参数（见 `main.py` 中的 `Args`）：

| 参数                  | 默认值  | 说明                                       |
| --------------------- | ------- | ------------------------------------------ |
| `--remote-host`       | 0.0.0.0 | 策略服务端 IP                              |
| `--remote-port`       | 8000    | 策略服务端端口                            |
| `--control-frequency` | 20.0    | 控制循环频率（Hz）                         |
| `--action-horizon`    | 30      | 策略每次返回的动作步数                     |
| `--open-loop-horizon` | 24      | 重新请求策略前先执行的动作步数            |
| `--prompt`            | —       | 语言指令                                   |

主程序通过 `ActionChunkBroker` 做动作分块：每次执行 `open_loop_horizon` 步后再向策略请求新一段动作块。

### ROS2 话题一览

`ros2_interface.py` 与 `arm_pd_controller.py` 涉及的话题如下：

**订阅（观测）**

| 话题                              | 类型                | 用途                          |
| --------------------------------- | ------------------- | ----------------------------- |
| `/xr_video_topic/ffmpeg`          | `FFMPEGPacket`      | 头部 2:1 双目（H.265），主视角 |
| `/left/color/image_raw/ffmpeg`    | `FFMPEGPacket`      | 左相机（H.265）               |
| `/right/color/image_raw/ffmpeg`   | `FFMPEGPacket`      | 右相机（H.265）               |
| `/left_arm/joint_states`          | `JointState`        | 左臂关节状态                  |
| `/right_arm/joint_states`         | `JointState`        | 右臂关节状态                  |

> 客户端直接订阅 H.265 的 `FFMPEGPacket` 码流并用 PyAV 解码（绕过 `ffmpeg_image_transport` 的
> republish 节点）；头部相机在 GPU 解码时即硬件缩放，再裁剪左眼 + 旋转 180° 得到 224×224，
> 与训练时 `rotate_head_camera=True` 的处理保持一致。注意 `/head/...` 是胸口相机，**不是**模型的头部输入。

**发布（动作）**

| 话题                          | 类型         | 发布者              | 用途                       |
| ----------------------------- | ------------ | ------------------- | -------------------------- |
| `/left_arm/model_joint_cmd`   | `JointState` | ros2_interface      | 左臂目标关节位置           |
| `/right_arm/model_joint_cmd`  | `JointState` | ros2_interface      | 右臂目标关节位置           |
| `/api/left_gripper/cmd`       | `Float32`    | ros2_interface      | 左夹爪指令                 |
| `/api/right_gripper/cmd`      | `Float32`    | ros2_interface      | 右夹爪指令                 |
| `/api/fsm/enable`             | `Float32`    | ros2_interface      | 使能 FSM                   |
| `/api/left_arm/joint_cmd`     | `JointState` | arm_pd_controller   | 左臂速度指令（100Hz）      |
| `/api/right_arm/joint_cmd`    | `JointState` | arm_pd_controller   | 右臂速度指令（100Hz）      |


## 关于归一化文件

训练保存检查点时，`norm_stats.json` 会被自动打包进检查点的 `assets/<repo_id>/` 目录，
策略服务端启动时会从该路径加载，正常情况下无需手动处理。

若需将检查点迁移到其他机器、或检查点中缺失该文件，请手动将第 3 步生成的
`norm_stats.json`（位于 `./assets/<config_name>/<repo_id>/`）复制到检查点对应的
`assets/<repo_id>/` 目录下，再进行部署。
