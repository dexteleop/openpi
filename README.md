# openpi（灵御双臂机器人）

> 📖 上游通用 openpi 文档（PyTorch 支持、模型清单、排错等）见 [README_openpi.md](README_openpi.md)。

该代码仓库基于 [openpi](https://github.com/Physical-Intelligence/openpi)，面向灵御（Teleavatar）双臂机器人，包含开源基础模型、用于微调训练的程序，以及在真机上通过 ROS2 部署推理的完整链路。

提供以下基础 VLA 模型权重用于微调训练：

| 基础模型 | 用途 | 描述                                                                  | 检查点路径                                 |
| -------- | ---- | ------------------------------------------------------------------- | ------------------------------------------ |
| π₀       | 微调 | [π₀ 基础模型](https://www.physicalintelligence.company/blog/pi0)    | `gs://openpi-assets/checkpoints/pi0_base`  |
| π₀.₅     | 微调 | [π₀.₅ 基础模型](https://www.physicalintelligence.company/blog/pi05) | `gs://openpi-assets/checkpoints/pi05_base` |


## 机器人版本

本仓库同时维护两代灵御机器人，两代代码**完全分开**（同上游 aloha/droid 的多机器人模式）。
本文档同时覆盖两代：共享步骤只写一遍，有差异的地方分别标注 **v1** / **v2**。

| | v1（官方发布版） | v2（当前开发版） |
| --- | --- | --- |
| 训练配置 | `pi0_teleavatar_v1` / `pi05_teleavatar_v1` / `pi0_teleavatar_v1_low_mem_finetune` | `pi0_teleavatar_v2` / `pi05_teleavatar_v2` / `pi0_teleavatar_v2_low_mem_finetune` |
| 策略模块 | [`teleavatar_v1_policy.py`](src/openpi/policies/teleavatar_v1_policy.py) | [`teleavatar_v2_policy.py`](src/openpi/policies/teleavatar_v2_policy.py) |
| 客户端 | [`examples/teleavatar_v1/`](examples/teleavatar_v1/) | [`examples/teleavatar_v2/`](examples/teleavatar_v2/) |
| 相机 | ROS2 FFMPEGPacket（头部倒装双目 + 单目双腕，PyAV 解码） | RTP/H265 拼接流（三路双目各取单眼，GStreamer 解码） |
| 臂控制 | `model_joint_cmd` + `arm_pd_controller`（100Hz PD 速度中继） | 位置指令直发 `/api/<arm>/joint_cmd` |
| 夹爪映射 | 非对称 ±7 线性曲线 | 对称分段曲线（过零点 trigger = 0.10） |


## 系统要求

要运行本仓库中的模型，需要配备至少以下规格的 NVIDIA GPU。当前训练脚本尚不支持多节点训练。

| 模式             | 所需内存 | 示例 GPU           |
| ---------------- | -------- | ------------------ |
| 推理             | > 8 GB   | RTX 4090           |
| 微调（LoRA）     | > 22 GB  | RTX 4090 / A100    |
| 微调（完整）     | > 70 GB  | A100 (80GB) / H100 |

真机部署的客户端软件依赖（ROS2 等）随 [`environment.yml`](environment.yml) 一并安装，详见
[环境安装](#环境安装)。相机解码建议客户端配 NVIDIA GPU：v1 为 H.265 ROS2 码流（PyAV /
`hevc_cuvid`，依赖随 environment.yml 安装）；v2 为 RTP/H.265 拼接视频流（不走 ROS2 话题），
需额外安装 GStreamer 并启用 `nvh265dec` 硬件解码，详见 [客户端环境](#客户端环境conda--ros2)。


## 环境安装

克隆本仓库时，请确保更新子模块。

```bash
git clone https://github.com/zhou-yh19/openpi.git
```

本项目分为**两套环境**，分别部署在不同（或相同）机器上：

| 环境       | 用途                                       | 包含          | 安装方式              |
| ---------- | ------------------------------------------ | ------------- | --------------------- |
| 服务端     | 训练、计算归一化、策略推理服务（JAX/GPU）  | openpi 本体   | `uv`                  |
| 客户端     | 真机部署，采集观测、下发动作（ROS2）       | ROS2 + 客户端 | `conda environment.yml` |

> 训练和计算归一化只需要服务端环境；真机部署时服务端跑策略服务，客户端通过 WebSocket 与之通信。

### 服务端环境（uv）

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

### 客户端环境（conda + ROS2）

真机部署侧需要 ROS2，本仓库提供了一份 conda 环境文件 [`environment.yml`](environment.yml)
（基于 robostack 的 ROS2 Humble，含 `rclpy`、`ffmpeg_image_transport_msgs`、`av` 等）。
用 conda/mamba 创建并激活：

```bash
conda env create -n teleavatar_client -f environment.yml
conda activate teleavatar_client
```

客户端脚本（`examples/teleavatar_v2/main.py` 等）依赖本仓库的 `openpi-client` 子包，在该环境中再安装一次：

```bash
pip install -e packages/openpi-client
```

**（仅 v1）** 相机解码用 PyAV（`av`）订阅 FFMPEGPacket 话题，依赖已包含在 environment.yml 中，无需额外安装。

**（仅 v2）** 相机图像通过 RTP/H.265 视频流接收（接收实现见
[examples/teleavatar_v2/rtp_video_interface.py](examples/teleavatar_v2/rtp_video_interface.py)），
需要系统级 GStreamer 与 NVIDIA 硬件解码：

```bash
sudo apt update
sudo apt install -y \
  python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  python3-opencv python3-numpy
```

环境自检（需 `nvidia-smi` 正常、各 GStreamer 插件齐全）：

```bash
nvidia-smi
for e in rtph265depay h265parse nvh265dec videoconvert appsink; do
  gst-inspect-1.0 "$e" >/dev/null && echo "$e OK" || echo "$e MISSING"
done
```


## 模型微调

### 1. 下载 base_model

提前将 `pi0_base`（或 `pi05_base`）放入到 `~/.cache/openpi/openpi-assets/checkpoints/` 目录中。
训练时也可由 `weight_loader` 直接从 `gs://openpi-assets/...` 拉取。


### 2. 把遥操作数据转换为 LeRobot 数据集

使用代码库 [rosbag_to_lerobot](https://github.com/dexteleop/rosbag_to_lerobot) 将 rosbag 转换为 LeRobot 数据集。
转换后的数据集需符合 [数据格式说明（参考）](#数据格式说明参考) 中描述的相机键、状态布局和 `action` 序列约定。


### 3. 计算训练集归一化参数

在开始训练之前，需要计算训练数据的归一化统计信息。使用训练配置名称运行以下脚本（v1 换成对应的 `*_v1` 配置名）。

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_teleavatar_v2
```

`norm_stats.json` 的存放位置取决于 `repo_id` 的写法：

- **`repo_id` 为数据集绝对路径**（本仓库推荐用法）：写入**数据集目录本身**
  （`<数据集>/norm_stats.json`），与数据放在一起；训练启动时会从该处读取。
- **`repo_id` 为裸名字**（配合 `HF_LEROBOT_HOME`）：写入 `./assets/<config_name>/<repo_id>/`
  （相对**运行命令时的工作目录**，训练时必须从同一目录启动才能读到）。

无论哪种方式，训练保存检查点时都会把 norm stats 复制进检查点的
`<step>/assets/<数据集名>/norm_stats.json`，推理时从检查点内读取，checkpoint 拷到其他机器可直接使用。
若训练启动日志出现 `Norm stats not found in ..., skipping.`，说明没有读到归一化参数，
训练会**静默地在不归一化的情况下继续**——务必确认日志中有 `Loaded norm stats from ...` 再继续。


### 4. 选择并配置训练参数

本仓库在 [`src/openpi/training/config.py`](src/openpi/training/config.py) 中为两代机器人各预置了
3 个训练配置，可根据显存和需求选择：

| 配置名（v1 / v2）                  | 基础模型 | 微调方式 | batch_size | 备注                              |
| --------------------------------- | -------- | -------- | ---------- | --------------------------------- |
| `pi0_teleavatar_v1` / `pi0_teleavatar_v2`   | pi0      | 完整微调 | 64         | 通用首选                          |
| `pi05_teleavatar_v1` / `pi05_teleavatar_v2` | pi05     | 完整微调 | 64         | 余弦学习率 + EMA                  |
| `pi0_teleavatar_v1_low_mem_finetune` / `pi0_teleavatar_v2_low_mem_finetune` | pi0 | LoRA | 16 | 低显存，冻结主干、关闭 EMA |

编辑对应的 `TrainConfig` 来调整训练参数。以 `pi0_teleavatar_v2` 为例：

```python
TrainConfig(
    name="pi0_teleavatar_v2",
    model=pi0_config.Pi0Config(
        action_dim=32,      # 保持 32 以匹配 pi0_base 预训练权重
        action_horizon=30,  # 每次预测 30 步动作
    ),
    data=LeRobotTeleavatarV2DataConfig(   # v1 用 LeRobotTeleavatarV1DataConfig
        repo_id="path-to-dataset",      # 替换为本地 LeRobot 数据集路径
        base_config=DataConfig(
            prompt_from_task=True,       # 从 LeRobot task 中读取语言指令
            action_sequence_keys=("action",),
        ),
        use_delta_joint_actions=False,   # 使用绝对关节位置（非增量）
        rotate_head_camera=False,        # v2 头部相机正装；v1 倒装需设为 True（v1 类的默认值）
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
- `rotate_head_camera`：是否在裁剪左眼前旋转 180°，需与数据采集时的相机朝向一致（v2 机器人正装为 `False`，v1 倒装机器人为 `True`）。
- `use_delta_joint_actions`：是否对关节位置使用增量动作（夹爪始终为绝对值），默认 `False`。

可通过 `checkpoint_base_dir`、`overwrite`、`resume`、`wandb_enabled` 等字段控制检查点保存位置与训练行为。


### 5. 运行训练脚本

现在可以使用以下命令启动训练（v1 换成对应的 `*_v1` 配置名）。

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar_v2 --exp-name=my_experiment
```

- `--exp-name`：实验名称，用于区分不同设置下微调后的权重保存路径。按上面命令微调后，
  权重保存在 `<checkpoint_base_dir>/pi0_teleavatar_v2/my_experiment/<step>`。
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`：允许 JAX 使用高达 90% 的 GPU 内存（默认 75%），以最大化显存利用。

训练过程可在 Weights & Biases 仪表板上监控（需 `wandb_enabled=True`）。


## 推理部署（真机）

部署采用 **策略服务端 + 机器人客户端** 的架构：策略服务端加载检查点并通过 WebSocket 提供推理；
机器人端采集观测、向服务端请求动作，并下发指令。两代的数据流：

**v2**（相机走 RTP 流；平台自带底层控制，位置指令直发，**不需要** PD 控制器）：

```
相机 (RTP/H265 拼接流) ┐
                      ├→  ros2_interface  →  main.py(env)  ──WebSocket──▶  serve_policy(策略)
关节 (ROS2)           ┘                            ▲                              │
                                              └──────── 16 维动作 ◀──────────┘
                                              │
                                              ▼
                          /api/left_arm/joint_cmd,  /api/right_arm/joint_cmd  (位置指令)
                          /api/left_gripper/cmd,    /api/right_gripper/cmd    (trigger 指令)
```

**v1**（相机走 ROS2 话题；需要额外运行 100Hz PD 速度中继）：

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

> 其中 `serve_policy` 跑在**服务端**（uv 环境）；`zero.py`、`main.py`（及其内部的
> `ros2_interface`）跑在**客户端**（conda + ROS2 环境）。`arm_pd_controller.py` 仅 v1 需要，
> **v2 上不要运行**——它与 v2 的 `ros2_interface` 发布相同话题，会冲突。

### 1. 机器人归零（客户端环境）

部署前先把双臂移动到固定的 home 位姿。在客户端运行对应版本的 `zero.py`：

```bash
python examples/teleavatar_v2/zero.py    # v1: python examples/teleavatar_v1/zero.py
```

该脚本是一个 ROS2 节点：从当前关节位置在 5 秒内插值到预设 home 位姿（左右臂目标位姿硬编码在脚本内，
关节限位读自工作目录下的 [`arm_config.yml`](arm_config.yml)，请从仓库根目录运行），以 100Hz 发布到
`/api/{left,right}_arm/joint_cmd`，同时发 `/api/fsm/enable=1` 使能；双臂收敛到容差内后自动退出。

> 归零必须在启动 `main.py`（v1 还包括 `arm_pd_controller`）**之前**完成——它们都向
> `/api/{arm}/joint_cmd` 下发指令，同时运行会冲突。

### 2. 启动策略服务端（服务端环境）

在一个终端中加载微调后的检查点并启动 WebSocket 策略服务端：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_v2 \
    --policy.dir=checkpoints/pi0_teleavatar_v2/my_experiment/20000
```

- `--policy.config`：训练时使用的配置名（如 `pi0_teleavatar_v2` / `pi05_teleavatar_v1`），
  **必须与检查点的训练配置一致**——两代的图像裁剪和夹爪换算不同，配错会静默出错。
- `--policy.dir`：检查点目录，其中需包含 `assets/<数据集名>/norm_stats.json`（训练时自动打包）。
- 默认监听端口为 `8000`，可用 `--port` 修改。

### 3.（仅 v1）启动底层臂控制器（客户端环境）

v1 平台的 `/api/<arm>/joint_cmd` 接收**速度指令**，策略输出的目标关节位置需要由 100Hz 的
PD 控制器转换。在机器人端启动
[`arm_pd_controller.py`](examples/teleavatar_v1/arm_pd_controller.py)：

```bash
python examples/teleavatar_v1/arm_pd_controller.py
```

该节点订阅 `/{left,right}_arm/model_joint_cmd` 与 `/{left,right}_arm/joint_states`，
按 `v = kp * (q_des - q_state)` 计算速度（带限幅与 0.5s 指令超时保护），
发布到 `/api/{left,right}_arm/joint_cmd`。**v2 平台自带底层控制，跳过本步骤。**

### 4. 运行机器人客户端（客户端环境）

在另一个终端启动对应版本的机器人控制主程序：

```bash
python examples/teleavatar_v2/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --prompt "stack the three blocks"    # v1: python examples/teleavatar_v1/main.py ...
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

### 话题与视频流一览（v2）

`examples/teleavatar_v2/ros2_interface.py` 涉及的话题如下：

**订阅（观测）**

| 话题                              | 类型                | 用途                          |
| --------------------------------- | ------------------- | ----------------------------- |
| `/left_arm/joint_states`          | `JointState`        | 左臂关节状态                  |
| `/right_arm/joint_states`         | `JointState`        | 右臂关节状态                  |

**相机（RTP/H.265 视频流，不走 ROS2）**

S100 将头部（2 目）+ 双腕（各 2 目）相机下采样后按曝光时间对齐拼接成一张 2720×1280 大图，
H.265 编码后经 RTP 推流（默认端口 8890，payload 96，约 45 fps）。客户端由
[`examples/teleavatar_v2/rtp_video_interface.py`](examples/teleavatar_v2/rtp_video_interface.py) 的
`RTPH265VideoInterface` 用 GStreamer（`nvh265dec`）解码，并按固定区域裁出六路子画面；
`ros2_interface` 取其中三路映射为策略输入（RTP 分割图为厂商命名，
该映射已在真机上核对、与训练裁剪逐像素一致）：

| 策略观测键     | RTP 分割图              | 尺寸 (H×W)  | 说明               |
| -------------- | ----------------------- | ----------- | ------------------ |
| `head_camera`  | `head_left_eye`         | 960×960     | 头部左目           |
| `left_color`   | `left_wrist_left_eye`   | 400×640     | 左腕（厂商命名左目）|
| `right_color`  | `right_wrist_right_eye` | 400×640     | 右腕（厂商命名右目）|

> 若发送端拼接布局变化，需同步修改接口内 `TELEAVATAR_SPLIT_REGIONS` 的裁剪坐标；
> 收流测试可用 [examples/teleavatar_v2/test.py](examples/teleavatar_v2/test.py)。

**发布（动作）**

| 话题                          | 类型         | 发布者              | 用途                       |
| ----------------------------- | ------------ | ------------------- | -------------------------- |
| `/api/left_arm/joint_cmd`     | `JointState` | ros2_interface      | 左臂目标关节位置           |
| `/api/right_arm/joint_cmd`    | `JointState` | ros2_interface      | 右臂目标关节位置           |
| `/api/left_gripper/cmd`       | `Float32`    | ros2_interface      | 左夹爪 trigger 指令        |
| `/api/right_gripper/cmd`      | `Float32`    | ros2_interface      | 右夹爪 trigger 指令        |
| `/api/fsm/enable`             | `Float32`    | ros2_interface      | 使能 FSM                   |

### 话题一览（v1）

`examples/teleavatar_v1/ros2_interface.py` 与 `arm_pd_controller.py` 涉及的话题如下：

**订阅（观测）**

| 话题                              | 类型                | 用途                          |
| --------------------------------- | ------------------- | ----------------------------- |
| `/xr_video_topic/ffmpeg`          | `FFMPEGPacket`      | 头部 2:1 双目（H.265），主视角 |
| `/left/color/image_raw/ffmpeg`    | `FFMPEGPacket`      | 左相机（H.265）               |
| `/right/color/image_raw/ffmpeg`   | `FFMPEGPacket`      | 右相机（H.265）               |
| `/left_arm/joint_states`          | `JointState`        | 左臂关节状态                  |
| `/right_arm/joint_states`         | `JointState`        | 右臂关节状态                  |

> 三路相机均为 H.265 码流，客户端订阅 `FFMPEGPacket` 后用 PyAV 解码；头部相机会裁剪左眼并旋转
> 180°，与训练时 `rotate_head_camera=True` 保持一致。

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


## 数据格式说明（参考）

理解灵御机器人的观测/动作约定，对配置训练和部署都很重要。相关实现位于
[`teleavatar_v1_policy.py`](src/openpi/policies/teleavatar_v1_policy.py) 与
[`teleavatar_v2_policy.py`](src/openpi/policies/teleavatar_v2_policy.py)。

### 相机

两代机器人均提供 3 路相机，映射到 π₀ 的三个图像输入。

**v2**：3 路均为**左右拼接双目立体**，送入模型前各裁剪出单眼视角：

| 数据集键        | 原始分辨率（数据集视频） | 模型输入键          | 训练时处理（`TeleavatarInputs`）               |
| --------------- | ------------------------ | ------------------- | ---------------------------------------------- |
| `head_camera`   | 3840×1920（2:1 双目）    | `base_0_rgb`        | 裁剪左眼 → 1920×1920 方形主视角                 |
| `left_color`    | 2560×800（双目）         | `left_wrist_0_rgb`  | 裁剪**右眼** → 1280×800（内侧，朝向桌面中央）    |
| `right_color`   | 2560×800（双目）         | `right_wrist_0_rgb` | 裁剪**左眼** → 1280×800（内侧，朝向桌面中央）    |

**v1**：仅头部为双目（倒装），左右相机为单目：

| 数据集键        | 原始分辨率               | 模型输入键          | 训练时处理（`TeleavatarInputs`）               |
| --------------- | ------------------------ | ------------------- | ---------------------------------------------- |
| `head_camera`   | 4320×2160（2:1 双目倒装）| `base_0_rgb`        | 先旋转 180° 再裁剪左眼 → 2160×2160              |
| `left_color`    | 848×480（单目）          | `left_wrist_0_rgb`  | 原样使用                                        |
| `right_color`   | 848×480（单目）          | `right_wrist_0_rgb` | 原样使用                                        |

> 头部朝向由配置项 `rotate_head_camera` 控制（v2 正装为 `False`，v1 倒装为 `True`），
> **训练和推理必须保持一致**。裁剪逻辑带有 `width >= 2 * height` 的宽高比判断：原始双目帧
> （头部 2:1、v2 左右相机 3.2:1）会被裁剪，而已裁剪的单眼帧（1:1、1.6:1）以及 v1 的 848×480
> 单目帧会自动跳过，不会重复处理。v2 推理时客户端从 RTP 拼接流中直接取已分割好的单眼视图
> （头部 960×960、腕部 400×640，为数据集视频下采样一半的分辨率），正好命中上述跳过分支，
> 随后统一由策略端缩放到 224×224。

### 状态（observation/state）

数据集存储的本体感觉状态以 48 维 `[位置(16), 速度(16), 力矩(16)]` 为基础布局（两代相同），
其后可追加任意字段（如末端位姿、底盘电机等，例如当前 v2 数据集为 62 或 72 维）；
模型只按固定索引取数，追加字段自动忽略。每个 16 维块内部布局一致：

```
[左臂关节 1-7, 左夹爪, 右臂关节 1-7, 右夹爪]
```

模型实际只使用其中的 **14 维关节位置**：左臂位置(7) + 右臂位置(7)，其索引在所有布局中一致；
末端位姿与底盘数据当前未被模型使用。

### 动作（action，16 维）

数据集中的 `action` 序列与状态同布局，训练时从中抽取 16 维；模型输出 16 维动作：

```
[左臂关节位置(7), 左夹爪力矩(1), 右臂关节位置(7), 右夹爪力矩(1)]
```

夹爪采用**力控**：平台接收 `[0, 1]` 的 trigger 值，按曲线映射为力矩（Nm），两代曲线不同。

**v2**（左右对称；正 = 张开方向，负 = 合爪方向，过零点在 trigger = 0.10）：

```
trigger <  0.10:  effort = +2.0 × (1 − trigger / 0.10)     # 张开段：+2.0 → 0
trigger >= 0.10:  effort = −1.6 × (trigger − 0.10) / 0.90  # 合爪段： 0 → −1.6
```

使用上：完全张开 → 0.0，抓取 → 0.6~1.0（1.0 为额定最大合爪力）；0.10~0.30 区间接近过零点、
力极小，应避开。

**v1**：左右**不对称**的 ±7 线性缩放，见
[`teleavatar_v1_policy.py`](src/openpi/policies/teleavatar_v1_policy.py) 中的换算函数。

训练时 `TeleavatarInputs` 用对应曲线的反函数把数据集中的夹爪力矩换算为 trigger；推理输出后
`TeleavatarOutputs` 再换算回力矩，最终由 `ros2_interface` 换算为 trigger 发布到
`/api/{left,right}_gripper/cmd`。**注意**：由于该换算发生在数据集 norm_stats 归一化之前，
若修改该逻辑，必须重新运行 `scripts/compute_norm_stats.py`。
