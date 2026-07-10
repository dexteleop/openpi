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
本文档只写两代**共享**的内容（环境安装、模型微调、共通数据格式）；各代的**部署流程、话题、
相机格式与夹爪映射**见各自的说明文档：

- **v1（官方发布版）**：[examples/teleavatar_v1/README.md](examples/teleavatar_v1/README.md)
- **v2（当前开发版）**：[examples/teleavatar_v2/README.md](examples/teleavatar_v2/README.md)

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
[环境安装](#环境安装)。相机解码建议客户端配 NVIDIA GPU，两代的解码依赖不同，见各代 README。


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

客户端脚本依赖本仓库的 `openpi-client` 子包，在该环境中再安装一次：

```bash
pip install -e packages/openpi-client
```

相机解码的额外依赖因机器人代数而异（v1 用 PyAV，已随 environment.yml 安装；v2 需系统级
GStreamer + `nvh265dec`），见各代 README 的"客户端额外依赖"一节。


## 模型微调

### 1. 下载 base_model

提前将 `pi0_base`（或 `pi05_base`）放入到 `~/.cache/openpi/openpi-assets/checkpoints/` 目录中。
训练时也可由 `weight_loader` 直接从 `gs://openpi-assets/...` 拉取。


### 2. 把遥操作数据转换为 LeRobot 数据集

使用代码库 [rosbag_to_lerobot](https://github.com/dexteleop/rosbag_to_lerobot) 将 rosbag 转换为 LeRobot 数据集。
转换后的数据集需符合 [数据格式说明（共通）](#数据格式说明共通) 中描述的状态布局和 `action`
序列约定，相机键与分辨率见各代 README。


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
        rotate_head_camera=False,        # 头部相机朝向，两代默认值不同，见各代 README
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
- `rotate_head_camera`：是否在裁剪左眼前旋转 180°，需与数据采集时的相机朝向一致
  （v2 正装为 `False`，v1 倒装为 `True`，均为对应数据类的默认值）。
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

部署采用 **策略服务端 + 机器人客户端** 的架构：策略服务端（uv 环境）加载检查点并通过 WebSocket
提供推理；机器人端（conda + ROS2 环境）采集观测、向服务端请求动作并下发指令。两代的相机链路与
臂控制方式不同，**具体部署步骤、话题一览与排查方法见各代 README**：

- **v1**：[examples/teleavatar_v1/README.md](examples/teleavatar_v1/README.md)（需要 PD 速度中继）
- **v2**：[examples/teleavatar_v2/README.md](examples/teleavatar_v2/README.md)（位置指令直发，另含
  episode 回放/复位工具）

> 通用注意：serve 时 `--policy.config` 必须与检查点的训练配置一致——两代的图像裁剪和夹爪换算
> 不同，配错代数会静默出错。


## 数据格式说明（共通）

以下约定两代通用；相机格式与夹爪曲线的差异见各代 README。相关实现位于
[`teleavatar_v1_policy.py`](src/openpi/policies/teleavatar_v1_policy.py) 与
[`teleavatar_v2_policy.py`](src/openpi/policies/teleavatar_v2_policy.py)。

### 状态（observation/state）

数据集存储的本体感觉状态以 48 维 `[位置(16), 速度(16), 力矩(16)]` 为基础布局（两代相同），
其后可追加任意字段（如末端位姿、底盘电机等，例如当前 v2 数据集为 62 或 72 维）；
模型只按固定索引取数，追加字段自动忽略。每个 16 维块内部布局一致：

```
[左臂关节 1-7, 左夹爪, 右臂关节 1-7, 右夹爪]
```

模型实际只使用其中的 **14 维关节位置**：左臂位置(7) + 右臂位置(7)，其索引在所有布局中一致。

### 动作（action，16 维）

数据集中的 `action` 序列与状态同布局，训练时从中抽取 16 维；模型输出 16 维动作：

```
[左臂关节位置(7), 左夹爪力矩(1), 右臂关节位置(7), 右夹爪力矩(1)]
```

夹爪采用**力控**：平台接收 `[0, 1]` 的 trigger 值，按曲线映射为力矩（Nm），**两代曲线不同**
（见各代 README）。训练时 `TeleavatarInputs` 用对应曲线的反函数把数据集中的夹爪力矩换算为
trigger；推理输出后 `TeleavatarOutputs` 再换算回力矩，最终由 `ros2_interface` 换算为 trigger
发布。**注意**：由于该换算发生在数据集 norm_stats 归一化之前，若修改该逻辑，必须重新运行
`scripts/compute_norm_stats.py`。
