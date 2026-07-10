# Teleavatar v1 部署与数据格式

v1 机器人的特点：相机走 ROS2 `FFMPEGPacket` 话题（头部倒装双目 + 单目双腕，PyAV 解码）；
臂控制需要 100Hz PD 速度中继（`arm_pd_controller.py`）；夹爪为非对称 ±7 线性映射。
训练配置：`pi0_teleavatar_v1` / `pi05_teleavatar_v1` / `pi0_teleavatar_v1_low_mem_finetune`
（数据类 `LeRobotTeleavatarV1DataConfig`，策略模块
[`teleavatar_v1_policy.py`](../../src/openpi/policies/teleavatar_v1_policy.py)）。

环境安装、模型微调流程与共通数据格式见[仓库主 README](../../README.md)。

## 客户端额外依赖

相机解码用 PyAV（`av`）订阅 FFMPEGPacket 话题，依赖已包含在 `environment.yml` 中，无需额外安装；
建议客户端配 NVIDIA GPU 以启用 `hevc_cuvid` 硬件解码，否则回退 CPU。

## 部署数据流

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

> `serve_policy` 跑在**服务端**（uv 环境）；`zero.py`、`arm_pd_controller.py`、`main.py`
> 跑在**客户端**（conda + ROS2 环境）。

## 部署步骤

### 1. 机器人归零（客户端环境）

```bash
python examples/teleavatar_v1/zero.py
```

从当前关节位置在 5 秒内插值到预设 home 位姿（目标位姿硬编码在脚本内，关节限位读自工作目录下的
`arm_config.yml`，请从仓库根目录运行），以 100Hz 发布到 `/api/{left,right}_arm/joint_cmd`，
同时发 `/api/fsm/enable=1` 使能；双臂收敛后自动退出。

> 归零必须在启动 `arm_pd_controller` / `main.py` **之前**完成——它们都向 `/api/{arm}/joint_cmd`
> 下发指令，同时运行会冲突。

### 2. 启动策略服务端（服务端环境）

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_v1 \
    --policy.dir=checkpoints/pi0_teleavatar_v1/my_experiment/20000
```

`--policy.config` 必须与检查点的训练配置一致（两代的图像裁剪和夹爪换算不同，配错会静默出错）；
`--policy.dir` 需包含 `assets/<数据集名>/norm_stats.json`（训练时自动打包）。默认端口 `8000`。

### 3. 启动底层臂控制器（客户端环境）

v1 平台的 `/api/<arm>/joint_cmd` 接收**速度指令**，策略输出的目标关节位置需要由 100Hz 的
PD 控制器转换：

```bash
python examples/teleavatar_v1/arm_pd_controller.py
```

该节点订阅 `/{left,right}_arm/model_joint_cmd` 与 `/{left,right}_arm/joint_states`，
按 `v = kp * (q_des - q_state)` 计算速度（带限幅与 0.5s 指令超时保护），发布到
`/api/{left,right}_arm/joint_cmd`。

### 4. 运行机器人客户端（客户端环境）

```bash
python examples/teleavatar_v1/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --prompt "stack the three blocks"
```

常用参数（见 `main.py` 中的 `Args`）：`--control-frequency`（控制频率）、`--action-horizon`
（每段动作步数）、`--open-loop-horizon`（重新请求策略前执行的步数）、`--prompt`（语言指令）。
主程序通过 `ActionChunkBroker` 做动作分块。

## ROS2 话题一览

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

## 相机格式

仅头部为双目（倒装），左右相机为单目：

| 数据集键        | 原始分辨率               | 模型输入键          | 训练时处理（`TeleavatarInputs`）               |
| --------------- | ------------------------ | ------------------- | ---------------------------------------------- |
| `head_camera`   | 4320×2160（2:1 双目倒装）| `base_0_rgb`        | 先旋转 180° 再裁剪左眼 → 2160×2160              |
| `left_color`    | 848×480（单目）          | `left_wrist_0_rgb`  | 原样使用                                        |
| `right_color`   | 848×480（单目）          | `right_wrist_0_rgb` | 原样使用                                        |

> 头部倒装需 `rotate_head_camera=True`（`LeRobotTeleavatarV1DataConfig` 的默认值），
> **训练和推理必须保持一致**。裁剪逻辑带宽高比判断，已裁剪的方形帧和 848×480 单目帧自动跳过。

## 夹爪映射

v1 的 trigger↔力矩为左右**不对称**的 ±7 线性缩放，实现见
[`teleavatar_v1_policy.py`](../../src/openpi/policies/teleavatar_v1_policy.py) 中的换算函数。
换算发生在数据集 norm_stats 归一化之前，若修改必须重新运行 `scripts/compute_norm_stats.py`。
