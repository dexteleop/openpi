# Teleavatar v2 部署与数据格式

v2 机器人的特点：三路相机均为双目，拼接成**一路 RTP/H.265 视频流**推送（GStreamer 解码，不走
ROS2 话题）；平台自带底层控制，位置指令直发 `/api/<arm>/joint_cmd`（**不需要** PD 中继）；
夹爪为左右对称的分段力控曲线。
训练配置：`pi0_teleavatar_v2` / `pi05_teleavatar_v2` / `pi0_teleavatar_v2_low_mem_finetune`
（数据类 `LeRobotTeleavatarV2DataConfig`，策略模块
[`teleavatar_v2_policy.py`](../../src/openpi/policies/teleavatar_v2_policy.py)）。

环境安装、模型微调流程与共通数据格式见[仓库主 README](../../README.md)。

## 客户端额外依赖

相机图像通过 RTP/H.265 视频流接收（接收实现见 [`rtp_video_interface.py`](rtp_video_interface.py)），
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

## 部署数据流

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

> `serve_policy` 跑在**服务端**（uv 环境）；`zero.py`、`main.py` 跑在**客户端**
> （conda + ROS2 环境）。v1 的 `arm_pd_controller.py` 在 v2 上**不要运行**——
> 它与 v2 的 `ros2_interface` 发布相同话题，会冲突。

## 部署步骤

### 1. 机器人归零（客户端环境）

```bash
python examples/teleavatar_v2/zero.py
```

从当前关节位置在 5 秒内插值到预设 home 位姿（目标位姿硬编码在脚本内，关节限位读自工作目录下的
`arm_config.yml`，请从仓库根目录运行），以 100Hz 发布到 `/api/{left,right}_arm/joint_cmd`，
同时发 `/api/fsm/enable=1` 使能；双臂收敛后自动退出。

> 归零必须在启动 `main.py` **之前**完成——两者都向 `/api/{arm}/joint_cmd` 下发指令，
> 同时运行会冲突。也可用 [`zero_from_episode.py`](zero_from_episode.py) 复位到数据集中
> 某条 episode 的起始位姿（见下方[附加工具](#附加工具)）。

### 2. 启动策略服务端（服务端环境）

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_v2 \
    --policy.dir=checkpoints/pi0_teleavatar_v2/my_experiment/20000
```

`--policy.config` 必须与检查点的训练配置一致（两代的图像裁剪和夹爪换算不同，配错会静默出错）；
`--policy.dir` 需包含 `assets/<数据集名>/norm_stats.json`（训练时自动打包）。默认端口 `8000`。

### 3. 运行机器人客户端（客户端环境）

```bash
python examples/teleavatar_v2/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --prompt "stack the three blocks"
```

常用参数（见 `main.py` 中的 `Args`）：`--control-frequency`（控制频率）、`--action-horizon`
（每段动作步数）、`--open-loop-horizon`（重新请求策略前执行的步数）、`--prompt`（语言指令）。
主程序通过 `ActionChunkBroker` 做动作分块。

## 话题与视频流一览

`ros2_interface.py` 涉及的话题如下：

**订阅（观测）**

| 话题                              | 类型                | 用途                          |
| --------------------------------- | ------------------- | ----------------------------- |
| `/left_arm/joint_states`          | `JointState`        | 左臂关节状态                  |
| `/right_arm/joint_states`         | `JointState`        | 右臂关节状态                  |

**相机（RTP/H.265 视频流，不走 ROS2）**

S100 将头部（2 目）+ 双腕（各 2 目）相机下采样后按曝光时间对齐拼接成一张 2720×1280 大图，
H.265 编码后经 RTP 推流（默认端口 8890，payload 96，约 45 fps）。客户端由
[`rtp_video_interface.py`](rtp_video_interface.py) 的 `RTPH265VideoInterface` 用 GStreamer
（`nvh265dec`）解码，并按固定区域裁出六路子画面；`ros2_interface` 取其中三路映射为策略输入
（RTP 分割图为厂商命名，该映射已在真机上核对、与训练裁剪逐像素一致）：

| 策略观测键     | RTP 分割图              | 尺寸 (H×W)  | 说明               |
| -------------- | ----------------------- | ----------- | ------------------ |
| `head_camera`  | `head_left_eye`         | 960×960     | 头部左目           |
| `left_color`   | `left_wrist_left_eye`   | 400×640     | 左腕（厂商命名左目）|
| `right_color`  | `right_wrist_right_eye` | 400×640     | 右腕（厂商命名右目）|

> 若发送端拼接布局变化，需同步修改接口内 `TELEAVATAR_SPLIT_REGIONS` 的裁剪坐标；
> 收流测试可用 [`test.py`](test.py)。

**发布（动作）**

| 话题                          | 类型         | 发布者              | 用途                       |
| ----------------------------- | ------------ | ------------------- | -------------------------- |
| `/api/left_arm/joint_cmd`     | `JointState` | ros2_interface      | 左臂目标关节位置           |
| `/api/right_arm/joint_cmd`    | `JointState` | ros2_interface      | 右臂目标关节位置           |
| `/api/left_gripper/cmd`       | `Float32`    | ros2_interface      | 左夹爪 trigger 指令        |
| `/api/right_gripper/cmd`      | `Float32`    | ros2_interface      | 右夹爪 trigger 指令        |
| `/api/fsm/enable`             | `Float32`    | ros2_interface      | 使能 FSM                   |

## 相机格式

3 路均为**左右拼接双目立体**，送入模型前各裁剪出单眼视角：

| 数据集键        | 原始分辨率（数据集视频） | 模型输入键          | 训练时处理（`TeleavatarInputs`）               |
| --------------- | ------------------------ | ------------------- | ---------------------------------------------- |
| `head_camera`   | 3840×1920（2:1 双目）    | `base_0_rgb`        | 裁剪左眼 → 1920×1920 方形主视角                 |
| `left_color`    | 2560×800（双目）         | `left_wrist_0_rgb`  | 裁剪**右眼** → 1280×800（内侧，朝向桌面中央）    |
| `right_color`   | 2560×800（双目）         | `right_wrist_0_rgb` | 裁剪**左眼** → 1280×800（内侧，朝向桌面中央）    |

> v2 头部相机为**正装**，`rotate_head_camera=False`（`LeRobotTeleavatarV2DataConfig` 的默认值），
> **训练和推理必须保持一致**。裁剪逻辑带 `width >= 2 * height` 的宽高比判断：原始双目帧
> （头部 2:1、左右相机 3.2:1）会被裁剪，已裁剪的单眼帧（1:1、1.6:1）自动跳过。推理时客户端从
> RTP 拼接流中直接取已分割好的单眼视图（头部 960×960、腕部 400×640，为数据集视频下采样一半的
> 分辨率），正好命中跳过分支，随后统一由策略端缩放到 224×224。

## 夹爪映射

平台接收 `[0, 1]` 的 trigger 值，按分段线性曲线映射为力矩（Nm，左右对称；
正 = 张开方向，负 = 合爪方向，过零点在 trigger = 0.10）：

```
trigger <  0.10:  effort = +2.0 × (1 − trigger / 0.10)     # 张开段：+2.0 → 0
trigger >= 0.10:  effort = −1.6 × (trigger − 0.10) / 0.90  # 合爪段： 0 → −1.6
```

使用上：完全张开 → 0.0，抓取 → 0.6–1.0（1.0 为额定最大合爪力）；0.10–0.30 区间接近过零点、
力极小，应避开。训练时 `TeleavatarInputs` 用该曲线的反函数把数据集中的夹爪力矩换算为 trigger，
推理输出后再换算回来；换算发生在数据集 norm_stats 归一化之前，若修改必须重新运行
`scripts/compute_norm_stats.py`。

## 附加工具

- [`replay_episode.py`](replay_episode.py)：在真机上回放数据集中的一条 episode
  （先从当前位姿缓起到第 0 帧再按录制帧率回放；`--dry-run` 无需 ROS2，只打印轨迹摘要）：

  ```bash
  python examples/teleavatar_v2/replay_episode.py --episode 0 [--dataset <数据集路径>] [--speed 0.5]
  ```

- [`zero_from_episode.py`](zero_from_episode.py)：把双臂缓移到某条 episode 的某一帧位姿
  （默认第 0 帧，`--frame -1` 为最后一帧），到位收敛后退出：

  ```bash
  python examples/teleavatar_v2/zero_from_episode.py --episode 0 [--frame -1]
  ```

- [`test.py`](test.py)：RTP 收流/解码/六路分割保存测试，用于部署前排查视频链路。
