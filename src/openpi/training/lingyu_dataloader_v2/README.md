# lingyu_dataloader_v2.0 的基本使用

## 指定训练数据地址
将数据平台导出的 csv 放到项目根目录 `~/openpi/` 下（根目录下有且仅允许存在一个 csv 文件），表头必须包含 `rosbag文件夹路径` 和 `所属作业` 两列。

本地挂载模式（默认，`rosbag文件夹路径` 列直接写本地目录）：
| rosbag文件夹路径 | 所属作业 |
| --- | --- |
| /mnt/data/shihaoran/mcaps_place_apple_on_plate/4MI0_20260923_100302 | Put an apple into the bowl. |
| ... | ... |
- `rosbag文件夹路径`：本地 rosbag 目录（绝对路径，或 `MCAP_LOCAL_ROOT` 挂载根下的目录名），其下的 `.mcap` 文件按本地文件读取
- 可选环境变量 `export MCAP_LOCAL_ROOT="/mnt/data/shihaoran/mcaps_place_apple_on_plate"`：仅当 csv 写相对目录名时用于拼接

OSS 模式（`--mcap-source oss`）：
| rosbag文件夹路径 | 所属作业 |
| --- | --- |
| collect-algo-2601016/DF-33/D6BQ/20260814-D6BQ_20261013-132847-952679_20261013-135544-949172 | Put an apple into the bowl. |
| ... | ... |
- `rosbag文件夹路径`：OSS 桶内的对象前缀，与 `KEY_PREFIX` 拼接后在 `lingyu-wulan-pai-oss` 桶中查找其下的 `.mcap` 文件
- `所属作业`：提示词（prompt），索引构建时按 csv 原文直接使用（默认为英文）。

>.csv的文件名称可任意，但是openpi项目目录下只能有一个.csv

## 阿里云数据桶连接（仅 `--mcap-source oss` 时需要）
将下面的环境变量放到 `~/.bashrc` 中
```bash
# >>> aliyun wulan oss >>>
export ACCESS_KEY_ID="xxx"
export SECRET_ACCESS_KEY="xxx"
export S3_ENDPOINT="https://oss-cn-wulanchabu-internal.aliyuncs.com"
export REGION_NAME="cn-wulanchabu"
export BUCKET_NAME="lingyu-wulan-pai-oss"
export KEY_PREFIX="<name>"
# <<< aliyun oss <<<
```

## 训练参数设定
在 src/openpi/training/config.py 中调整 `pi0_teleavatar_v2_lingyu` 训练配置

## （可选）挂载 /mnt/data 中的文件夹路径到 openpi
目的：防止训练步数过多，中间保存模型占用大量存储导致程序中断

示例命令：
```bash
cd ~/openpi
ln -s /mnt/data/xxx/openpi_checkpoints ./checkpoints
```

## 选中环境
```bash
cd ~/openpi
source .venv/bin/activate
```

## 生成索引
```bash
cd ~/openpi/src/openpi/training/lingyu_dataloader_v2
python build_sample_idx.py                    # 默认 --mcap-source local, csv 中为本地目录
python build_sample_idx.py --mcap-source oss  # csv 中为 OSS 桶内前缀
```
在 `src/openpi/training/lingyu_dataloader_v2/iceberg_warehouse` 中生成样本索引。
索引 locations 中记录的 `mcap_url` 形式随来源不同（本地绝对路径 / 桶内 key），训练读取侧按其是否以 `/` 开头自动分派本地 pread 或对象存储读取，无需配置。

## 计算归一化
```bash
cd ~/openpi
uv run scripts/compute_norm_stats_lingyu.py --config-name pi0_teleavatar_v2_lingyu
```

## 模型训练
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar_v2_lingyu --exp-name=xxx
```
>当前在阿里云训练机上的运行情况：训练数据是从 oss://lingyu-wulan-pai-oss/ 上访问，训练过程中的模型保存在 /mnt/data 下