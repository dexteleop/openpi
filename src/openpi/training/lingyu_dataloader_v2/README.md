# lingyu_dataloader_v2.0 的基本使用

## 阿里云数据桶连接
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

## 指定训练数据地址
将数据平台导出的 csv 放到项目根目录 `~/openpi/` 下（根目录下有且仅允许存在一个 csv 文件），表头必须包含 `rosbag文件夹路径` 和 `所属作业` 两列：
| rosbag文件夹路径 | 所属作业 |
| --- | --- |
| collect-algo-2601016/DF-33/D6BQ/20260814-D6BQ_20261013-132847-952679_20261013-135544-949172 | Put an apple into the bowl. |
| collect-algo-2601016/DF-33/4MI0/20260721-4MI0_20260923-100302-383890_20260923-101920-781126 | Put an apple into the bowl. |
| ... | ... |
- `rosbag文件夹路径`：OSS 桶内的对象前缀，与 `KEY_PREFIX` 拼接后在 `lingyu-wulan-pai-oss` 桶中查找其下的 `.mcap` 文件
- `所属作业`：提示词（prompt），索引构建时按 csv 原文直接使用（默认为英文）。

>.csv的文件名称可任意，但是openpi项目目录下只能有一个.csv

## 训练参数设定
在 src/openpi/training/config.py 中调整 `pi0_teleavatar_v2_lingyu` 训练配置

## （可选）挂载 /mnt/data 中的文件夹路径到 openpi
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
python build_sample_idx.py
```
在 `src/openpi/training/lingyu_dataloader_v2/iceberg_warehouse` 中生成样本索引

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