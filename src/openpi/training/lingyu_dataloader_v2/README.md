# lingyu_dataloader_v2.0 的 基本使用命令

## 在当前终端选中环境
cd ~/openpi
source .venv/bin/activate

## 生成索引
cd ~/openpi/src/openpi/training/lingyu_dataloader_v2
python build_sample_idx.py

## 计算归一化
cd ~/openpi
uv run scripts/compute_norm_stats_lingyu.py --config-name pi0_teleavatar_v2_lingyu

## 模型训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar_v2_lingyu --exp-name=my_experiment
