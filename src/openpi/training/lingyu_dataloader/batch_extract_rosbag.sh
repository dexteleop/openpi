#!/usr/bin/env bash
#
# 批量转换 .mcap：逐个调用 extract_rosbag.py，输出统一放到 DATA_ROOT 下。
#
# 每个 bag 的输出目录用 .mcap 所在的上级目录名（session 名，已确认 171 条全部唯一）：
#   /mnt/minio/.../<SESSION>/rec_xxx_0.mcap  ->  $DATA_ROOT/<SESSION>/{data.parquet,*.mp4}
#
# 用法:
#   ./batch_extract_rosbag.sh                     # 转换全部，跳过已完成的
#   FORCE=1 ./batch_extract_rosbag.sh             # 重新转换（含已完成的）
#   LIST=other.txt DATA_ROOT=/tmp/out ./batch_extract_rosbag.sh
#
# 说明: 单个 bag 失败不会中断整体流程，失败列表在结尾汇总，
#       日志写在 $DATA_ROOT/logs/<SESSION>.log。

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIST="${LIST:-/home/ubuntu/rosbag_mcap_paths.txt}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/openpi/data}"
PYTHON="${PYTHON:-python}"
EXTRACT="${EXTRACT:-$SCRIPT_DIR/extract_rosbag.py}"
FORCE="${FORCE:-0}"

LOG_DIR="$DATA_ROOT/logs"

for f in "$LIST" "$EXTRACT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: 文件不存在: $f" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR"

# 读取列表：忽略空行与 # 注释行
mapfile -t BAGS < <(grep -vE '^[[:space:]]*(#|$)' "$LIST")
TOTAL=${#BAGS[@]}

echo "列表:      $LIST  ($TOTAL 个 bag)"
echo "输出根目录: $DATA_ROOT"
echo "日志目录:   $LOG_DIR"
echo "FORCE=$FORCE"
echo

n_ok=0
n_skip=0
n_fail=0
failed=()
t_all_start=$SECONDS

idx=0
for bag in "${BAGS[@]}"; do
    idx=$((idx + 1))
    session="$(basename "$(dirname "$bag")")"
    out_dir="$DATA_ROOT/$session"
    log_file="$LOG_DIR/$session.log"

    printf '[%d/%d] %s\n' "$idx" "$TOTAL" "$session"

    if [[ ! -f "$bag" ]]; then
        echo "        SKIP: .mcap 不存在 ($bag)"
        n_fail=$((n_fail + 1))
        failed+=("$session (missing mcap)")
        continue
    fi

    # data.parquet 是最后一步的产物，存在即视为上次已完整跑完
    if [[ "$FORCE" != "1" && -f "$out_dir/data.parquet" ]]; then
        echo "        SKIP: 已存在 $out_dir/data.parquet"
        n_skip=$((n_skip + 1))
        continue
    fi

    mkdir -p "$out_dir"
    t_start=$SECONDS
    if "$PYTHON" "$EXTRACT" "$bag" "$out_dir" >"$log_file" 2>&1; then
        echo "        OK  ($((SECONDS - t_start))s)  -> $out_dir"
        n_ok=$((n_ok + 1))
    else
        rc=$?
        echo "        FAIL (exit $rc, $((SECONDS - t_start))s)  见 $log_file"
        tail -n 5 "$log_file" | sed 's/^/          | /'
        n_fail=$((n_fail + 1))
        failed+=("$session (exit $rc)")
    fi
done

echo
echo "================ 汇总 ================"
echo "总计:   $TOTAL"
echo "成功:   $n_ok"
echo "跳过:   $n_skip"
echo "失败:   $n_fail"
echo "耗时:   $((SECONDS - t_all_start))s"
if ((n_fail > 0)); then
    echo
    echo "失败列表:"
    for f in "${failed[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
