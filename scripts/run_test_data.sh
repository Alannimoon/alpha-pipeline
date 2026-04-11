#!/usr/bin/env bash
# 测试集数据准备：提取 → 重采样 → 清洗 → base → 因子
#
# 股票池：config/vol_top100.csv（100只高波动股票）
# 日期范围：202512xx 和 202601xx
# 输出：data/test/（提取）、result/test/（后续各阶段）
#
# 用法：
#   bash run_test_data.sh                  # 全量
#   bash run_test_data.sh --workers 16     # 指定并行数

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

WORKERS=32
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "未知参数：$1" >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="logs/test_data_${TIMESTAMP}.log"

FACTORS=(acc_mom amp_slice bap market_state mom neg_skew ofd oir pv_corr rigidity rsrs)

trap 'echo "[FAIL] 行号 $LINENO, 命令: $BASH_COMMAND" | tee -a "$LOG_FILE"' ERR

run_cmd() {
  echo -e "\n==================================================" | tee -a "$LOG_FILE"
  echo "[START] $(date '+%F %T')  $*" | tee -a "$LOG_FILE"
  echo "==================================================" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  echo "[ END ] $(date '+%F %T')  $*" | tee -a "$LOG_FILE"
}

echo "========================================"          | tee -a "$LOG_FILE"
echo "测试集数据准备 $(date '+%F %T')"                    | tee -a "$LOG_FILE"
echo "workers=$WORKERS  因子数=${#FACTORS[@]}"           | tee -a "$LOG_FILE"
echo "========================================"          | tee -a "$LOG_FILE"

# ── 1. 提取 ──────────────────────────────────────────────────────────────────
run_cmd python pipeline/extract.py \
    --universe-csv config/vol_top100.csv \
    --outdir data/test \
    --date-prefix 202512 202601 \
    --workers "$WORKERS"

# ── 2. 重采样 ─────────────────────────────────────────────────────────────────
run_cmd python run.py sample --test --workers "$WORKERS"

# ── 3. 清洗 ──────────────────────────────────────────────────────────────────
run_cmd python run.py clean --test --workers "$WORKERS"

# ── 4. Base ──────────────────────────────────────────────────────────────────
run_cmd python run.py base --test --workers "$WORKERS"

# ── 5. 因子 ──────────────────────────────────────────────────────────────────
for factor in "${FACTORS[@]}"; do
  run_cmd python run.py factors --test --factor "$factor" --workers "$WORKERS"
done

echo ""                                                  | tee -a "$LOG_FILE"
echo "========================================"          | tee -a "$LOG_FILE"
echo "[DONE] $(date '+%F %T') 测试集数据准备完成"         | tee -a "$LOG_FILE"
echo "  extract → data/test/"                           | tee -a "$LOG_FILE"
echo "  factor  → result/test/factor/"                  | tee -a "$LOG_FILE"
echo "日志：$LOG_FILE"                                   | tee -a "$LOG_FILE"
echo "========================================"          | tee -a "$LOG_FILE"
