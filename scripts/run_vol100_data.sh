#!/usr/bin/env bash
# vol100 全年数据准备：提取 → 重采样 → 清洗 → base → 因子
#
# 股票池：config/vol_top100.csv（100只高波动股票）
# 日期范围：2025全年
# 输出：data/vol100/（提取）、result/vol100/（后续各阶段）
#
# 用法：
#   bash scripts/run_vol100_data.sh
#   bash scripts/run_vol100_data.sh --workers 16

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
LOG_FILE="logs/vol100_data_${TIMESTAMP}.log"

FACTORS=(acc_mom amp_slice bap market_state mom neg_skew ofd oir pv_corr rigidity rsrs vol_turnover)

trap 'echo "[FAIL] 行号 $LINENO, 命令: $BASH_COMMAND" | tee -a "$LOG_FILE"' ERR

run_cmd() {
  echo -e "\n==================================================" | tee -a "$LOG_FILE"
  echo "[START] $(date '+%F %T')  $*" | tee -a "$LOG_FILE"
  echo "==================================================" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  echo "[ END ] $(date '+%F %T')  $*" | tee -a "$LOG_FILE"
}

echo "========================================"          | tee -a "$LOG_FILE"
echo "vol100 全年数据准备 $(date '+%F %T')"               | tee -a "$LOG_FILE"
echo "workers=$WORKERS  因子数=${#FACTORS[@]}"           | tee -a "$LOG_FILE"
echo "========================================"          | tee -a "$LOG_FILE"

# ── 1. 提取 ──────────────────────────────────────────────────────────────────
run_cmd python pipeline/extract.py \
    --universe-csv config/vol_top100.csv \
    --outdir data/vol100 \
    --date-prefix 2025 \
    --workers "$WORKERS"

# ── 2. 重采样 ─────────────────────────────────────────────────────────────────
run_cmd python run.py sample --pool vol100 --workers "$WORKERS"

# ── 3. 清洗 ──────────────────────────────────────────────────────────────────
run_cmd python run.py clean --pool vol100 --workers "$WORKERS"

# ── 4. Base ──────────────────────────────────────────────────────────────────
run_cmd python run.py base --pool vol100 --workers "$WORKERS"

# ── 5. 因子 ──────────────────────────────────────────────────────────────────
for factor in "${FACTORS[@]}"; do
  run_cmd python run.py factors --pool vol100 --factor "$factor" --workers "$WORKERS"
done

echo ""                                                  | tee -a "$LOG_FILE"
echo "========================================"          | tee -a "$LOG_FILE"
echo "[DONE] $(date '+%F %T') vol100 全年数据准备完成"    | tee -a "$LOG_FILE"
echo "  extract → data/vol100/"                          | tee -a "$LOG_FILE"
echo "  factor  → result/vol100/factor/"                 | tee -a "$LOG_FILE"
echo "日志：$LOG_FILE"                                   | tee -a "$LOG_FILE"
echo "========================================"          | tee -a "$LOG_FILE"
