#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=32
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/full_pipeline_${TIMESTAMP}.log"

FACTORS=(acc_mom amp_slice bap mom neg_skew ofd oir pv_corr rigidity rsrs)
SCORE_METHODS=(rank zscore minmax)
FACTOR_POOLS=(threshold union intersection)
N_GROUPS_LIST=(10 20)

trap 'echo "[FAIL] 行号 $LINENO, 命令: $BASH_COMMAND" | tee -a "$LOG_FILE"' ERR

run_cmd() {
  echo -e "\n==================================================" | tee -a "$LOG_FILE"
  echo "[START] $(date '+%F %T')  $*" | tee -a "$LOG_FILE"
  echo "==================================================" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  echo "[ END ] $(date '+%F %T')  $*" | tee -a "$LOG_FILE"
}

run_cmd python pipeline/extract.py --workers "$WORKERS"
run_cmd python run.py sample --workers "$WORKERS"
run_cmd python run.py clean  --workers "$WORKERS"
run_cmd python run.py base   --workers "$WORKERS"

for factor in "${FACTORS[@]}"; do
  run_cmd python run.py factors     --factor "$factor" --workers "$WORKERS"
  run_cmd python run.py cs_ic       --factor "$factor" --workers "$WORKERS"
  run_cmd python run.py ic_report   --factor "$factor"
  run_cmd python run.py cs_quantile --factor "$factor" --workers "$WORKERS"
done

for ng in "${N_GROUPS_LIST[@]}"; do
  for sm in "${SCORE_METHODS[@]}"; do
    for fp in "${FACTOR_POOLS[@]}"; do
      run_cmd python run.py multi_factor_quantile \
        --workers "$WORKERS" \
        --n-groups "$ng" \
        --score-method "$sm" \
        --factor-pool "$fp"
    done
  done
done

echo "[DONE] $(date '+%F %T') 全部完成，日志: $LOG_FILE" | tee -a "$LOG_FILE"