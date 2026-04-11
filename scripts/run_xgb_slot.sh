#!/bin/bash
# 分时段 XGBoost：逐时段完成训练 + 全量推理 + 验证集推理，最后统一汇总
#
# 每个时段跑完即可查看该时段结果，不需要等全部完成
#
# 用法（tmux 里执行）：
#   tmux new -s xgb_slot
#   bash run_xgb_slot.sh 2>&1 | tee logs/xgb_slot.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

DATA_WORKERS=32
PRED_WORKERS=8
NUM_ROUNDS=1000

echo "========================================"
echo "分时段 XGBoost 开始：$(date)"
echo "========================================"

for SLOT in open morning afternoon close; do
    echo ""
    echo "════════════════════════════════════════"
    echo ">>> slot=${SLOT} 开始：$(date)"
    echo "════════════════════════════════════════"

    # ── 1. 训练 ──────────────────────────────────────────────────────────────
    echo ""
    echo "--- [${SLOT}] 训练 ---"
    python run.py xgb_train \
        --slot         ${SLOT} \
        --factor-pool  union intersection all \
        --n-groups     10 20 \
        --ret-horizon  ret100 ret200 ret300 \
        --num-rounds   ${NUM_ROUNDS} \
        --data-workers ${DATA_WORKERS}

    # ── 2. 全量推理 ──────────────────────────────────────────────────────────
    echo ""
    echo "--- [${SLOT}] 全量推理 ---"
    python run.py xgb_predict \
        --slot        ${SLOT} \
        --factor-pool union intersection all \
        --n-groups    10 20 \
        --ret-horizon ret100 ret200 ret300 \
        --workers     ${PRED_WORKERS}

    # ── 3. 验证集推理 ────────────────────────────────────────────────────────
    echo ""
    echo "--- [${SLOT}] 验证集推理 ---"
    python run.py xgb_predict \
        --slot        ${SLOT} \
        --factor-pool union intersection all \
        --n-groups    10 20 \
        --ret-horizon ret100 ret200 ret300 \
        --workers     ${PRED_WORKERS} \
        --val-only

    echo ""
    echo ">>> slot=${SLOT} 完成：$(date)"
    echo "════════════════════════════════════════"
done

# ── 4. 汇总（所有时段跑完后）────────────────────────────────────────────────
echo ""
echo ">>> 生成 PnL 汇总"
python pnl_summary.py

echo ""
echo "========================================"
echo "全部完成：$(date)"
echo "========================================"
