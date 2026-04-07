#!/bin/bash
# 分时段 XGBoost：训练 72 个模型 + 全量推理 + 验证集推理 + PnL 汇总
#
# 时段说明（stride 自动按时段长度调整，约每天 20 个截面）：
#   open      09:30-10:00  stride=10
#   morning   10:00-11:30  stride=90
#   afternoon 13:00-14:00  stride=60
#   close     14:00-14:57  stride=57
#
# 用法（tmux 里执行）：
#   tmux new -s xgb_slot
#   bash run_xgb_slot.sh 2>&1 | tee logs/xgb_slot.log

set -e
cd "$(dirname "$0")"
mkdir -p logs

POOLS="union intersection all"
GROUPS="10 20"
HORIZONS="ret100 ret200 ret300"
DATA_WORKERS=32
PRED_WORKERS=8
NUM_ROUNDS=1000

echo "========================================"
echo "分时段 XGBoost 开始：$(date)"
echo "========================================"

# ══════════════════════════════════════════════════════════════════════════════
# 第一阶段：训练（4 slot × 3 pool × 2 n_groups × 3 ret = 72 个模型）
# ══════════════════════════════════════════════════════════════════════════════

for SLOT in open morning afternoon close; do
    echo ""
    echo "════════════════════════════════════════"
    echo ">>> 训练 slot=${SLOT}  $(date)"
    echo "════════════════════════════════════════"
    python run.py xgb_train \
        --slot        ${SLOT} \
        --factor-pool ${POOLS} \
        --n-groups    ${GROUPS} \
        --ret-horizon ${HORIZONS} \
        --num-rounds  ${NUM_ROUNDS} \
        --data-workers ${DATA_WORKERS}
done

echo ""
echo "========================================"
echo "所有训练完成：$(date)"
echo "========================================"

# ══════════════════════════════════════════════════════════════════════════════
# 第二阶段：全量推理
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo ">>> 开始全量推理（4 slot × 18 组合）：$(date)"

for SLOT in open morning afternoon close; do
    echo ""
    echo "--- 全量推理 slot=${SLOT} ---"
    python run.py xgb_predict \
        --slot        ${SLOT} \
        --factor-pool ${POOLS} \
        --n-groups    ${GROUPS} \
        --ret-horizon ${HORIZONS} \
        --workers     ${PRED_WORKERS}
done

echo ""
echo "========================================"
echo "全量推理完成：$(date)"
echo "========================================"

# ══════════════════════════════════════════════════════════════════════════════
# 第三阶段：验证集推理
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo ">>> 开始验证集推理（4 slot × 18 组合）：$(date)"

for SLOT in open morning afternoon close; do
    echo ""
    echo "--- 验证集推理 slot=${SLOT} ---"
    python run.py xgb_predict \
        --slot        ${SLOT} \
        --factor-pool ${POOLS} \
        --n-groups    ${GROUPS} \
        --ret-horizon ${HORIZONS} \
        --workers     ${PRED_WORKERS} \
        --val-only
done

echo ""
echo "========================================"
echo "验证集推理完成：$(date)"
echo "========================================"

# ══════════════════════════════════════════════════════════════════════════════
# 第四阶段：PnL 汇总
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo ">>> 生成 PnL 汇总"
python pnl_summary.py

echo ""
echo "========================================"
echo "全部完成：$(date)"
echo "========================================"
