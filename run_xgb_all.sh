#!/bin/bash
# 训练全部 18 个 XGBoost 模型 + 全量推理 + PnL 汇总
# 用法：在 tmux 里执行
#   tmux new -s xgb
#   bash run_xgb_all.sh 2>&1 | tee logs/xgb_all.log

set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "========================================"
echo "开始时间：$(date)"
echo "========================================"

# ── 1. union（全部重训，--force 覆盖旧的 g10/ret100 500 轮模型）────────────────
echo ""
echo ">>> [1/3] 训练 union（6 个模型，强制重训）"
python run.py xgb_train \
    --factor-pool union \
    --n-groups 10 20 \
    --ret-horizon ret100 ret200 ret300 \
    --num-rounds 1000 \
    --data-workers 32 \
    --force

# ── 2. intersection（intersection/g10/ret100 已完成会自动跳过，训练其余 5 个）──
echo ""
echo ">>> [2/3] 训练 intersection（5 个新模型，已有的自动跳过）"
python run.py xgb_train \
    --factor-pool intersection \
    --n-groups 10 20 \
    --ret-horizon ret100 ret200 ret300 \
    --num-rounds 1000 \
    --data-workers 32

# ── 3. all（6 个模型）─────────────────────────────────────────────────────────
echo ""
echo ">>> [3/3] 训练 all（6 个模型）"
python run.py xgb_train \
    --factor-pool all \
    --n-groups 10 20 \
    --ret-horizon ret100 ret200 ret300 \
    --num-rounds 1000 \
    --data-workers 32

echo ""
echo "========================================"
echo "所有训练完成：$(date)"
echo "========================================"

# ── 4. 全量推理（18 个组合 × 243 天）────────────────────────────────────────
echo ""
echo ">>> 开始推理（全部 18 个组合）"
python run.py xgb_predict \
    --factor-pool union intersection all \
    --n-groups 10 20 \
    --ret-horizon ret100 ret200 ret300 \
    --workers 8

echo ""
echo "========================================"
echo "推理完成：$(date)"
echo "========================================"

# ── 5. 汇总 PnL ───────────────────────────────────────────────────────────
echo ""
echo ">>> 生成 PnL 汇总"
python pnl_summary.py

echo ""
echo "========================================"
echo "全部完成：$(date)"
echo "========================================"
