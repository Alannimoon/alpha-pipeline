#!/bin/bash
# 用测试集股票池（vol_top100）训练模型 + 测试集推理
#
# 配置：原因子 + market_state（7个特征），g20，ret300，3个因子池
# 输出模型：result/eval/xgb_quantile_market_state_vol100/
# 推理结果：result/eval/xgb_quantile_market_state_vol100_test/
#
# 用法：
#   bash scripts/run_xgb_vol100.sh
#   bash scripts/run_xgb_vol100.sh 2>&1 | tee logs/xgb_vol100.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

echo "========================================"
echo "vol_top100 池训练：原因子 + market_state"
echo "========================================"
echo "开始时间：$(date)"

# ── 1. 训练（vol_top100 数据，g20，ret300）────────────────────────────────────
echo ""
echo ">>> [1/2] 训练（vol_top100 池，g20，ret300，3个因子池）"
python run.py xgb_train \
    --factor-pool union intersection all \
    --n-groups 20 \
    --ret-horizon ret300 \
    --num-rounds 1000 \
    --data-workers 32 \
    --extra-features market_state \
    --test \
    --force

echo ""
echo "========================================"
echo "训练完成：$(date)"
echo "========================================"

# ── 2. 测试集推理 ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/2] 测试集推理"
OMP_NUM_THREADS=1 python run.py xgb_predict \
    --factor-pool union intersection all \
    --n-groups 20 \
    --ret-horizon ret300 \
    --test \
    --workers 8 \
    --extra-features market_state_vol100

echo ""
echo "========================================"
echo "全部完成：$(date)"
echo "========================================"
