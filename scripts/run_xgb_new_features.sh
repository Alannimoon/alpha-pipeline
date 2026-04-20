#!/bin/bash
# 新特征完整训练+推理
#
# 12个模型训练：
#   A500  × {+7, +23} × {union, intersection, all}  = 6个
#   vol100 × {+7, +23} × {union, intersection, all}  = 6个
#
# 18次推理：
#   A500  × {+7, +23} × {验证集, 测试集}  × 3池 = 12次
#   vol100 × {+7, +23} × 测试集            × 3池 = 6次
#
# 用法：
#   bash scripts/run_xgb_new_features.sh
#   bash scripts/run_xgb_new_features.sh 2>&1 | tee logs/xgb_new_features.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

POOLS="union intersection all"
EXTRA_LIST="market_state market_state_vol_turnover"

echo "========================================"
echo "新特征完整训练+推理"
echo "开始时间：$(date)"
echo "========================================"

# ══════════════════════════════════════════════════════════════════════════════
# A500 池：训练 + 验证集推理 + 测试集推理
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════"
echo "A500 池 — 训练"
echo "════════════════════════════════"

for EXTRA in $EXTRA_LIST; do
    echo ""
    echo ">>> A500 训练  extra-features=${EXTRA}"
    python run.py xgb_train \
        --factor-pool union intersection all \
        --n-groups 20 \
        --ret-horizon ret300 \
        --num-rounds 1000 \
        --data-workers 32 \
        --extra-features "${EXTRA}" \
        --force
done

echo ""
echo "════════════════════════════════"
echo "A500 池 — 验证集推理"
echo "════════════════════════════════"

for EXTRA in $EXTRA_LIST; do
    echo ""
    echo ">>> A500 验证集推理  extra-features=${EXTRA}"
    OMP_NUM_THREADS=1 python run.py xgb_predict \
        --factor-pool union intersection all \
        --n-groups 20 \
        --ret-horizon ret300 \
        --val-only \
        --workers 8 \
        --extra-features "${EXTRA}"
done

echo ""
echo "════════════════════════════════"
echo "A500 池 — 测试集推理"
echo "════════════════════════════════"

for EXTRA in $EXTRA_LIST; do
    echo ""
    echo ">>> A500 测试集推理  extra-features=${EXTRA}"
    OMP_NUM_THREADS=1 python run.py xgb_predict \
        --factor-pool union intersection all \
        --n-groups 20 \
        --ret-horizon ret300 \
        --test \
        --workers 8 \
        --extra-features "${EXTRA}"
done

# ══════════════════════════════════════════════════════════════════════════════
# vol100 池：训练 + 测试集推理
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════"
echo "vol100 池 — 训练 + 测试集推理"
echo "════════════════════════════════"

for EXTRA in $EXTRA_LIST; do
    MODEL_TAG="${EXTRA}_vol100"
    echo ""
    echo ">>> vol100 训练  extra-features=${EXTRA}  → 模型: xgb_quantile_${MODEL_TAG}/"
    python run.py xgb_train \
        --factor-pool union intersection all \
        --n-groups 20 \
        --ret-horizon ret300 \
        --num-rounds 1000 \
        --data-workers 32 \
        --extra-features "${EXTRA}" \
        --test \
        --force

    echo ""
    echo ">>> vol100 测试集推理  model-tag=${MODEL_TAG}"
    OMP_NUM_THREADS=1 python run.py xgb_predict \
        --factor-pool union intersection all \
        --n-groups 20 \
        --ret-horizon ret300 \
        --test \
        --workers 8 \
        --extra-features "${EXTRA}" \
        --model-tag "${MODEL_TAG}"
done

echo ""
echo "========================================"
echo ">>> PnL 汇总"
echo "========================================"
python pipeline/eval/xgb_quantile/pnl_summary.py

echo ""
echo "========================================"
echo "全部完成：$(date)"
echo "========================================"
