#!/bin/bash
# 新特征完整训练+推理
#
# 12个模型训练：
#   A500  × {+7, +23} × {union, intersection, all}  = 6个
#   vol100 × {+7, +23} × {union, intersection, all}  = 6个
#
# 推理（每个模型做验证集+测试集，共18次）：
#   A500  模型：验证集（A500全年val）+ 测试集（43天）× 2套特征 × 3池 = 12次
#   vol100 模型：验证集（vol100全年val）+ 测试集（43天）× 2套特征 × 3池 = 12次
#   合计：24次推理（每个模型都做验证集+测试集）
#
# 用法：
#   bash scripts/run_xgb_new_features.sh
#   bash scripts/run_xgb_new_features.sh 2>&1 | tee logs/xgb_new_features.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

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
        --pool a500 \
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
        --pool test \
        --workers 8 \
        --extra-features "${EXTRA}"
done

# ══════════════════════════════════════════════════════════════════════════════
# vol100 池：训练 + 验证集推理 + 测试集推理
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════"
echo "vol100 池 — 训练 + 推理"
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
        --pool vol100 \
        --force

    echo ""
    echo ">>> vol100 验证集推理  model-tag=${MODEL_TAG}"
    OMP_NUM_THREADS=1 python run.py xgb_predict \
        --factor-pool union intersection all \
        --n-groups 20 \
        --ret-horizon ret300 \
        --pool vol100 \
        --val-only \
        --workers 8 \
        --extra-features "${EXTRA}" \
        --model-tag "${MODEL_TAG}"

    echo ""
    echo ">>> vol100 测试集推理  model-tag=${MODEL_TAG}"
    OMP_NUM_THREADS=1 python run.py xgb_predict \
        --factor-pool union intersection all \
        --n-groups 20 \
        --ret-horizon ret300 \
        --pool test \
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
