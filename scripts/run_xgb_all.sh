#!/bin/bash
# 训练全部 18 个 XGBoost 模型 + 验证集推理 + PnL 汇总
#
# 用法：
#   bash run_xgb_all.sh                          # baseline（三个池 × g10/g20 × ret100/200/300）
#   bash run_xgb_all.sh --extra-features market_state  # 同上 + 追加7个市场状态特征
#
# 在 tmux 中运行（推荐）：
#   tmux new -s xgb
#   bash run_xgb_all.sh 2>&1 | tee logs/xgb_all.log
#   bash run_xgb_all.sh --extra-features market_state 2>&1 | tee logs/xgb_all_ms.log

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

# ── 解析 --extra-features 参数 ────────────────────────────────────────────────
EXTRA_FEATURES_ARG=""
EXTRA_FEATURES_TAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --extra-features)
            EXTRA_FEATURES_TAG="$2"
            EXTRA_FEATURES_ARG="--extra-features $2"
            shift 2
            ;;
        *)
            echo "未知参数：$1" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$EXTRA_FEATURES_TAG" ]]; then
    echo "========================================"
    echo "模式：额外特征集 = ${EXTRA_FEATURES_TAG}"
    echo "========================================"
else
    echo "========================================"
    echo "模式：baseline（无额外特征）"
    echo "========================================"
fi

echo "开始时间：$(date)"
echo "========================================"

# ── 1. 训练全部 18 个模型（--force 覆盖已存在）──────────────────────────────
echo ""
echo ">>> [1/3] 训练全部 18 个模型（union + intersection + all × g10/g20 × ret100/200/300）"
python run.py xgb_train \
    --factor-pool union intersection all \
    --n-groups 10 20 \
    --ret-horizon ret100 ret200 ret300 \
    --num-rounds 1000 \
    --data-workers 32 \
    --force \
    $EXTRA_FEATURES_ARG

echo ""
echo "========================================"
echo "训练完成：$(date)"
echo "========================================"

# ── 2. 验证集推理（18 个组合）────────────────────────────────────────────────
echo ""
echo ">>> [2/4] 验证集推理（全部 18 个组合）"
python run.py xgb_predict \
    --factor-pool union intersection all \
    --n-groups 10 20 \
    --ret-horizon ret100 ret200 ret300 \
    --val-only \
    --workers 8 \
    $EXTRA_FEATURES_ARG

echo ""
echo "========================================"
echo "验证集推理完成：$(date)"
echo "========================================"

# ── 3. 测试集推理（18 个组合）────────────────────────────────────────────────
echo ""
echo ">>> [3/4] 测试集推理（全部 18 个组合）"
python run.py xgb_predict \
    --factor-pool union intersection all \
    --n-groups 10 20 \
    --ret-horizon ret100 ret200 ret300 \
    --test \
    --workers 8 \
    $EXTRA_FEATURES_ARG

echo ""
echo "========================================"
echo "测试集推理完成：$(date)"
echo "========================================"

# ── 4. 汇总 PnL ───────────────────────────────────────────────────────────────
echo ""
echo ">>> [4/4] 生成 PnL 汇总"
python pnl_summary.py

echo ""
echo "========================================"
echo "全部完成：$(date)"
echo "========================================"
