#!/bin/bash
# 只对验证集日期（~60 天）做推理，结果写入 xgb_quantile_val/
# 模型从 xgb_quantile/ 加载，不覆盖全量推理结果
#
# 用法：
#   bash run_xgb_predict_val.sh 2>&1 | tee logs/xgb_predict_val.log

set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "========================================"
echo "开始时间：$(date)"
echo "========================================"

echo ""
echo ">>> 验证集推理（18 个组合，~60 天，结果写入 xgb_quantile_val/）"
python run.py xgb_predict \
    --factor-pool union intersection all \
    --n-groups 10 20 \
    --ret-horizon ret100 ret200 ret300 \
    --val-only \
    --workers 8

echo ""
echo "========================================"
echo "完成：$(date)"
echo "========================================"
