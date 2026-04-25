# analysis/

离线分析脚本。均可在项目根目录下直接运行，路径默认指向 `result/` 和服务器原始数据。

---

## 数据集构建（一次性，已完成）

### `check_high_price_stocks.py`
检查 `vol_top100` 中的高价股。  
**输入**：`result/test/base/`  
**输出**：打印高价股列表（688 开头均价 ≥100，其余均价 ≥200）

### `build_vol100_v2.py`
构建第二测试集股票池 `vol_top100_v2`。从 vol100 中剔除高价股，从 A500 候选股按43天 `ret_fwd_300` 标准差排序补入高波动股至100只。  
**输入**：`result/test/base/`、`result/base/`、`config/vol_top100.csv`、`config/a500.csv`  
**输出**：`config/vol_top100_v2.csv`

### `build_test2_base.py`
生成 `result/test2/base/`：vol_top100_v2 × 43个测试日的 base 数据。83只原有股票从 `result/test/base/` 复用，17只新增股票从 `result/base/` 提取。  
**输入**：`result/test/base/`、`result/base/`  
**输出**：`result/test2/base/{date}.parquet`

---

## 模型评估与分析

### `plot_new_features.py`
对比4个模型配置（A500+7、A500+23、vol100+7、vol100+23）在8个时间段的分组收益曲线。  
**输入**：`result/eval/xgb_quantile_*/`  
**输出**：`result/eval/analysis/new_features_slot_pnl.png`

### `analyze_filter_threshold.py`
截面过滤阈值分析。计算两种过滤方法（`std`=截面 score 标准差，`dist`=多空两端得分差）在不同 top% 阈值下的多空收益，用于确定实盘截面过滤方案。  
**输入**：`result/eval/xgb_quantile_market_state_vol_turnover_test/{pool}/g20/ret300/`  
**输出**：`result/eval/analysis/filter_threshold_{test/test2}/` 下的14张图 + 5个 CSV  
**用法**：
```bash
python analysis/analyze_filter_threshold.py --pool test
python analysis/analyze_filter_threshold.py --pool test2
```

---

## 可视化

### `vol100_price_charts.py`
生成 vol100 全部100只股票的 tick 级价格折线图（A500归属标注）。  
**输入**：`result/test/base/`  
**输出**：`result/eval/analysis/vol100_price_charts/` 下100张 PNG + `a500_membership.txt`

### `vol100_v2_price_charts.py`
同上，针对 vol_top100_v2 股票池。  
**输入**：`result/test2/base/`  
**输出**：`result/eval/analysis/vol100_v2_price_charts/` 下100张 PNG

---

## 数据验证

### `check_limit_onesided.py`
验证深交所数据中"单边盘口缺失"与"涨跌停"的等价性。对比我们的判断条件（单边缺失）与交易所 ground truth（`LastPrice == HighLimitPrice/LowLimitPrice`），同时验证涨跌停价格准确性。直接读原始 ZIP，不经过 extract/sample 流程。  
**输入**：`/home/fund/data/{date}/mdl_6_28_0.csv.zip`（服务器原始数据）  
**输出**：`result/eval/analysis/check_limit_*.csv`（每天的匹配统计）  
**用法**：
```bash
# A500 深交所，202501~202601
python analysis/check_limit_onesided.py \
    --universe-csv config/a500.csv \
    --start-date 20250101 --end-date 20260131 \
    --workers 8 --out result/eval/analysis/check_limit_a500_full.csv

# vol_top100_v2 深交所，202512~202601
python analysis/check_limit_onesided.py \
    --universe-csv config/vol_top100_v2.csv \
    --start-date 20251201 --end-date 20260131 \
    --workers 8 --out result/eval/analysis/check_limit_v2_test.csv
```

### `show_limit_doublesided.py`
展示涨跌停边界样本：漏报（已到涨跌停价但盘口双边）和误报（单边盘口但未达涨跌停价）各5条，并计算加权中间价（wmid）。通过 `check_limit_*.csv` 精准定位目标日期，无需全量扫描。  
**输入**：`/home/fund/data/{date}/mdl_6_28_0.csv.zip` + `check_limit_*.csv`  
**输出**：打印样本到终端  
**用法**：
```bash
python analysis/show_limit_doublesided.py \
    --days 20250113 20250217 20250127 20250205 20250102 \
    --n-samples 5 --workers 8
```
