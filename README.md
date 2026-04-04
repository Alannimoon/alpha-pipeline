# 因子评测流水线

高频市场微结构数据的 Alpha 因子研究框架，支持因子计算、IC 分析、截面分层、多因子合成，以及基于代价敏感 XGBoost 的截面分层模型。

---

## 目录

- [整体架构](#整体架构)
- [项目结构](#项目结构)
- [快速上手](#快速上手)
- [各阶段说明](#各阶段说明)
- [因子列表](#因子列表)
- [评估方法](#评估方法)
- [XGBoost 截面分层模型](#xgboost-截面分层模型)
- [全局配置](#全局配置)

---

## 整体架构

```
原始 Tick 数据（3 秒快照）
    ↓ sample     重采样到等间隔时间网格
    ↓ clean      剔除停牌日
    ↓ base       计算 WMID 价格、掩码、前向收益率
    ↓ factors    计算各 Alpha 因子
    ├─ cs_ic           截面 IC / RankIC 分析
    ├─ cs_quantile     单因子截面分层
    ├─ multi_factor    多因子 IC 加权合成分层
    └─ xgb_quantile    XGBoost 代价敏感分类分层  ← 当前重点
```

数据以日为单位处理，所有中间结果以 Parquet 格式存储在 `result/` 目录下。

---

## 项目结构

```
alpha-pipeline/
├── data/                        原始快照数据（只读）
│   └── 20250102/
│       ├── 000001.csv
│       └── ...
├── result/                      各阶段处理结果（自动生成，不入库）
│   ├── sampled/                 重采样后的 Parquet
│   ├── cleaned/                 清洗后的 Parquet
│   ├── base/                    价格 + 掩码 + 前向收益（每日一文件）
│   ├── factor/                  各因子值（按因子名分目录）
│   ├── eval/                    评估结果
│   │   ├── cs_ic/               截面 IC
│   │   ├── cs_quantile/         单因子分层
│   │   ├── multi_factor_quantile/  多因子合成分层
│   │   └── xgb_quantile/        XGBoost 分层模型
│   └── cache/                   宽表缓存
├── pipeline/
│   ├── ingest/
│   │   ├── sample.py            重采样
│   │   ├── clean.py             清洗
│   │   └── base.py              价格定义、掩码、收益率
│   ├── factor/
│   │   ├── _core.py             公共工具（窗口有效性、滚动均值）
│   │   ├── bap.py / mom.py ...  各因子计算模块
│   │   └── compute.py           因子计算编排
│   └── eval/
│       ├── _panel.py            截面数据变换、IC 计算
│       ├── ic/                  IC 分析
│       ├── quantile/
│       │   ├── single/          单因子分层
│       │   └── multi/           多因子合成分层
│       └── xgb_quantile/        XGBoost 分层模型
│           ├── dataset.py       数据加载与标注
│           ├── loss.py          代价敏感目标函数
│           ├── train.py         训练编排
│           └── predict.py       推理
├── config/
│   ├── __init__.py              全局路径与参数配置
│   ├── drop_overrides.csv       人工剔除覆盖表
│   ├── factor_pool_union.txt    并集因子池（51 个）
│   └── factor_pool_intersection.txt  交集因子池（25 个）
├── run.py                       统一命令行入口
├── run_all.sh                   批量运行脚本
└── pnl_summary.py               汇总各方法 PnL 对比
```

---

## 快速上手

### 完整流程（单日示例）

```bash
# 1. 数据准备
python run.py sample   --date 20250102 --workers 8
python run.py clean    --date 20250102 --workers 8
python run.py base     --date 20250102 --workers 8

# 2. 计算因子（以 bap 为例，省略 --date 处理所有日期）
python run.py factors  --date 20250102 --factor bap --workers 8

# 3. 单因子评估
python run.py cs_ic       --date 20250102 --factor bap
python run.py cs_quantile --date 20250102 --factor bap

# 4. 多因子合成分层
python run.py multi_factor_quantile --factor-pool intersection --score-method minmax --n-groups 10

# 5. XGBoost 分层模型
python run.py xgb_train   --factor-pool intersection --n-groups 10 --ret-horizon ret100 --num-rounds 1000 --data-workers 32
python run.py xgb_predict --factor-pool intersection --n-groups 10 --ret-horizon ret100 --workers 32

# 6. 汇总对比
python pnl_summary.py
```

---

## 各阶段说明

### 1. sample — 重采样

将原始不规则 Tick 数据（UpdateTime 驱动）重采样到 3 秒等间隔时间网格：

- 上午：09:30:00 – 11:29:57
- 下午：13:00:00 – 14:56:57
- 共约 **4740 个时刻/天**
- 方式：向前对齐（取最近一笔 Tick 的值），记录 `GapSec`（距上一笔实际 Tick 的秒数）

### 2. clean — 清洗

- 剔除全天停牌股票（InstruStatus/TradingPhaseCode 全天为 SUSP/HALT）
- 读取 `config/drop_overrides.csv` 人工覆盖剔除
- 标记 `MaxGapSec > 60s` 的股票日写入 `_gap_review.csv` 供人工复核

### 3. base — 价格与掩码

**价格定义**（加权中间价 WMID）：

```
状态           价格类型                计算公式
正常双边报价   NORMAL_WMID    (BidP1×AskV1 + AskP1×BidV1) / (BidV1 + AskV1)
仅有卖一       LIMIT_DOWN     AskP1（跌停，只有卖盘）
仅有买一       LIMIT_UP       BidP1（涨停，只有买盘）
其他           INVALID        NaN
```

**掩码**：
- `CanUsePrice`：价格有效，可用于计算因子
- `CanUseDoubleSideBook`：双边报价有效（买卖价均合理）
- `CanUseFiveLevelBook`：五档报价有效且单调

**前向收益率**（内联计算）：
```
ret_fwd_100(t) = Price(t+100) / Price(t) - 1   ≈ 5 分钟后收益
ret_fwd_200(t) = Price(t+200) / Price(t) - 1   ≈ 10 分钟后收益
ret_fwd_300(t) = Price(t+300) / Price(t) - 1   ≈ 15 分钟后收益
```
当前和未来时刻均需 `CanUsePrice=True` 才有效。

### 4. factors — 因子计算

以股票为单位，逐日计算，多进程并行。输出格式：

```
Date, SampleTime, SecurityID, Market, {因子列...}
```

详见[因子列表](#因子列表)。

### 5. cs_ic — 截面 IC

对每个时刻，计算因子值与未来收益的截面相关性：

- **IC**：Pearson 相关系数
- **RankIC**：Spearman 秩相关系数

输出每天各时刻的 IC 序列，`ic_report` 命令生成 ICIR 统计和图表。

### 6. cs_quantile — 单因子截面分层

每个时刻按因子值排名，等量分为 5 组（g1 最小，g5 最大），统计各组平均前向收益。理想情况下 g5-g1 收益差越大越好。

### 7. multi_factor_quantile — 多因子合成分层

1. 对每个因子计算截面标准化得分（rank / zscore / minmax 三种方式）
2. 以因子 IC 为权重加权求和，得到合成得分
3. 按合成得分排名，分为 10 或 20 组

---

## 因子列表

| 因子 | 说明 | 输出列 |
|------|------|--------|
| `bap` | 买卖压力（Bid-Ask Pressure）：买盘量与卖盘量的不平衡度 | `bap_15m` ~ `bap_75m`（5档） |
| `mom` | 价格动量：过去 W 分钟对数收益率 | `mom_5m` ~ `mom_90m`（7档） |
| `acc_mom` | 加速动量：短窗口动量减去长窗口动量 | 多档 |
| `neg_skew` | 收益率负偏度：捕捉价格下行尾风险 | `neg_skew_15m` ~ `neg_skew_60m` |
| `amp_slice` | 振幅切片：分段价格振幅 | 多档 |
| `oir` | 委托不平衡率（Order Imbalance Ratio） | `oir_15m` ~ `oir_75m` |
| `ofd` | 订单流动态（Order Flow Dynamics） | `ofd_15m` ~ `ofd_75m` |
| `pv_corr` | 量价相关性 | `pv_corr_100t` ~ `pv_corr_300t` |
| `rigidity` | 价格刚性：价格对订单流的敏感度 | `rigidity_10m` ~ `rigidity_105m` |
| `rsrs` | 阻力支撑相对强度 | `rsrs_30m` ~ `rsrs_105m` |

并集因子池：51 个因子列（`config/factor_pool_union.txt`）
交集因子池：25 个核心因子列（`config/factor_pool_intersection.txt`）

---

## 评估方法

### PnL 指标：多空平均 PnL（bps/tick）

以最高组和最低组的收益差衡量分层效果：

```
多空 PnL = mean(g_top) - mean(g_bottom)
```

单位为 bps/tick（每个 3 秒 tick 的基点收益），越大越好。

### 当前结果汇总（ret100，5 分钟前向收益）

| 方法 | 因子池 | 分组 | PnL (bps/tick) |
|------|--------|------|----------------|
| 多因子 minmax | intersection | g10 | 2.31 |
| 多因子 minmax | intersection | g20 | 2.96 |
| **XGBoost** | union | g10 | **3.25** |

---

## XGBoost 截面分层模型

### 核心思路

将"预测股票未来收益排名"转化为**代价敏感多分类问题**，用 XGBoost 学习因子与收益分位数之间的非线性关系，替代多因子线性加权打分。

与多因子模型的核心区别：
- 多因子：线性加权，无法捕捉因子间交互效应
- XGBoost：决策树，自动学习"bap 高且 mom 低时特别有效"等非线性规律

### 数据准备（dataset.py）

**样本定义**：每一个样本 = 某只股票在某一时刻的因子值向量

```
输入特征：[bap_15m, bap_30m, ..., mom_5m, ..., pv_corr_100t, ...]  ← 25 或 51 维
标注标签：该股票在本时刻截面内按 ret_fwd_100 排名所在的组别（0~9）
```

**截面标注**：每个时刻独立做 qcut，将当前截面所有股票按未来收益等量分为 10 组：
```
收益率 最低 10% → label=0
         ...
收益率 最高 10% → label=9
```

**训练集 / 验证集划分**（严格按日期划分，防止数据泄漏）：
- 11、12 月全部 → 验证集（时间靠后，模拟真实预测）
- 1–10 月每月随机抽 2 天 → 验证集
- 其余 → 训练集

**降采样（stride=100）**：相邻 3 秒的截面因子值高度相关，每 100 个时刻取一个截面（约 5 分钟间隔），每天约 47 个截面。每天 47 截面 × ~500 只股票 = 约 23500 个独立样本，180 天共约 410 万个训练样本。

### 代价敏感损失函数（loss.py）

**为什么不用普通分类损失？**

普通多分类损失对所有错误惩罚相同。但在量化场景中，把高收益股票误判为低收益（该买的没买）远比在低收益组内部混淆代价大。

**惩罚矩阵设计**（以 10 分类为例）：

```
P[真实标签 i, 预测标签 j] =
    0                      i == j（预测正确，无惩罚）
    0                      i,j 均 ≤ 1（最差两组互混，允许）
    |i-j| × 0.5            同侧错分（线性递增轻惩罚）
    5.0 + |i-j|            跨越中线（重惩罚）
```

惩罚矩阵示例（n=10，mid=5）：
```
真实=8，预测=9：同侧，距离1，惩罚=0.5（小）
真实=8，预测=2：跨界，距离6，惩罚=11.0（大）
真实=0，预测=1：最差两组，惩罚=0（允许混淆）
```

**自定义梯度推导**：

```
模型输出 10 个 logit → softmax → 概率分布 [p0, ..., p9]
期望代价 = Σ P[真实, c] × p_c
梯度     = p_k × (P[真实, k] - 期望代价)
二阶梯度 = p_k × (1 - p_k)
```

每轮训练时，XGBoost 调用此函数计算梯度，按梯度方向建树修正错误。

### 训练过程（train.py）

```
构建惩罚矩阵
    ↓
并行加载训练集（180天，n_workers=32）→ 约 410 万样本
并行加载验证集（63天，n_workers=32） → 约 145 万样本
    ↓
XGBoost 训练（最多 1000 轮，早停 30 轮）
  每轮：
    ① 模型对所有样本预测 logits
    ② 调用 cost_obj 计算代价敏感梯度
    ③ 按梯度建 10 棵树（每个分组一棵）
    ④ 在验证集计算 cost，连续 30 轮无改善则停止
    ↓
保存模型（model.ubj）及元数据
```

**关键超参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_depth` | 6 | 树的最大深度，越深越复杂 |
| `eta` | 0.05 | 学习率，越小收敛越慢但越精确 |
| `subsample` | 0.8 | 每轮随机采样 80% 的样本 |
| `colsample_bytree` | 0.8 | 每棵树随机采样 80% 的特征 |
| `min_child_weight` | 20 | 叶节点最少样本数，防止过拟合 |
| `max_bin` | 128 | 直方图分桶数，越少越快 |
| `nthread` | 128 | 树构建的 CPU 线程数 |

### 推理过程（predict.py）

```
加载模型（每个 worker 进程只加载一次）
    ↓
对每个测试日期的全部 4740 个时刻，读取所有股票的因子值
    ↓
模型输出：每只股票 10 个 logit 值
    ↓ softmax
概率分布 [p0, ..., p9]，表示该股票属于各组的置信度
    ↓
得分 = Σ p_c × c（期望分组值，0~9 之间的连续数）
    ↓
同一时刻所有股票按得分排名，等量分为 10 组
    ↓
计算各组平均前向收益，输出 g1~g10
```

**得分直觉示例**：
```
概率 = [0.02, 0.03, 0.09, 0.18, 0.33, 0.20, 0.09, 0.04, 0.02, 0.01]
         g1    g2    g3    g4    g5    g6    g7    g8    g9   g10

得分 = 1×0.02 + 2×0.03 + ... + 10×0.01 = 4.8
→ 模型认为此股票大概率在第 5 组附近
```

### 命令行参数

**训练**：
```bash
python run.py xgb_train \
    --factor-pool intersection \   # 因子池：all / union / intersection
    --n-groups 10 \                # 分组数：10 或 20
    --ret-horizon ret100 \         # 收益窗口：ret100 / ret200 / ret300
    --num-rounds 1000 \            # 最大 boosting 轮次
    --early-stop 30 \              # 早停耐心轮次
    --stride 100 \                 # 截面降采样间隔（默认 100）
    --data-workers 32 \            # 数据加载并行进程数
    --within-scale 0.5 \           # 同侧错分惩罚系数
    --cross-base 5.0 \             # 跨界基础惩罚
    --neighbor-zero 2 \            # 最差前 k 组互混零惩罚
    --force                        # 强制重训（覆盖已有模型）
```

**推理**：
```bash
python run.py xgb_predict \
    --factor-pool intersection \
    --n-groups 10 \
    --ret-horizon ret100 \
    --workers 32               # 推理并行进程数（每 worker 只加载一次模型）
```

### 输出文件

```
result/eval/xgb_quantile/{factor_pool}/g{n}/{ret_h}/
    model.ubj            训练好的 XGBoost 模型
    feature_names.txt    输入特征列名
    penalty_matrix.npy   惩罚矩阵
    split_info.json      训练元数据（日期划分、最优轮次等）
    {date}.parquet       每日推理结果（Date, SampleTime, g1~g10）
    _daily.csv           每日各组均值
    _summary.csv         跨日整体均值
    _cum_daily.csv       跨日累计收益
    _chart_tick.png      全天 tick 级别曲线图
    _chart_slot_*.png    8 个 30 分钟时段图
```

---

## 全局配置

所有路径和参数集中在 `config/__init__.py`，修改配置只需改这一处。

### 路径配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAW_ROOT` | `{ROOT}/data` | 原始快照数据根目录 |
| `SAMPLED_ROOT` | `{ROOT}/result/sampled` | 重采样输出 |
| `CLEANED_ROOT` | `{ROOT}/result/cleaned` | 清洗输出 |
| `BASE_ROOT` | `{ROOT}/result/base` | base 数据输出 |
| `FACTOR_ROOT` | `{ROOT}/result/factor` | 因子输出 |
| `EVAL_ROOT` | `{ROOT}/result/eval` | 评估结果输出 |
| `FACTOR_POOL_UNION_TXT` | `config/factor_pool_union.txt` | 并集因子池白名单 |
| `FACTOR_POOL_INTERSECTION_TXT` | `config/factor_pool_intersection.txt` | 交集因子池白名单 |

### 采样参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SAMPLE_FREQ` | `"3s"` | 采样间隔 |
| `AM_START / AM_END` | `09:30 / 11:30` | 上午交易时段 |
| `PM_START / PM_END` | `13:00 / 14:57` | 下午交易时段 |
| `RETURN_HORIZONS` | `[100, 200, 300]` | 前向收益率窗口（tick） |
| `GAP_REVIEW_THRESHOLD` | `60.0 s` | 数据间隔复核阈值 |

### config/drop_overrides.csv

人工指定需额外删除的股票日，格式：

```csv
Date,SecurityID,Reason
20250512,ALL_SZ,深交所系统性数据缺失
20250217,002916,全天数据异常
```

`SecurityID` 支持 6 位股票代码、`ALL`（全部）、`ALL_SH`（沪市全部）、`ALL_SZ`（深市全部）。
