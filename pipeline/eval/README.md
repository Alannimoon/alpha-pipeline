# pipeline/eval — 因子评估模块

```
pipeline/eval/
├── _panel.py   ← 共享数学工具（因子列提取、宽表 IC 计算）
├── cs_ic.py    ← 截面 IC
├── ts_ic.py    ← 时序 IC
├── ic_stats.py ← IC 汇总统计（均值、标准差、ICIR）
└── ic_plot.py  ← IC 画图
```

整体流程：

```
result/factor/{factor}/
    ↓ cs_ic.py / ts_ic.py
result/eval/cs_ic/{factor}/   result/eval/ts_ic/{factor}/
    ↓ ic_stats.py
result/eval/ic_stats/{factor}/cs_ic_stats.csv  ts_ic_stats.csv
    ↓ ic_plot.py
result/eval/ic_stats/{factor}/cs_ret100.png  ...  ts_ret300.png
```

---

## 1. _panel.py — 共享数学工具

纯计算工具，不做任何文件 I/O。供 `cs_ic.py` 调用；`ts_ic.py` 只使用 `get_factor_cols`。

### `get_factor_cols(df, factor_name)` → `list[str]`

从 DataFrame 列名中提取属于 `factor_name` 的**因子值列**（排除 `_has_limit` 列）。

例：`factor_name="bap"` → `["bap_15m", "bap_30m", "bap_45m", "bap_60m", "bap_75m"]`

### `compute_ic_pair(f_wide, r_wide, axis=1)` → `(ic, rankic)`

对两张同维度宽表按行（axis=1）计算 IC 和 RankIC，供 CS-IC 使用。

| 参数 | 说明 |
|---|---|
| `f_wide` | 因子宽表，行 = 时间点（4740行），列 = SecurityID（498列） |
| `r_wide` | 收益率宽表，同维度 |

- **IC**（Pearson）：每行取有效位联合排除 NaN，计算相关系数；分母 < 1e-12 时返回 NaN
- **RankIC**（Spearman）：用 `_nanrank_2d`（`argsort(argsort(...))`，纯 numpy）对每行排名后再算 Pearson，比 `pd.DataFrame.rank()` 快约 2-3 倍

---

## 2. cs_ic.py — 截面 IC

### 计算逻辑

对每个时间点 `(Date, SampleTime)`，在当日所有股票的**截面**上计算因子值与未来收益率的相关系数。

```
CS-IC(t) = Pearson( factor(t, 所有股票), ret_fwd_h(t, 所有股票) )
```

**关键：必须在同一时刻聚合所有股票，因此需要宽表。**

#### 构建宽表（直接组装，不经过长表）

逐文件读取时只加载必要的列（`usecols`），以 SampleTime 为 index，将每只股票的各列存入字典，最后用 `pd.DataFrame(dict_of_series)` 一步组装为宽表：

```
读 000001.csv → {"000001": Series(SampleTime→bap_15m)}
读 000002.csv → {"000002": Series(SampleTime→bap_15m)}
...
pd.DataFrame({"000001": ..., "000002": ..., ...})
→ 因子宽表（4740行 × 498列），无需 concat 长表再 unstack
```

每列（5个因子 + 3个收益率）各建一张宽表，共 8 张，每张 4740 × 498。

#### 计算与输出

```
对每个 (factor_col, ret_horizon)：
  compute_ic_pair(f_wide, r_wide, axis=1)
  → IC Series（4740个值）+ RankIC Series（4740个值）

组装结果 DataFrame（行 = 时间点），按 SampleTime 切行得到 am / pm 子集
（IC 只算一遍，session 仅为行过滤）
```

### Session 定义

| session | SampleTime 范围 | 行数 |
|---|---|---|
| `all` | 全天 | 4740 |
| `am` | `<= "11:29:57"` | 2400 |
| `pm` | `>= "13:00:00"` | 2340 |

### 并行策略

按天并行：每个 worker 处理一整天（读 498 个文件 + 算全天 IC）。

### 输出目录结构

```
result/eval/cs_ic/{factor_name}/
├── ret100_all/20250102.csv
├── ret100_am/20250102.csv
├── ret100_pm/20250102.csv
├── ret200_all/20250102.csv
├── ...
└── ret300_pm/20250102.csv    ← 共 3 horizon × 3 session = 9 个子目录
```

每个 CSV 列：`Date, SampleTime, ic_{fc}, rankic_{fc}, ...`（BAP 共 10 个 IC 列 + 2 个索引列，4740 行）

---

## 3. ts_ic.py — 时序 IC

### 计算逻辑

对每只股票的单日时间序列，计算因子值与未来收益率的相关系数。

```
TS-IC(stock) = Pearson( factor(stock, 所有时刻), ret_fwd_h(stock, 所有时刻) )
```

**关键：每只股票独立计算，不需要聚合所有股票，直接对单文件的列做相关。**

#### 单文件直接计算

```
读 000001.csv（4740行）
  → 掩掉 has_limit=True 的因子值（置 NaN）
  → 按 SampleTime 切 session 子集（all/am/pm）
  → 对每个 (session, ret_horizon, factor_col)：
      f = sub[fc].values      ← 一维数组（约 4740/2400/2340 个值）
      r = sub[h_col].values   ← 一维数组
      IC     = _pearson_1d(f, r)    → 一个标量
      RankIC = _spearman_1d(f, r)   → 一个标量
重复 498 只股票，每只股票贡献 3×3×5×2 = 90 个标量
```

无需 concat、无需 unstack，每次运算的数组长度最多 4740。

#### `_pearson_1d` / `_spearman_1d`

- 联合排除 NaN 后计算，有效样本 < 2 时返回 NaN
- Spearman：`argsort(argsort(valid_values))` 对联合有效的子集排名，再算 Pearson

### `has_limit` 处理

计算前将 `{factor_col}_has_limit=True` 的 tick 的因子值置 NaN，使涨跌停时段不参与时序相关系数计算。

> 对 BAP 而言，`has_limit` 恒为 False，此步骤不影响实际结果，但保持了与其他因子逻辑的一致性。

### 并行策略

按天并行：每个 worker 读取当日 498 个文件并逐一计算，汇总后写出 9 个 CSV。

### 输出目录结构

```
result/eval/ts_ic/{factor_name}/
├── ret100_all/20250102.csv
├── ret100_am/20250102.csv
├── ...
└── ret300_pm/20250102.csv    ← 同样 9 个子目录
```

每个 CSV 列：`Date, SecurityID, ts_ic_{fc}, ts_rankic_{fc}, ...`（BAP 共 10 列 + 2 个索引列，498 行）

---

## 4. ic_stats.py — IC 汇总统计

### 作用

读取 cs_ic / ts_ic 的逐日结果，输出每个 `(ret_horizon, session, factor_window)` 组合的 IC 均值和 ICIR。CS 和 TS 采用不同的聚合口径。

### CS-IC 统计逻辑

```
① 每个交易日：对该日所有时间点的 IC 取均值 → 日度 IC 均值
② 跨所有交易日：对日度均值取均值 → ic_mean
③ 跨所有交易日：对日度均值取标准差（ddof=1）→ ic_std
④ ICIR = ic_mean / ic_std
```

日内先平均再跨日，消除日内高频自相关对标准差估计的干扰。

### TS-IC 统计逻辑（按股票聚合）

```
① 每只股票：收集其逐日 TS-IC 值序列
② 每只股票：ICIR_stock = mean(逐日 IC) / std(逐日 IC, ddof=0)
③ 最终 ICIR   = mean(各股票 ICIR_stock)
④ 最终 ic_mean = mean(各股票逐日 IC 均值)
```

含义：衡量"典型股票"的因子预测力在时间维度上的稳定性。与 CS-IC ICIR（整体信号的跨日稳定性）度量角度不同，不宜直接横向比较大小。`ddof=0` 与历史脚本保持一致。

### 输出文件

```
result/eval/ic_stats/{factor_name}/
├── cs_ic_stats.csv
└── ts_ic_stats.csv
```

**CS 输出列：**

| 列 | 说明 |
|---|---|
| `ret_horizon` | `ret100 / ret200 / ret300` |
| `session` | `all / am / pm` |
| `factor_window` | 因子计算窗口（分钟） |
| `factor_col` | 完整列名，如 `bap_15m` |
| `ic_mean` | IC 均值 |
| `rankic_mean` | RankIC 均值 |
| `ic_std` | IC 标准差（ddof=1） |
| `rankic_std` | RankIC 标准差 |
| `icir` | `ic_mean / ic_std` |
| `rankic_ir` | `rankic_mean / rankic_std` |
| `n_days` | 参与统计的交易日数 |

**TS 输出列：**

| 列 | 说明 |
|---|---|
| `ret_horizon` | 同上 |
| `session` | 同上 |
| `factor_window` | 同上 |
| `factor_col` | 同上 |
| `ic_mean` | 各股票逐日 IC 均值的均值 |
| `rankic_mean` | 各股票逐日 RankIC 均值的均值 |
| `icir` | 各股票 ICIR 的均值（ddof=0） |
| `rankic_ir` | 各股票 RankIC IR 的均值（ddof=0） |
| `n_days` | 交易日文件数 |
| `n_stocks` | 参与统计的股票数 |

---

## 5. ic_plot.py — IC 画图

### 作用

读取 `ic_stats` 输出的汇总统计，为每个因子生成 6 张图（CS × 3 收益率窗口 + TS × 3 收益率窗口）。

### 图形规格

每张图包含**上下两个子图**：

- **上图**：IC 均值（`ic_mean` / `rankic_mean`）
- **下图**：ICIR（`icir` / `rankic_ir`）
- **X 轴**：因子计算窗口（分钟），两图共享
- **每个子图 6 条线**：

| 线 | 颜色 | 样式 |
|---|---|---|
| IC all | 蓝色 `#1f77b4` | 实线 |
| RankIC all | 蓝色 `#1f77b4` | 虚线 |
| IC am | 橙色 `#ff7f0e` | 实线 |
| RankIC am | 橙色 `#ff7f0e` | 虚线 |
| IC pm | 绿色 `#2ca02c` | 实线 |
| RankIC pm | 绿色 `#2ca02c` | 虚线 |

### 输出文件

```
result/eval/ic_stats/{factor_name}/
├── cs_ret100.png
├── cs_ret200.png
├── cs_ret300.png
├── ts_ret100.png
├── ts_ret200.png
└── ts_ret300.png
```
