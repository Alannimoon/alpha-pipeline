# pipeline/factor — 因子计算模块

```
pipeline/factor/
├── _core.py      ← 共享工具（数据加载、窗口工具函数）
├── bap.py        ← BAP 因子（买卖盘压力）
└── compute.py    ← 编排器（批量调用所有因子、写文件）
```

输入来自 `result/base/`，输出写入 `result/factor/`。收益率在 `_core.py` 中内联计算，无需单独的 returns 目录。

---

## 1. _core.py — 共享工具

### 数据加载

```python
load_data(base_path, horizons=[100, 200, 300]) -> pd.DataFrame
```

读取单只股票的 base 文件，内联计算前向收益率后返回完整 DataFrame。

- 来自 base：`Price, CumVolume, CanUsePrice, CanUseDoubleSideBook, CanUseFiveLevelBook, BidPrice1-5, AskPrice1-5, BidVolume1-5, AskVolume1-5`
- 内联计算：`ret_fwd_100, ret_fwd_200, ret_fwd_300`

收益率定义：`ret_fwd_{h} = P(t+h)/P(t) - 1`，仅当 `CanUsePrice(t)` 和 `CanUsePrice(t+h)` 均为 True 时有效，否则为 NaN。

所有数值列强制转为 `float64`，三个掩码列转为 `bool`。

### 工具函数

#### `is_limit_tick(df)` → `np.ndarray[bool]`

判断每个 tick 是否为涨跌停：
```
CanUsePrice == True  AND  CanUseDoubleSideBook == False
```
即：有有效价格（单边盘口），但双边盘口不可用。

#### `window_valid_mask(can_use, window, max_invalid_ratio=0.10)` → `np.ndarray[bool]`

滚动窗口有效性检查。对每个 tick，检查过去 `window` 个 tick（含当前）中 `can_use=False` 的占比：
- 占比 > `max_invalid_ratio`（默认 10%）→ `False`（因子输出 NaN）
- 占比 ≤ 10% → `True`
- 历史不足 `window` 个 tick → `False`（`min_periods=window`，不足时为 NaN，NaN > 0.10 → False）

> **注意**：`can_use` 参数由调用方传入，不同因子传入不同数组。BAP 传入 `np.isfinite(bap_raw)`，价格类因子传入 `CanUsePrice`。原则：**用什么数算，就看什么数的有效占比**。

#### `rolling_mean_masked(values, mask, window)` → `np.ndarray[float]`

滚动均值，仅对 `mask=True` 的 tick 参与计算（`min_periods=1`）：
```python
pd.Series(values).where(mask).rolling(window, min_periods=1).mean()
```
窗口有效性的判断不在此函数内，由调用方通过 `window_valid_mask` 的结果覆盖。

#### `rolling_any(bool_arr, window)` → `np.ndarray[bool]`

滚动 OR：窗口内任意一个 tick 为 True 则返回 True（`min_periods=window`，历史不足返回 False）。

### 常量

| 常量 | 值 | 含义 |
|---|---|---|
| `TICKS_PER_MIN` | 20 | 每分钟 tick 数（3 秒/tick × 20 = 60 秒） |

---

## 2. bap.py — BAP 因子（买卖盘压力）

### 因子含义

**Bid-Ask Pressure**：衡量买卖盘的相对力量。正值表示买盘压力更大，负值表示卖盘压力更大。

### 计算步骤

**第一步：逐 tick 原始值**（仅 `CanUseDoubleSideBook=True` 时有效）

```
BAP_raw(t) = (BidVolume1(t) - AskVolume1(t)) / (BidVolume1(t) + AskVolume1(t))
```

分母为 0 或 `CanUseDoubleSideBook=False` 时，`BAP_raw = NaN`。

**第二步：窗口有效性检查**

过去 W 个 tick 中 `BAP_raw` 非 NaN 的比例 < 90% → 因子记 NaN。

直接检查 `np.isfinite(bap_raw)` 的滚动占比，无需单独看 `CanUseDoubleSideBook`，因为二者等价（`BAP_raw` 非 NaN 当且仅当 `CanUseDoubleSideBook=True` 且分母 > 0）。

**第三步：滚动均值**

```
bap_{W}m(t) = mean(BAP_raw(i), i ∈ [t-W+1, t], BAP_raw(i) 非 NaN)
```

`min_periods=1`：窗口内只要有一个有效 tick 就计算均值；有效性的主要门控来自第二步的 90% 阈值。

**第四步：`has_limit` 标记**

```
bap_{W}m_has_limit(t) = 过去 W 个 tick 内是否存在涨跌停 tick
```

> 对 BAP 而言，涨跌停 tick 的 `BAP_raw` 本身就是 NaN，不会参与均值计算，因此 `has_limit` 在实际计算上不影响因子值，**恒为 False**。保留此列仅为与其他因子的输出格式对齐。

### 计算窗口

| 窗口（分钟） | 对应 tick 数 |
|---|---|
| 15 | 300 |
| 30 | 600 |
| 45 | 900 |
| 60 | 1200 |
| 75 | 1500 |

### 输出列

每个窗口输出两列：

| 列名 | 类型 | 说明 |
|---|---|---|
| `bap_{W}m` | float | BAP 因子值，无效时为 NaN |
| `bap_{W}m_has_limit` | bool | 窗口内是否出现过涨跌停 tick（BAP 恒 False） |

---

## 3. compute.py — 因子编排器

### 作用

批量读取 base 数据（收益率内联计算），调用所有注册的因子模块，将所有因子列拼接后写出 CSV。

### 并行策略
所有日期、所有股票的 `(day, stock)` 任务打平为单一任务池并行执行（日间和日内同时并行）。`_worker` 内部负责创建输出目录，任务完成后按日期分组回写 per-day `_summary.csv`。

### 输出文件格式

```
Date, SampleTime, SecurityID, Market,
ret_fwd_100, ret_fwd_200, ret_fwd_300,
<factor columns...>
```

收益率列由 `load_data` 内联计算后透传到输出，方便 IC 计算模块直接读取因子文件。

### 输出目录结构

```
data/factor/
├── 20250102/
│   ├── 000001.csv
│   ├── ...
│   └── _summary.csv    ← 各因子列的有效 tick 数统计（nnz_{factor_col}）
└── _summary.csv
```

`_summary.csv` 中的 `nnz_{col}` 统计仅针对因子值列（不含 `_has_limit` 列）。

### 如何添加新因子

1. 在 `pipeline/factor/` 下新建 `<name>.py`，实现：
   ```python
   def compute(df: pd.DataFrame) -> pd.DataFrame:
       # 输入：单只股票单日完整 df（load_data 输出）
       # 输出：只含因子列的 df，index 与输入对齐
   ```

2. 在 `compute.py` 中注册：
   ```python
   from . import <name>
   _FACTORS = [..., <name>]
   ```

编排器自动合并所有因子列，无需其他改动。
