# pipeline/ingest — 数据摄入模块

```
pipeline/ingest/
├── sample.py  ← 重采样：原始快照 → 固定 3s 时间网格
├── clean.py   ← 清洗：删除停牌日，标记大间隔待复核
└── base.py    ← Base 生成：价格定义、涨跌停标注、盘口掩码
```

三个模块串行构成完整的数据预处理链，输出供 `pipeline/factor/` 使用：

```
data/{date}/{secid}.csv          ← 原始快照（UpdateTime 驱动）
    ↓ sample.py
result/sampled/{date}/{secid}.csv
    ↓ clean.py
result/cleaned/{date}/{secid}.csv
    ↓ base.py
result/base/{date}/{secid}.csv   → pipeline/factor/ 读取
```

> **命名说明**：模块命名为 `ingest` 而非 `data`，避免与根目录的原始数据目录 `data/` 产生 `.gitignore` 层面的冲突（`.gitignore` 中 `data/` 会匹配任意层级的同名目录）。

---

## 1. sample.py — 重采样

### 作用

将不规则的原始 tick 流（由 `UpdateTime` 驱动，频率不固定）映射到固定 3s 时间网格。

### 采样网格

| 时段 | 范围 | 间隔 | Tick 数 |
|---|---|---|---|
| 上午 | 09:30:00 – 11:29:57 | 3s | 2400 |
| 下午 | 13:00:00 – 14:56:57 | 3s | 2340 |

使用左闭右开区间（`inclusive="left"`），即包含起始时刻、不含终止时刻（11:30:00 和 14:57:00 不在网格内）。

### 核心算法

`pd.merge_asof` backward asof merge：对每个采样时刻 `t`，取原始序列中 `UpdateTime <= t` 的最近一条记录。

预处理：
1. 解析 `UpdateTime` 为带日期的 `datetime`（支持 `%H:%M:%S` 和 `%H:%M:%S.%f`）
2. 同一毫秒多条记录保留最后一条（`drop_duplicates("_ts", keep="last")`）
3. 按 `UpdateTime` 排序后执行 asof merge

### 输入格式

```
data/{date}/{secid}.csv
```

| 列 | 说明 |
|---|---|
| `Date` | 日期，如 `20250102` |
| `UpdateTime` | 原始 tick 时间，如 `09:30:01.234` |
| `SecurityID` | 证券代码 |
| `PreCloPrice` | 昨收价 |
| `LastPrice` | 最新成交价 |
| `Turnover` | 累计成交金额 |
| `TradVolume` / `Volume` | 累计成交量（沪深字段名不同） |
| `InstruStatus` / `TradingPhaseCode` | 交易状态码（沪深字段名不同） |
| `AskPrice1-5`, `AskVolume1-5` | 卖方五档 |
| `BidPrice1-5`, `BidVolume1-5` | 买方五档 |

### 输出格式

在原始列基础上新增：

| 列 | 说明 |
|---|---|
| `SampleTime` | 采样时刻，格式 `%H:%M:%S`，如 `09:30:00` |
| `UpdateTime` | 保留原始 tick 时间（诊断用） |
| `GapSec` | 原始 tick 与采样时刻的时间差（秒），值越大说明数据越"陈旧" |

`Date` 列改写为 `%Y-%m-%d` 格式（如 `2025-01-02`）。

### Summary 字段

每日写出 `_summary.csv`，字段含义：

| 字段 | 说明 |
|---|---|
| `RawRows` | 有效 tick 数（解析时间后） |
| `SampleRows` | 采样点数（正常为 4740） |
| `MaxGapSec` | 最大间隔秒数 |
| `MeanGapSec` | 平均间隔秒数 |
| `PctGapGt60` | GapSec > 60s 的采样点占比 |
| `Status` | `OK` / `LARGE_GAP`（MaxGapSec > 60s）/ `NO_RAW_DATA` / `NO_VALID_TIME` / `FAIL: ...` |

---

## 2. clean.py — 清洗

### 作用

在重采样数据基础上删除无交易价值的股票日；保留的异常案例输出供人工复核。

### 删除规则

#### 规则 1：全天停牌（`ALL_DAY_SUSPEND`）

若该股票日所有 tick 的状态码均属于 SuspendLike，则整天停牌，直接删除。

**SH/SZ 状态码分类：**

| 交易所 | 字段 | TradeLike | AuctionLike | SuspendLike |
|---|---|---|---|---|
| 上交所（SH） | `InstruStatus` | `TRADE` | `OCALL / CCALL / ICALL / FCALL` | `SUSP / HALT` 及其他所有 |
| 深交所（SZ） | `TradingPhaseCode`（两位字符） | 第一位 `T`（且第二位非 `1`） | 第一位 `O / C`（且第二位非 `1`） | 第二位 `1`（全天停牌标志，优先），或第一位 `H / B / S / E / A / V` 及其他 |

市场类型由字段名自动判断：有 `InstruStatus` 列 → SH；有 `TradingPhaseCode` 列 → SZ。

#### 规则 2：手动覆盖（`MANUAL_OVERRIDE`）

从 `config/drop_overrides.csv` 读取人工确认的删除条目，格式：

```csv
Date,SecurityID,Reason
20250102,000001,数据异常
20250102,ALL_SZ,深市全日系统故障
ALL,600000,长期退市处理
```

`SecurityID` 支持：具体代码（6位）、`ALL`（当日全部）、`ALL_SH`（沪市全部）、`ALL_SZ`（深市全部）。`Date` 为 `ALL` 时对所有日期生效。

### 仅标记、不自动删除

保留的股票日中若 `MaxGapSec > gap_threshold`（默认 60s），写入 `_gap_review.csv` 供人工复核。确认后手动加入 `drop_overrides.csv`，下次重跑 clean 时生效。

### 输出文件

```
result/cleaned/
├── {date}/{secid}.csv      ← 通过清洗的股票日（内容与 sampled 完全相同，未做任何列修改）
├── _dropped.csv            ← 被删除的股票日明细（Date, SecurityID, DropReason）
└── _gap_review.csv         ← 保留但 MaxGapSec 超阈值的股票日（Date, SecurityID, MaxGapSec）
```

---

## 3. base.py — Base 数据生成

### 作用

在清洗后数据的基础上，为每个 tick 计算标准化价格和三个可用性掩码，供下游因子计算使用。

### 价格定义

仅 TradeLike 状态的 tick 有价格，优先级如下：

| 条件 | PriceType | Price |
|---|---|---|
| 双边有效（bid 和 ask 均存在且 `BidPrice1 ≤ AskPrice1`） | `NORMAL_WMID` | `(BidPrice1 × AskVolume1 + AskPrice1 × BidVolume1) / (BidVolume1 + AskVolume1)` |
| 仅卖单（无 bid，有 ask）→ 跌停 | `LIMIT_DOWN_ONE_SIDED` | `AskPrice1` |
| 仅买单（无 ask，有 bid）→ 涨停 | `LIMIT_UP_ONE_SIDED` | `BidPrice1` |
| 双边均缺或 spread 倒挂 | `INVALID` | NaN |
| 非 TradeLike 状态 | — | NaN |

**WMID（反向加权中间价）**：用对手方的量作权重，买方量大 → 价格偏向卖方，反映实际成交压力。

### 三个掩码定义

| 掩码 | 条件 |
|---|---|
| `CanUsePrice` | TradeLike AND Price 非 NaN |
| `CanUseDoubleSideBook` | TradeLike AND 双边一档有效（bid & ask 均存在且 `BidPrice1 ≤ AskPrice1`） |
| `CanUseFiveLevelBook` | TradeLike AND 五档完整（全部 20 个价量列非 NaN）AND ask/bid 价格梯度单调 AND `BidPrice1 ≤ AskPrice1` |

**五档梯度校验**：
- Ask：`AskPrice1 ≤ AskPrice2 ≤ AskPrice3 ≤ AskPrice4 ≤ AskPrice5`
- Bid：`BidPrice1 ≥ BidPrice2 ≥ BidPrice3 ≥ BidPrice4 ≥ BidPrice5`

### 输出格式

```
result/base/{date}/{secid}.csv
```

| 列 | 类型 | 说明 |
|---|---|---|
| `Date` | str | 日期 |
| `SampleTime` | str | 采样时刻 |
| `SecurityID` | str | 证券代码 |
| `Market` | str | `SH` 或 `SZ`（由字段名自动判断） |
| `CumVolume` | float | 累计成交量（原 `TradVolume`/`Volume`） |
| `GapSec` | float | 原始 tick 间隔（透传自 sampled） |
| `PriceType` | str | 见价格定义表 |
| `Price` | float | 当前可用价格，无效时为 NaN |
| `CanUsePrice` | bool | 价格可用 |
| `CanUseDoubleSideBook` | bool | 双边一档盘口可用 |
| `CanUseFiveLevelBook` | bool | 五档盘口完整可用 |
| `BidPrice1-5` | float | 买方五档价格 |
| `AskPrice1-5` | float | 卖方五档价格 |
| `BidVolume1-5` | float | 买方五档量 |
| `AskVolume1-5` | float | 卖方五档量 |

> `PriceType` 和 `GapSec` 列由 `pipeline/factor/_core.py` 的 `usecols` 过滤，不参与因子计算，仅保留供诊断使用。

### Summary 字段

| 字段 | 说明 |
|---|---|
| `NormalWmidTicks` | PriceType = NORMAL_WMID 的 tick 数 |
| `LimitDownTicks` | PriceType = LIMIT_DOWN_ONE_SIDED 的 tick 数 |
| `LimitUpTicks` | PriceType = LIMIT_UP_ONE_SIDED 的 tick 数 |
| `CanUsePriceTicks` | CanUsePrice = True 的 tick 数 |
| `CanUseDoubleTicks` | CanUseDoubleSideBook = True 的 tick 数 |
| `CanUseFiveTicks` | CanUseFiveLevelBook = True 的 tick 数 |

---

## 并行策略（三个模块通用）

所有模块均按日期串行、日内股票并行（`ProcessPoolExecutor`）：

```
for day in dates:              ← 串行，保证每日 summary 按序写出
    tasks = [每只股票的处理任务]
    pool.submit(tasks)         ← 日内并行
    写 _summary.csv
写 _summary.csv（全局）
```

- `max_workers=None`：默认使用 CPU 核数
- `max_workers=1`：单进程模式，便于调试（跳过 `ProcessPoolExecutor`）
- Worker 捕获所有异常，单只股票失败不中断整体进度，错误记录在 summary 的 `Status` 字段
