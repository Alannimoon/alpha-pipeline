# 因子评测流水线

## 项目结构

```
factor/
├── data/                    ← 原始快照数据（只读）
│   └── 20250102/
│       ├── 000001.csv
│       └── ...
├── result/                  ← 各阶段处理结果（自动生成，不入库）
│   ├── sampled/
│   ├── cleaned/
│   ├── base/
│   ├── factor/
│   └── eval/
├── pipeline/
│   ├── ingest/              ← 数据处理（README 见同目录）
│   │   ├── sample.py
│   │   ├── clean.py
│   │   └── base.py
│   ├── factor/              ← 因子计算（README 见同目录）
│   │   ├── _core.py
│   │   ├── bap.py
│   │   ├── mom.py
│   │   └── compute.py
│   └── eval/                ← 因子评估（README 见同目录）
│       ├── _panel.py
│       ├── cs_ic.py
│       ├── ts_ic.py
│       ├── ic_stats.py
│       └── ic_plot.py
├── config/
│   ├── __init__.py          ← 全局配置
│   └── drop_overrides.csv   ← 人工删除覆盖表
└── run.py                   ← 统一命令行入口
```

---

## 快速上手

```bash
python run.py sample   --date 20250102          # 1. 重采样
python run.py clean    --date 20250102          # 2. 清洗
python run.py base     --date 20250102          # 3. 生成 base（价格、掩码）
python run.py factors  --date 20250102 --factor bap  # 4. 计算因子（含内联收益率）
python run.py cs_ic    --date 20250102 --factor bap  # 5. 截面 IC
python run.py ts_ic    --date 20250102 --factor bap  # 6. 时序 IC
python run.py ic_stats --factor bap             # 7. IC 汇总统计
python run.py ic_plot  --factor bap             # 8. IC 画图
```

省略 `--date` 时自动处理所有已有日期。

---

## run.py — 命令行入口

所有阶段通过 `run.py` 统一调用，实际逻辑分散在各 pipeline 子模块中。

### 公共参数

| 参数 | 适用阶段 | 说明 |
|---|---|---|
| `--date` | sample / clean / base / factors / cs_ic / ts_ic | 只处理指定日期，如 `20250102`；省略则处理所有日期 |
| `--workers` | sample / clean / base / factors / cs_ic / ts_ic | 并行进程数；省略则使用 CPU 核数 |
| `--factor` | factors / cs_ic / ts_ic / ic_stats / ic_plot | 因子名称，如 `bap` / `mom`；默认 `bap` |

### 各阶段说明

| 阶段 | 输入 | 输出 |
|---|---|---|
| `sample` | `data/{date}/` | `result/sampled/{date}/` |
| `clean` | `result/sampled/{date}/` | `result/cleaned/{date}/` |
| `base` | `result/cleaned/{date}/` | `result/base/{date}/` |
| `factors` | `result/base/{date}/` | `result/factor/{factor}/{date}/` |
| `cs_ic` | `result/factor/{factor}/{date}/` | `result/eval/cs_ic/{factor}/` |
| `ts_ic` | `result/factor/{factor}/{date}/` | `result/eval/ts_ic/{factor}/` |
| `ic_stats` | `result/eval/cs_ic/` + `result/eval/ts_ic/` | `result/eval/ic_stats/{factor}/`（CS 含 ic_std/ICIR，TS 含 per-stock ICIR 均值） |
| `ic_plot` | `result/eval/ic_stats/{factor}/` | `result/eval/ic_stats/{factor}/*.png`（每张图含 IC 均值 + ICIR 上下两子图） |

> 前向收益率（ret_fwd_100/200/300）在 `factors` 阶段内联计算，无独立步骤。

---

## config/__init__.py — 全局配置

所有路径和参数集中在此文件，修改配置只需改这一处。

### 路径配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAW_ROOT` | `{ROOT}/data` | 原始快照数据根目录 |
| `SAMPLED_ROOT` | `{ROOT}/result/sampled` | 重采样输出 |
| `CLEANED_ROOT` | `{ROOT}/result/cleaned` | 清洗输出 |
| `BASE_ROOT` | `{ROOT}/result/base` | base 数据输出 |
| `FACTOR_ROOT` | `{ROOT}/result/factor` | 因子输出 |
| `EVAL_ROOT` | `{ROOT}/result/eval` | 评估结果输出 |

`ROOT` 自动推导为 `config/` 的上级目录（即本项目根目录）。

### 采样参数

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SAMPLE_FREQ` | `"3s"` | 采样间隔 |
| `AM_START` | `"09:30:00"` | 上午采样起点（含） |
| `AM_END` | `"11:30:00"` | 上午采样终点（不含） |
| `PM_START` | `"13:00:00"` | 下午采样起点（含） |
| `PM_END` | `"14:57:00"` | 下午采样终点（不含） |

### 清洗参数

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DROP_OVERRIDES_CSV` | `config/drop_overrides.csv` | 人工删除覆盖表路径 |
| `GAP_REVIEW_THRESHOLD` | `60.0` | MaxGapSec 超过此值（秒）的保留股票日写入 `_gap_review.csv` 供复核 |

### 收益率 / 因子参数

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RETURN_HORIZONS` | `[100, 200, 300]` | 前向收益率窗口（tick），对应 5 / 10 / 15 分钟 |

---

## config/drop_overrides.csv — 人工删除覆盖表

供 `clean.py` 读取，人工指定需要额外删除的股票日（在自动停牌识别之外）。

### 格式

```
Date,SecurityID,Reason
20250512,ALL_SZ,深交所系统性数据缺失，经gap诊断确认
20250217,002916,人工复核确认全天数据异常
```

| 字段 | 说明 |
|---|---|
| `Date` | 8 位日期，如 `20250512` |
| `SecurityID` | 6 位股票代码；`ALL` 表示该日所有股票；`ALL_SH` / `ALL_SZ` 表示该日对应交易所全部股票 |
| `Reason` | 说明原因，便于后续追溯 |

注释行（`#` 开头）和空行自动跳过。文件为空或只有注释行时不影响运行。
