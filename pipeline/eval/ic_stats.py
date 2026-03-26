"""
IC 汇总统计模块。

读取 cs_ic / ts_ic 结果文件，输出每个（因子窗口, 收益率窗口, session）
组合的 IC 均值、ICIR。

CS-IC 统计逻辑
--------------
1. 对每个交易日，将日内所有时间点的 IC 取均值 → 日度 IC 均值
2. 跨所有交易日对日度均值取均值 → ic_mean
3. 跨所有交易日对日度均值取标准差（ddof=1）→ ic_std
4. ICIR = ic_mean / ic_std

TS-IC 统计逻辑（方法 C，按股票聚合）
--------------------------------------
1. 对每只股票，收集其逐日 TS-IC 序列
2. 每只股票的 ICIR = mean(逐日 IC) / std(逐日 IC, ddof=0)
3. 最终 ICIR   = mean(各股票 ICIR)
4. 最终 ic_mean = mean(各股票逐日 IC 均值)

含义：衡量"典型股票"的因子预测力在时间维度上的稳定性。
ddof=0 与 origin/daily_year_ts.py 中 RunningStats 保持一致。

输出
----
result/eval/ic_stats/{factor_name}/cs_ic_stats.csv
result/eval/ic_stats/{factor_name}/ts_ic_stats.csv

CS 列：ret_horizon, session, factor_window, factor_col,
        ic_mean, rankic_mean, ic_std, rankic_std, icir, rankic_ir, n_days
TS 列：ret_horizon, session, factor_window, factor_col,
        ic_mean, rankic_mean, icir, rankic_ir, n_days, n_stocks
"""

import os
import glob
import re

import numpy as np
import pandas as pd


def _parse_window(fc: str, factor_name: str) -> int:
    """从因子列名提取窗口数值（用于排序）。

    去掉 '{factor_name}_' 前缀后，取剩余部分的第一个整数。
    e.g. bap_15m → 15, acc_mom_25_50t → 25, neg_skew_30m → 30
    """
    suffix = fc[len(factor_name) + 1:]
    m = re.search(r'\d+', suffix)
    return int(m.group()) if m else 0

_RET_HORIZONS_DEFAULT = ["ret100", "ret200", "ret300"]
_SESSIONS             = ["all", "am", "pm"]

# 向后兼容别名
_RET_HORIZONS = _RET_HORIZONS_DEFAULT


# ── CS-IC ─────────────────────────────────────────────────────────────────────

def _cs_stats_one(csv_dir: str, factor_cols: list[str]) -> dict:
    """
    读取某个 (ret_horizon, session) 目录下所有日期文件，
    返回每个因子列的统计量字典。
    """
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        return {}

    daily_ic     = {fc: [] for fc in factor_cols}
    daily_rankic = {fc: [] for fc in factor_cols}

    for f in files:
        df = pd.read_csv(f)
        for fc in factor_cols:
            ic_col  = f"ic_{fc}"
            ric_col = f"rankic_{fc}"
            if ic_col in df.columns:
                daily_ic[fc].append(df[ic_col].mean())
            if ric_col in df.columns:
                daily_rankic[fc].append(df[ric_col].mean())

    rows = {}
    for fc in factor_cols:
        ic_arr  = np.array(daily_ic[fc],     dtype=np.float64)
        ric_arr = np.array(daily_rankic[fc], dtype=np.float64)

        ic_mean      = np.nanmean(ic_arr)
        ic_std       = np.nanstd(ic_arr, ddof=1) if len(ic_arr) > 1 else np.nan
        rankic_mean  = np.nanmean(ric_arr)
        rankic_std   = np.nanstd(ric_arr, ddof=1) if len(ric_arr) > 1 else np.nan

        rows[fc] = {
            "ic_mean":    ic_mean,
            "rankic_mean": rankic_mean,
            "ic_std":     ic_std,
            "rankic_std": rankic_std,
            "icir":       ic_mean / ic_std if (not np.isnan(ic_std) and ic_std > 1e-12) else np.nan,
            "rankic_ir":  rankic_mean / rankic_std if (not np.isnan(rankic_std) and rankic_std > 1e-12) else np.nan,
            "n_days":     len(ic_arr),
        }
    return rows


def compute_cs_stats(eval_root: str, factor_name: str, ret_horizons: list | None = None) -> pd.DataFrame:
    base_dir = os.path.join(eval_root, "cs_ic", factor_name)

    # 从第一个可用文件推断因子列名
    first_file = next(
        (glob.glob(os.path.join(base_dir, d, "*.csv"))[0]
         for d in os.listdir(base_dir)
         if glob.glob(os.path.join(base_dir, d, "*.csv"))),
        None,
    )
    if first_file is None:
        raise FileNotFoundError(f"cs_ic 结果目录为空：{base_dir}")

    sample_df = pd.read_csv(first_file, nrows=0)
    factor_cols = [c[3:] for c in sample_df.columns if c.startswith("ic_")]

    horizons = ret_horizons if ret_horizons is not None else _RET_HORIZONS_DEFAULT
    records = []
    for ret_h in horizons:
        for sess in _SESSIONS:
            csv_dir = os.path.join(base_dir, f"{ret_h}_{sess}")
            stats = _cs_stats_one(csv_dir, factor_cols)
            for fc, s in stats.items():
                window = _parse_window(fc, factor_name)
                records.append({"ret_horizon": ret_h, "session": sess,
                                 "factor_window": window, "factor_col": fc, **s})

    return pd.DataFrame(records).sort_values(
        ["ret_horizon", "session", "factor_window"]
    ).reset_index(drop=True)


# ── TS-IC ─────────────────────────────────────────────────────────────────────

def _stock_ir(series: pd.Series) -> float:
    """单只股票的 ICIR：均值 / 总体标准差（ddof=0）。有效样本 < 2 时返回 NaN。"""
    vals = series.dropna()
    if len(vals) < 2:
        return np.nan
    s = vals.std(ddof=0)
    return float(vals.mean() / s) if s > 1e-12 else np.nan


def _ts_stats_one(csv_dir: str, factor_cols: list[str]) -> dict:
    """
    读取某个 (ret_horizon, session) 目录下所有日期文件。

    方法 C：
    - 对每只股票收集其逐日 IC 序列
    - 每只股票 ICIR = mean / std(ddof=0)
    - 最终 ICIR = mean(各股票 ICIR)
    """
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        return {}

    dfs = [pd.read_csv(f, dtype={"SecurityID": str}) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    rows = {}
    for fc in factor_cols:
        ic_col  = f"ts_ic_{fc}"
        ric_col = f"ts_rankic_{fc}"
        if ic_col not in combined.columns:
            continue

        grouped_ic  = combined.groupby("SecurityID")[ic_col]
        grouped_ric = combined.groupby("SecurityID")[ric_col]

        # 每只股票的逐日 IC 均值（用于 ic_mean）
        stock_ic_means  = grouped_ic.mean()
        stock_ric_means = grouped_ric.mean()

        # 每只股票的 ICIR（mean / std, ddof=0）
        stock_ic_ir  = grouped_ic.apply(_stock_ir)
        stock_ric_ir = grouped_ric.apply(_stock_ir)

        rows[fc] = {
            "ic_mean":     stock_ic_means.mean(),
            "rankic_mean": stock_ric_means.mean(),
            "icir":        stock_ic_ir.mean(),
            "rankic_ir":   stock_ric_ir.mean(),
            "n_days":      len(files),
            "n_stocks":    len(stock_ic_means),
        }
    return rows


def compute_ts_stats(eval_root: str, factor_name: str, ret_horizons: list | None = None) -> pd.DataFrame:
    base_dir = os.path.join(eval_root, "ts_ic", factor_name)

    first_file = next(
        (glob.glob(os.path.join(base_dir, d, "*.csv"))[0]
         for d in os.listdir(base_dir)
         if glob.glob(os.path.join(base_dir, d, "*.csv"))),
        None,
    )
    if first_file is None:
        raise FileNotFoundError(f"ts_ic 结果目录为空：{base_dir}")

    sample_df = pd.read_csv(first_file, nrows=0)
    factor_cols = [c[6:] for c in sample_df.columns if c.startswith("ts_ic_")]

    horizons = ret_horizons if ret_horizons is not None else _RET_HORIZONS_DEFAULT
    records = []
    for ret_h in horizons:
        for sess in _SESSIONS:
            csv_dir = os.path.join(base_dir, f"{ret_h}_{sess}")
            stats = _ts_stats_one(csv_dir, factor_cols)
            for fc, s in stats.items():
                window = _parse_window(fc, factor_name)
                records.append({"ret_horizon": ret_h, "session": sess,
                                 "factor_window": window, "factor_col": fc, **s})

    return pd.DataFrame(records).sort_values(
        ["ret_horizon", "session", "factor_window"]
    ).reset_index(drop=True)


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def run_ic_stats(eval_root: str, factor_name: str, ret_horizons: list | None = None):
    out_dir = os.path.join(eval_root, "ic_stats", factor_name)
    os.makedirs(out_dir, exist_ok=True)

    cs_df = compute_cs_stats(eval_root, factor_name, ret_horizons)
    cs_path = os.path.join(out_dir, "cs_ic_stats.csv")
    cs_df.to_csv(cs_path, index=False)
    print(f"CS-IC 统计完成：{cs_path}")

    ts_df = compute_ts_stats(eval_root, factor_name, ret_horizons)
    ts_path = os.path.join(out_dir, "ts_ic_stats.csv")
    ts_df.to_csv(ts_path, index=False)
    print(f"TS-IC 统计完成：{ts_path}")
