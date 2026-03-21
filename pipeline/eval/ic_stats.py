"""
IC 汇总统计模块。

读取 cs_ic / ts_ic 结果文件，输出每个（因子窗口, 收益率窗口, session）
组合的 IC 均值、标准差、ICIR。

CS-IC 统计逻辑
--------------
1. 对每个交易日，将日内所有时间点的 IC 取均值 → 日度 IC 均值
2. 跨所有交易日对日度均值取均值 → ic_mean
3. 跨所有交易日对日度均值取标准差 → ic_std（仅 1 天时为 NaN）
4. ICIR = ic_mean / ic_std

TS-IC 统计逻辑
--------------
1. 对每只股票，跨所有交易日取 IC 均值 → 个股 IC 均值
2. 对所有个股均值取均值 → ic_mean
3. 对所有个股均值取标准差 → ic_std
4. ICIR = ic_mean / ic_std

输出
----
data/eval/ic_stats/{factor_name}/cs_ic_stats.csv
data/eval/ic_stats/{factor_name}/ts_ic_stats.csv

列：ret_horizon, session, factor_window, ic_mean, rankic_mean,
    ic_std, rankic_std, icir, rankic_ir, n_days
"""

import os
import glob

import numpy as np
import pandas as pd

_RET_HORIZONS = ["ret100", "ret200", "ret300"]
_SESSIONS     = ["all", "am", "pm"]


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


def compute_cs_stats(eval_root: str, factor_name: str) -> pd.DataFrame:
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

    records = []
    for ret_h in _RET_HORIZONS:
        for sess in _SESSIONS:
            csv_dir = os.path.join(base_dir, f"{ret_h}_{sess}")
            stats = _cs_stats_one(csv_dir, factor_cols)
            for fc, s in stats.items():
                window = int(fc.split("_")[1].replace("m", ""))
                records.append({"ret_horizon": ret_h, "session": sess,
                                 "factor_window": window, "factor_col": fc, **s})

    return pd.DataFrame(records).sort_values(
        ["ret_horizon", "session", "factor_window"]
    ).reset_index(drop=True)


# ── TS-IC ─────────────────────────────────────────────────────────────────────

def _ts_stats_one(csv_dir: str, factor_cols: list[str]) -> dict:
    """
    读取某个 (ret_horizon, session) 目录下所有日期文件，
    对每只股票跨日取均值后，再对所有股票取均值/标准差。
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

        # 每只股票跨日均值
        per_stock_ic  = combined.groupby("SecurityID")[ic_col].mean()
        per_stock_ric = combined.groupby("SecurityID")[ric_col].mean()

        ic_mean     = per_stock_ic.mean()
        ic_std      = per_stock_ic.std(ddof=1) if len(per_stock_ic) > 1 else np.nan
        rankic_mean = per_stock_ric.mean()
        rankic_std  = per_stock_ric.std(ddof=1) if len(per_stock_ric) > 1 else np.nan

        rows[fc] = {
            "ic_mean":    ic_mean,
            "rankic_mean": rankic_mean,
            "ic_std":     ic_std,
            "rankic_std": rankic_std,
            "icir":       ic_mean / ic_std if (not np.isnan(ic_std) and ic_std > 1e-12) else np.nan,
            "rankic_ir":  rankic_mean / rankic_std if (not np.isnan(rankic_std) and rankic_std > 1e-12) else np.nan,
            "n_days":     len(files),
        }
    return rows


def compute_ts_stats(eval_root: str, factor_name: str) -> pd.DataFrame:
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

    records = []
    for ret_h in _RET_HORIZONS:
        for sess in _SESSIONS:
            csv_dir = os.path.join(base_dir, f"{ret_h}_{sess}")
            stats = _ts_stats_one(csv_dir, factor_cols)
            for fc, s in stats.items():
                window = int(fc.split("_")[1].replace("m", ""))
                records.append({"ret_horizon": ret_h, "session": sess,
                                 "factor_window": window, "factor_col": fc, **s})

    return pd.DataFrame(records).sort_values(
        ["ret_horizon", "session", "factor_window"]
    ).reset_index(drop=True)


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def run_ic_stats(eval_root: str, factor_name: str):
    out_dir = os.path.join(eval_root, "ic_stats", factor_name)
    os.makedirs(out_dir, exist_ok=True)

    cs_df = compute_cs_stats(eval_root, factor_name)
    cs_path = os.path.join(out_dir, "cs_ic_stats.csv")
    cs_df.to_csv(cs_path, index=False)
    print(f"CS-IC 统计完成：{cs_path}")

    ts_df = compute_ts_stats(eval_root, factor_name)
    ts_path = os.path.join(out_dir, "ts_ic_stats.csv")
    ts_df.to_csv(ts_path, index=False)
    print(f"TS-IC 统计完成：{ts_path}")
