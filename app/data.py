"""
数据加载层。所有文件读取和缓存逻辑集中在此。
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

import config


def available_factors() -> list[str]:
    """返回已有 ic_stats 结果的因子列表。"""
    stats_dir = os.path.join(config.EVAL_ROOT, "ic_stats")
    if not os.path.exists(stats_dir):
        return []
    return sorted(
        d for d in os.listdir(stats_dir)
        if os.path.isdir(os.path.join(stats_dir, d))
    )


def sort_factor_cols(cols: list[str]) -> list[str]:
    """按列名中的数字排序，如 mom_5m < mom_10m < mom_45m。"""
    def key(c):
        m = re.search(r"(\d+)", c)
        return int(m.group(1)) if m else 0
    return sorted(cols, key=key)


def available_quantile_dates(factor_name: str, ret_horizon: str, session: str) -> list[str]:
    """返回 cs_quantile 目录下所有可用日期。"""
    csv_dir = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name, f"{ret_horizon}_{session}"
    )
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def available_cs_dates(factor_name: str, ret_horizon: str, session: str) -> list[str]:
    """返回 cs_ic 目录下所有可用日期（字符串列表，如 ['20250102', ...]）。"""
    csv_dir = os.path.join(
        config.EVAL_ROOT, "cs_ic", factor_name, f"{ret_horizon}_{session}"
    )
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


@st.cache_data
def load_ic_stats(factor_name: str, ic_type: str) -> pd.DataFrame:
    """读取 ic_stats 汇总表。ic_type: 'cs' 或 'ts'"""
    path = os.path.join(
        config.EVAL_ROOT, "ic_stats", factor_name, f"{ic_type}_ic_stats.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_cs_daily_trend(
    factor_name: str, ret_horizon: str, session: str
) -> pd.DataFrame:
    """
    CS-IC 跨日趋势：每个交易日对所有时间点取均值 → 日度 IC 序列。
    X 轴 = Date，适合观察因子信号跨日的稳定性与趋势。
    返回列：Date, ic_{fc}, rankic_{fc}, ...
    """
    csv_dir = os.path.join(
        config.EVAL_ROOT, "cs_ic", factor_name, f"{ret_horizon}_{session}"
    )
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        return pd.DataFrame()

    rows = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f, dtype={"SampleTime": str})
        row = {"Date": day}
        for c in df.columns:
            if c.startswith("ic_") or c.startswith("rankic_"):
                row[c] = df[c].mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_quantile_summary(
    factor_name: str, ret_horizon: str, session: str, factor_col: str,
) -> pd.DataFrame:
    """
    读取预聚合的 _summary.csv，过滤出指定 factor_col 的5组均值。
    返回 DataFrame，列：group(1~5), mean_ret, long_short
    """
    path = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name,
        f"{ret_horizon}_{session}", "_summary.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()

    df  = pd.read_csv(path)
    row = df[df["factor_col"] == factor_col]
    if row.empty:
        return pd.DataFrame()

    g_cols = [f"g{g}" for g in range(1, 6)]
    result = pd.DataFrame({
        "group":    list(range(1, 6)),
        "mean_ret": [row[c].values[0] for c in g_cols],
    })
    result["long_short"] = result["mean_ret"].iloc[-1] - result["mean_ret"].iloc[0]
    return result


@st.cache_data
def load_quantile_daily(
    factor_name: str, ret_horizon: str, session: str, factor_col: str,
) -> pd.DataFrame:
    """
    读取预聚合的 _daily.csv，过滤出指定 factor_col 的日度5组均值序列。
    返回列：Date, g1, g2, g3, g4, g5, long_short
    """
    path = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name,
        f"{ret_horizon}_{session}", "_daily.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()

    df  = pd.read_csv(path, dtype={"Date": str})
    out = df[df["factor_col"] == factor_col].copy()
    if out.empty:
        return pd.DataFrame()

    out["Date"]       = pd.to_datetime(out["Date"])
    out["long_short"] = out["g5"] - out["g1"]
    return out.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_quantile_one_day(
    factor_name: str, ret_horizon: str, session: str, day: str, factor_col: str,
) -> pd.DataFrame:
    """
    截面分层单日日内曲线：读取指定日期文件。
    返回列：SampleTime, g1, g2, g3, g4, g5
    """
    path = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name,
        f"{ret_horizon}_{session}", f"{day}.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()

    g_cols = [f"g{g}_{factor_col}" for g in range(1, 6)]
    df = pd.read_csv(path, dtype={"SampleTime": str})
    cols_present = [c for c in g_cols if c in df.columns]
    if not cols_present:
        return pd.DataFrame()

    out = df[["SampleTime"] + cols_present].copy()
    out.columns = ["SampleTime"] + [f"g{i+1}" for i in range(len(cols_present))]
    return out


@st.cache_data
def load_cs_one_day(
    factor_name: str, ret_horizon: str, session: str, day: str
) -> pd.DataFrame:
    """
    CS-IC 单日日内曲线：读取指定日期的文件。
    X 轴 = SampleTime，适合观察因子信号在日内的分布规律。
    返回列：SampleTime, ic_{fc}, rankic_{fc}, ...
    """
    path = os.path.join(
        config.EVAL_ROOT, "cs_ic", factor_name, f"{ret_horizon}_{session}", f"{day}.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"SampleTime": str})
