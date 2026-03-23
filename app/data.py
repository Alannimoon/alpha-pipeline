"""
数据加载层。所有文件读取和缓存逻辑集中在此。
"""

import glob
import os
import re

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


@st.cache_data
def load_ic_stats(factor_name: str, ic_type: str) -> pd.DataFrame:
    """
    读取 ic_stats 汇总表。
    ic_type: "cs" 或 "ts"
    """
    path = os.path.join(
        config.EVAL_ROOT, "ic_stats", factor_name, f"{ic_type}_ic_stats.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_ts_daily(
    factor_name: str, ret_horizon: str, session: str
) -> pd.DataFrame:
    """
    TS-IC：读取逐日文件，每日对所有股票取均值 → 日度均值时序。
    返回 DataFrame，列：Date, ts_ic_{fc}, ts_rankic_{fc}, ...
    """
    csv_dir = os.path.join(
        config.EVAL_ROOT, "ts_ic", factor_name, f"{ret_horizon}_{session}"
    )
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        return pd.DataFrame()

    rows = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f, dtype={"SecurityID": str})
        row = {"Date": day}
        for c in df.columns:
            if c.startswith("ts_ic_") or c.startswith("ts_rankic_"):
                row[c] = df[c].mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_cs_intraday(
    factor_name: str, ret_horizon: str, session: str
) -> pd.DataFrame:
    """
    CS-IC：读取所有日期文件，按 SampleTime 分组取均值 → 日内 IC 模式。
    返回 DataFrame，列：SampleTime, ic_{fc}, rankic_{fc}, ...
    """
    csv_dir = os.path.join(
        config.EVAL_ROOT, "cs_ic", factor_name, f"{ret_horizon}_{session}"
    )
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        return pd.DataFrame()

    dfs = [pd.read_csv(f, dtype={"SampleTime": str}) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    ic_cols = [c for c in combined.columns if c.startswith("ic_") or c.startswith("rankic_")]
    return combined.groupby("SampleTime")[ic_cols].mean().reset_index()
