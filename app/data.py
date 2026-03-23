"""
数据加载层。所有文件读取和缓存逻辑集中在此。
"""

import glob
import os

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
def load_daily_ic(
    factor_name: str, ic_type: str, ret_horizon: str, session: str
) -> pd.DataFrame:
    """
    读取逐日 IC 文件并聚合为日度时序。

    CS：每日对所有时间点取均值 → 日度 IC
    TS：每日对所有股票取均值   → 日度 IC
    """
    csv_dir = os.path.join(
        config.EVAL_ROOT, f"{ic_type}_ic", factor_name, f"{ret_horizon}_{session}"
    )
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        return pd.DataFrame()

    rows = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f, dtype={"SecurityID": str})
        row = {"Date": day}
        if ic_type == "cs":
            for c in df.columns:
                if c.startswith("ic_") or c.startswith("rankic_"):
                    row[c] = df[c].mean()
        else:
            for c in df.columns:
                if c.startswith("ts_ic_") or c.startswith("ts_rankic_"):
                    row[c] = df[c].mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)
