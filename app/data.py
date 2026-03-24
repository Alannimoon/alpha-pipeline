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
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
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
def load_quantile_tick_one_day(
    factor_name: str, ret_horizon: str, session: str, day: str, factor_col: str,
) -> pd.DataFrame:
    """
    单日 tick 级别累计收益，读取预计算的 _cum_tick_{day}.csv。

    值为日内从 0 开始的累计收益（不含跨日偏移），纯文件读取。
    返回列：SampleTime, g1, g2, g3, g4, g5, long_short
    """
    path = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name,
        f"{ret_horizon}_{session}", f"_cum_tick_{day}.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"SampleTime": str})
    return (
        df[df["factor_col"] == factor_col]
        .drop(columns="factor_col")
        .reset_index(drop=True)
    )


@st.cache_data
def load_quantile_tick_cum(
    factor_name: str, ret_horizon: str, session: str, factor_col: str,
) -> pd.DataFrame:
    """
    tick 级别跨日累计收益。

    读取所有 _cum_tick_{date}.csv（日内累计，从 0 开始），
    叠加 _cum_daily.csv 中前一天的结束值作为跨日 offset，
    拼接后得到连续跨日累计曲线。

    返回列：Date, SampleTime, g1, g2, g3, g4, g5, long_short
    """
    csv_dir = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name, f"{ret_horizon}_{session}"
    )
    tick_files = sorted(glob.glob(os.path.join(csv_dir, "_cum_tick_*.csv")))
    if not tick_files:
        return pd.DataFrame()

    # 读 _cum_daily.csv 获取每天结束时的跨日累计值（用于计算 offset）
    daily_path = os.path.join(csv_dir, "_cum_daily.csv")
    if not os.path.exists(daily_path):
        return pd.DataFrame()
    daily_df = pd.read_csv(daily_path, dtype={"Date": str})
    daily_fc = (
        daily_df[daily_df["factor_col"] == factor_col]
        .set_index("Date")
        .drop(columns="factor_col")
    )
    g_cols = [c for c in ["g1", "g2", "g3", "g4", "g5", "long_short"]
              if c in daily_fc.columns]

    dfs = []
    prev_date = None
    for f in tick_files:
        m = re.search(r"_cum_tick_(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        day = m.group(1)
        df  = pd.read_csv(f, dtype={"SampleTime": str})
        df  = df[df["factor_col"] == factor_col].drop(columns="factor_col").copy()
        if df.empty:
            prev_date = day
            continue

        # 叠加前一天结束时的累计值作为 offset
        if prev_date is not None and prev_date in daily_fc.index:
            offset = daily_fc.loc[prev_date, g_cols]
            df[g_cols] = df[g_cols] + offset.values

        df.insert(0, "Date", day)
        dfs.append(df)
        prev_date = day

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@st.cache_data
def load_quantile_daily_cum(
    factor_name: str, ret_horizon: str, session: str, factor_col: str,
) -> pd.DataFrame:
    """
    日频累计收益，读取预计算的 _cum_daily.csv。

    返回列：Date, g1, g2, g3, g4, g5, long_short
    """
    path = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name,
        f"{ret_horizon}_{session}", "_cum_daily.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"Date": str})
    out = (
        df[df["factor_col"] == factor_col]
        .drop(columns="factor_col")
        .reset_index(drop=True)
    )
    out["Date"] = pd.to_datetime(out["Date"])
    return out


@st.cache_data
def load_quantile_pnl_stats(
    factor_name: str, ret_horizon: str, session: str, factor_col: str,
) -> dict:
    """
    从 _cum_daily.csv 读取最后一行，返回各组总累计收益和每 tick 平均收益。

    返回 dict 包含：
      g1~g5, long_short: 总累计收益
      n_ticks: 总 tick 数
      avg_g1~avg_g5, avg_long_short: 每 tick 平均收益
    """
    path = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name,
        f"{ret_horizon}_{session}", "_cum_daily.csv"
    )
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"Date": str})
    sub = df[df["factor_col"] == factor_col]
    if sub.empty:
        return {}
    last = sub.iloc[-1]
    result = {}
    g_cols = [f"g{g}" for g in range(1, 6)] + ["long_short"]
    for c in g_cols:
        if c in last.index:
            result[c] = float(last[c])
    n_ticks = float(last["n_ticks"]) if "n_ticks" in last.index else None
    result["n_ticks"] = n_ticks
    if n_ticks and n_ticks > 0:
        for c in g_cols:
            if c in result:
                result[f"avg_{c}"] = result[c] / n_ticks
    return result


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
