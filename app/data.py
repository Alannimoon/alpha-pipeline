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


def _horizon_step(ret_horizon: str) -> int:
    """从 'ret100' 提取步长 100。"""
    return int(ret_horizon.replace("ret", ""))


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
def _load_quantile_raw_nonolap(
    factor_name: str, ret_horizon: str, session: str, factor_col: str,
) -> pd.DataFrame:
    """
    内部缓存：读取所有日期文件，非重叠采样后拼接，返回原始（未累计）数据。

    步长 = ret_horizon 对应的 tick 数（100/200/300），从每天第 0 行开始采样。
    每天末尾不足一个完整窗口的 tick 自然缺失（ret_fwd 为 NaN 不参与后续计算）。

    返回列：Date, SampleTime, g1, g2, g3, g4, g5
    """
    csv_dir = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name, f"{ret_horizon}_{session}"
    )
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return pd.DataFrame()

    step = _horizon_step(ret_horizon)
    g_cols_raw = [f"g{g}_{factor_col}" for g in range(1, 6)]

    dfs = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df  = pd.read_csv(f, dtype={"SampleTime": str})
        cols_present = [c for c in g_cols_raw if c in df.columns]
        if not cols_present:
            continue
        df_sub = df[["SampleTime"] + cols_present].iloc[::step].copy()
        df_sub.insert(0, "Date", day)
        dfs.append(df_sub)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.rename(columns={f"g{g}_{factor_col}": f"g{g}" for g in range(1, 6)})
    return combined  # 列：Date, SampleTime, g1, g2, g3, g4, g5


@st.cache_data
def load_quantile_tick_one_day(
    factor_name: str, ret_horizon: str, session: str, day: str, factor_col: str,
) -> pd.DataFrame:
    """
    单日 tick 级别累计收益。

    读取指定日期文件，非重叠采样后在日内做 cumsum()（从 0 开始，不跨日）。
    返回列：SampleTime, g1, g2, g3, g4, g5, long_short
    """
    path = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name,
        f"{ret_horizon}_{session}", f"{day}.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()

    step = _horizon_step(ret_horizon)
    g_cols_raw = [f"g{g}_{factor_col}" for g in range(1, 6)]

    df = pd.read_csv(path, dtype={"SampleTime": str})
    cols_present = [c for c in g_cols_raw if c in df.columns]
    if not cols_present:
        return pd.DataFrame()

    df_sub = df[["SampleTime"] + cols_present].iloc[::step].copy()
    df_sub = df_sub.rename(columns={f"g{g}_{factor_col}": f"g{g}" for g in range(1, 6)})

    g_cols = [c for c in ["g1", "g2", "g3", "g4", "g5"] if c in df_sub.columns]
    df_sub[g_cols] = df_sub[g_cols].cumsum()
    df_sub["long_short"] = df_sub["g5"] - df_sub["g1"]
    return df_sub.reset_index(drop=True)


@st.cache_data
def load_quantile_tick_cum(
    factor_name: str, ret_horizon: str, session: str, factor_col: str,
) -> pd.DataFrame:
    """
    tick 级别累计收益。

    对所有交易日的非重叠采样点（共 N_days × ~(4740/step) 行），
    跨日连续做 cumsum()，每行代表该持仓期结束时的累计收益。

    返回列：Date, SampleTime, g1, g2, g3, g4, g5, long_short
    """
    df = _load_quantile_raw_nonolap(factor_name, ret_horizon, session, factor_col)
    if df.empty:
        return df

    g_cols = [f"g{g}" for g in range(1, 6)]
    out = df.copy()
    out[g_cols] = out[g_cols].cumsum()
    out["long_short"] = out["g5"] - out["g1"]
    return out


@st.cache_data
def load_quantile_daily_cum(
    factor_name: str, ret_horizon: str, session: str, factor_col: str,
) -> pd.DataFrame:
    """
    日频累计收益。

    每天先将所有非重叠采样点的收益求和（代表当天策略总收益），
    再按日期顺序做 cumsum()，每行代表截至该交易日的累计收益。

    与 tick 视图的关系：日频累计值 = tick 累计曲线上每天最后一个采样点的值。

    返回列：Date, g1, g2, g3, g4, g5, long_short
    """
    df = _load_quantile_raw_nonolap(factor_name, ret_horizon, session, factor_col)
    if df.empty:
        return df

    g_cols = [f"g{g}" for g in range(1, 6)]
    daily = df.groupby("Date")[g_cols].sum().reset_index()
    daily = daily.sort_values("Date").reset_index(drop=True)
    daily[g_cols] = daily[g_cols].cumsum()
    daily["long_short"] = daily["g5"] - daily["g1"]
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily


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
