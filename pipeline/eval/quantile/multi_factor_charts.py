"""
多因子分层图表与汇总模块。

包含将逐日 CSV 结果整理为汇总文件和图表的所有函数：
  _build_daily            → _daily.csv（每日各组均值）
  _build_summary          → _summary.csv（跨日整体均值）
  _build_cum_tick         → _cum_tick_{date}.csv（日内累计）
  _build_cum_daily        → _cum_daily.csv（跨日累计末值）
  _build_cum_tick_chart   → _chart_tick.png（跨日 tick 连续曲线）
  _build_intraday_slot_charts → _chart_slot_*.png（8个30分钟时段图）
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .factor_score import N_GROUPS


# ── 颜色 & 时段常量 ────────────────────────────────────────────────────────────

_GROUP_COLORS = [
    "#d62728", "#e07b2a", "#d4a017", "#9acd32", "#3cb371",
    "#2ca02c", "#17becf", "#1f77b4", "#7b52ab", "#8c564b",
]

# 8个30分钟时间窗口（左闭右开，HH:MM:SS字符串直接比较）
_INTRADAY_SLOTS = [
    ("09:30-10:00", "09:30:00", "10:00:00"),
    ("10:00-10:30", "10:00:00", "10:30:00"),
    ("10:30-11:00", "10:30:00", "11:00:00"),
    ("11:00-11:30", "11:00:00", "11:30:00"),
    ("13:00-13:30", "13:00:00", "13:30:00"),
    ("13:30-14:00", "13:30:00", "14:00:00"),
    ("14:00-14:30", "14:00:00", "14:30:00"),
    ("14:30-14:57", "14:30:00", "23:59:59"),
]


# ── 汇总生成 ───────────────────────────────────────────────────────────────────

def _build_daily(csv_dir: str) -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return

    rows = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df  = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in df.columns]
        if not g_cols:
            continue
        day_means = df[g_cols].mean()
        row = {"Date": day}
        for gc in g_cols:
            row[gc] = day_means.get(gc, np.nan)
        rows.append(row)

    if not rows:
        return
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, "_daily.csv"), index=False)


def _build_summary(csv_dir: str) -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return

    daily_means = []
    for f in files:
        df = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in df.columns]
        if g_cols:
            daily_means.append(df[g_cols].mean())

    if not daily_means:
        return

    overall = pd.DataFrame(daily_means).mean()
    summary = pd.DataFrame([overall.to_dict()])
    summary.insert(0, "label", "composite")
    summary.to_csv(os.path.join(csv_dir, "_summary.csv"), index=False)


def _build_cum_tick(csv_dir: str) -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return

    file_iter = tqdm(files, desc="cum_tick") if tqdm else files
    for f in file_iter:
        day    = os.path.splitext(os.path.basename(f))[0]
        df     = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in df.columns]
        if not g_cols:
            continue

        sub = df[["SampleTime"] + g_cols].copy()
        sub[g_cols] = sub[g_cols].cumsum()
        sub["long_short"] = sub[f"g{N_GROUPS}"] - sub["g1"]

        sub.to_csv(os.path.join(csv_dir, f"_cum_tick_{day}.csv"), index=False)


def _build_cum_daily(csv_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(csv_dir, "_cum_tick_*.csv")))
    if not files:
        return

    rows = []
    for f in files:
        m = re.search(r"_cum_tick_(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        day = m.group(1)
        df  = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in df.columns]
        ls_cols = g_cols + (["long_short"] if "long_short" in df.columns else [])

        row: dict = {"Date": day}
        for gc in ls_cols:
            valid = df[gc].dropna()
            row[gc] = float(valid.iloc[-1]) if len(valid) > 0 else np.nan
        row["n_ticks"] = int(df[g_cols[0]].notna().sum()) if g_cols else 0
        rows.append(row)

    if not rows:
        return

    all_last = (
        pd.DataFrame(rows)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    g_cols   = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in all_last.columns]
    cum_cols = g_cols + ["n_ticks"]
    all_last[cum_cols] = all_last[cum_cols].cumsum()
    all_last["long_short"] = all_last[f"g{N_GROUPS}"] - all_last["g1"]

    keep = ["Date"] + g_cols + ["long_short", "n_ticks"]
    all_last[keep].to_csv(os.path.join(csv_dir, "_cum_daily.csv"), index=False)


# ── 跨日 tick 连续曲线图 ────────────────────────────────────────────────────────

def _build_cum_tick_chart(csv_dir: str) -> None:
    tick_files = sorted(glob.glob(os.path.join(csv_dir, "_cum_tick_*.csv")))
    if not tick_files:
        return

    daily_path = os.path.join(csv_dir, "_cum_daily.csv")
    if not os.path.exists(daily_path):
        return
    daily_df = pd.read_csv(daily_path, dtype={"Date": str}).set_index("Date")

    g_cols   = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in daily_df.columns]
    all_cols = g_cols + (["long_short"] if "long_short" in daily_df.columns else [])

    dfs = []
    prev_date = None
    for f in tick_files:
        m = re.search(r"_cum_tick_(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        day = m.group(1)
        df  = pd.read_csv(f, dtype={"SampleTime": str})
        avail_cols = [c for c in all_cols if c in df.columns]
        df = df[["SampleTime"] + avail_cols].copy()
        if df.empty:
            prev_date = day
            continue
        if prev_date is not None and prev_date in daily_df.index:
            offset = daily_df.loc[prev_date, avail_cols]
            df[avail_cols] = df[avail_cols] + offset.values
        df.insert(0, "Date", day)
        dfs.append(df)
        prev_date = day

    if not dfs:
        return

    tick_df = pd.concat(dfs, ignore_index=True)
    x = np.arange(len(tick_df))

    day_starts = tick_df.groupby("Date", sort=False).apply(
        lambda g: g.index[0] - tick_df.index[0], include_groups=False
    )

    # 多空平均 PnL 注释
    ls_label = f"L/S(g{N_GROUPS}-g1)"
    if "n_ticks" in daily_df.columns and "long_short" in daily_df.columns:
        total_ticks = daily_df["n_ticks"].iloc[-1]
        total_ls    = daily_df["long_short"].iloc[-1]
        if pd.notna(total_ticks) and pd.notna(total_ls) and total_ticks > 0:
            avg_pnl_bps = total_ls / total_ticks * 1e4
            ls_label = f"L/S(g{N_GROUPS}-g1)  avg={avg_pnl_bps:+.3f}bps/tick"

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, gc in enumerate(g_cols):
        if gc in tick_df.columns:
            ax.plot(x, tick_df[gc],
                    color=_GROUP_COLORS[i % len(_GROUP_COLORS)],
                    alpha=0.7, linewidth=0.7, label=gc)
    if "long_short" in tick_df.columns:
        ax.plot(x, tick_df["long_short"], color="black",
                linewidth=1.2, label=ls_label)

    tick_positions, tick_labels = [], []
    prev_month = None
    for date, pos in day_starts.items():
        month = date[:6]
        if month != prev_month:
            ax.axvline(pos, color="gray", linewidth=0.4, linestyle="--")
            tick_positions.append(pos)
            tick_labels.append(f"{date[:4]}-{date[4:6]}")
            prev_month = month

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax.set_ylabel("Cumulative Return (tick-level)")
    ax.set_title("Multi-Factor Composite  Cross-Day Tick Cumulative Return")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left", ncol=N_GROUPS + 1, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(csv_dir, "_chart_tick.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 日内时间段分层图 ────────────────────────────────────────────────────────────

def _build_intraday_slot_charts(csv_dir: str) -> None:
    """
    对每个 30 分钟时段，跨日累计各组日内该时段的收益总和，
    生成 _chart_slot_{idx}_{label}.png。
    """
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return

    first_cols = pd.read_csv(files[0], nrows=0).columns.tolist()
    g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in first_cols]
    if not g_cols:
        return

    for slot_idx, (label, t_start, t_end) in enumerate(_INTRADAY_SLOTS, start=1):
        daily_rows = []
        for f in files:
            day = os.path.splitext(os.path.basename(f))[0]
            df  = pd.read_csv(f, dtype={"SampleTime": str})

            mask = (df["SampleTime"] >= t_start) & (df["SampleTime"] < t_end)
            sub  = df.loc[mask, [c for c in g_cols if c in df.columns]]

            if sub.empty:
                continue

            row = {"Date": day}
            for gc in g_cols:
                row[gc] = float(sub[gc].sum(skipna=True)) if gc in sub.columns else np.nan
            row["n_ticks"] = int(sub[g_cols[0]].notna().sum()) if g_cols[0] in sub.columns else 0
            daily_rows.append(row)

        if not daily_rows:
            continue

        daily = (
            pd.DataFrame(daily_rows)
            .sort_values("Date")
            .reset_index(drop=True)
        )

        cum_cols = [gc for gc in g_cols if gc in daily.columns] + ["n_ticks"]
        daily[cum_cols] = daily[cum_cols].cumsum()
        daily["long_short"] = daily[f"g{N_GROUPS}"] - daily["g1"]

        total_ticks = daily["n_ticks"].iloc[-1]
        total_ls    = daily["long_short"].iloc[-1]
        ls_label = f"L/S(g{N_GROUPS}-g1)"
        if pd.notna(total_ticks) and pd.notna(total_ls) and total_ticks > 0:
            avg_pnl_bps = total_ls / total_ticks * 1e4
            ls_label = f"L/S(g{N_GROUPS}-g1)  avg={avg_pnl_bps:+.3f}bps/tick"

        x = np.arange(len(daily))
        fig, ax = plt.subplots(figsize=(14, 5))

        for i, gc in enumerate(g_cols):
            if gc in daily.columns:
                ax.plot(x, daily[gc],
                        color=_GROUP_COLORS[i % len(_GROUP_COLORS)],
                        alpha=0.7, linewidth=0.9, label=gc)
        if "long_short" in daily.columns:
            ax.plot(x, daily["long_short"], color="black",
                    linewidth=1.4, label=ls_label)

        tick_positions, tick_labels_ax = [], []
        prev_month = None
        for xi, date_val in enumerate(daily["Date"]):
            month = str(date_val)[:6]
            if month != prev_month:
                ax.axvline(xi, color="gray", linewidth=0.4, linestyle="--")
                tick_positions.append(xi)
                tick_labels_ax.append(f"{str(date_val)[:4]}-{str(date_val)[4:6]}")
                prev_month = month

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
        ax.set_ylabel("Cumulative Return")
        ax.set_title(
            f"Multi-Factor Composite  Slot {slot_idx}: {label}  "
            f"Daily Cumulative Return"
        )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_ax, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper left", ncol=N_GROUPS + 1, fontsize=7)
        fig.tight_layout()

        safe_label = label.replace(":", "").replace("-", "_")
        fig.savefig(
            os.path.join(csv_dir, f"_chart_slot_{slot_idx}_{safe_label}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
