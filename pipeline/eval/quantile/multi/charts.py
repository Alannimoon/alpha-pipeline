"""
多因子分层图表与汇总模块。

_build_daily            → _daily.csv（每日各组均值）
_build_summary          → _summary.csv（跨日整体均值）
_build_cum_daily        → _cum_daily.csv（跨日累计末值，直接从 parquet 计算）
_build_cum_tick_chart   → _chart_tick.png（跨日 tick 连续曲线）
_build_intraday_slot_charts → _chart_slot_*.png（8个30分钟时段图）

注：不再生成 _cum_tick_{date}.csv 中间文件。
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

_GROUP_COLORS = [
    "#d62728", "#e07b2a", "#d4a017", "#9acd32", "#3cb371",
    "#2ca02c", "#17becf", "#1f77b4", "#7b52ab", "#8c564b",
    "#e377c2", "#f7b6d2", "#bcbd22", "#dbdb8d", "#17becf",
    "#9edae5", "#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5",
]

def _g_cols(df) -> list[str]:
    """从 DataFrame 列名中提取 g1, g2, ... 列，按数字排序。"""
    return sorted([c for c in df.columns if re.match(r"^g\d+$", c)],
                  key=lambda c: int(c[1:]))


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


def _parquet_files(parquet_dir: str) -> list[str]:
    return sorted(
        f for f in glob.glob(os.path.join(parquet_dir, "*.parquet"))
        if not os.path.basename(f).startswith("_")
    )


# ── 汇总生成 ───────────────────────────────────────────────────────────────────

def _build_daily(parquet_dir: str) -> None:
    files = _parquet_files(parquet_dir)
    if not files:
        return

    rows = []
    for f in files:
        day    = os.path.splitext(os.path.basename(f))[0]
        df     = pd.read_parquet(f)
        g_cols = _g_cols(df)
        if not g_cols:
            continue
        row = {"Date": day}
        for gc in g_cols:
            row[gc] = df[gc].mean()
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(parquet_dir, "_daily.csv"), index=False)


def _build_summary(parquet_dir: str) -> None:
    files = _parquet_files(parquet_dir)
    if not files:
        return

    daily_means = []
    for f in files:
        df     = pd.read_parquet(f)
        g_cols = _g_cols(df)
        if g_cols:
            daily_means.append(df[g_cols].mean())

    if not daily_means:
        return

    overall = pd.DataFrame(daily_means).mean()
    summary = pd.DataFrame([overall.to_dict()])
    summary.insert(0, "label", "composite")
    summary.to_csv(os.path.join(parquet_dir, "_summary.csv"), index=False)


# ── 跨日累计末值（直接从 parquet sum）────────────────────────────────────────

def _build_cum_daily(parquet_dir: str) -> None:
    """
    直接对每日 parquet 求列和（= 日内 cumsum 的最后一行）→ 跨日 cumsum →
    _cum_daily.csv（Date, g1..g{N}, long_short, n_ticks,
                    ls_{slot}, n_ticks_{slot}, ...  共 8 个时段）。

    时段列命名规则：标签 "09:30-10:00" → safe 键 "0930_1000"
      ls_0930_1000      : 跨日累计多空收益（g_last - g1，仅限该时段 tick）
      n_ticks_0930_1000 : 跨日累计有效 tick 数（该时段）
    """
    files = _parquet_files(parquet_dir)
    if not files:
        return

    rows = []
    for f in files:
        day    = os.path.splitext(os.path.basename(f))[0]
        df     = pd.read_parquet(f)
        g_cols = _g_cols(df)
        if not g_cols:
            continue
        row = {"Date": day}
        # ── 全时段 ────────────────────────────────────────────────────────────
        for gc in g_cols:
            row[gc] = float(df[gc].sum(skipna=True))
        row["n_ticks"] = int(df[g_cols[0]].notna().sum())
        # ── 8 个 30 分钟时段 ──────────────────────────────────────────────────
        for label, t_start, t_end in _INTRADAY_SLOTS:
            safe = label.replace(":", "").replace("-", "_")
            mask = (df["SampleTime"] >= t_start) & (df["SampleTime"] < t_end)
            sub  = df.loc[mask]
            row[f"ls_{safe}"]      = (float(sub[g_cols[-1]].sum(skipna=True))
                                      - float(sub[g_cols[0]].sum(skipna=True)))
            row[f"n_ticks_{safe}"] = int(sub[g_cols[0]].notna().sum())
        rows.append(row)

    if not rows:
        return

    all_days = (
        pd.DataFrame(rows)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    g_cols = _g_cols(all_days)

    # 全时段累计列 + 8 个时段累计列一并 cumsum
    slot_extra = []
    for label, _, _ in _INTRADAY_SLOTS:
        safe = label.replace(":", "").replace("-", "_")
        slot_extra += [f"ls_{safe}", f"n_ticks_{safe}"]

    cum_cols = g_cols + ["n_ticks"] + slot_extra
    all_days[cum_cols] = all_days[cum_cols].cumsum()
    all_days["long_short"] = all_days[g_cols[-1]] - all_days["g1"]

    keep = ["Date"] + g_cols + ["long_short", "n_ticks"] + slot_extra
    all_days[keep].to_csv(os.path.join(parquet_dir, "_cum_daily.csv"), index=False)


# ── 跨日 tick 连续曲线图 ────────────────────────────────────────────────────────

def _build_cum_tick_chart(parquet_dir: str) -> None:
    """
    直接读各日 parquet，日内 cumsum 后以 _cum_daily.csv 的前一日末值做偏移，
    拼出跨日连续 tick 曲线并画图 → _chart_tick.png。
    """
    files = _parquet_files(parquet_dir)
    if not files:
        return

    daily_path = os.path.join(parquet_dir, "_cum_daily.csv")
    if not os.path.exists(daily_path):
        return
    daily_df = pd.read_csv(daily_path, dtype={"Date": str}).set_index("Date")

    g_cols   = _g_cols(daily_df)
    all_cols = g_cols + (["long_short"] if "long_short" in daily_df.columns else [])

    dfs       = []
    prev_date = None

    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df  = pd.read_parquet(f)
        avail = [c for c in all_cols if c in df.columns]
        if not avail:
            prev_date = day
            continue

        sub = df[["SampleTime"] + avail].copy()
        sub[g_cols] = sub[g_cols].cumsum()
        sub["long_short"] = sub[g_cols[-1]] - sub["g1"]

        if prev_date is not None and prev_date in daily_df.index:
            offset = daily_df.loc[prev_date, all_cols]
            sub[all_cols] = sub[all_cols] + offset.values

        sub.insert(0, "Date", day)
        dfs.append(sub)
        prev_date = day

    if not dfs:
        return

    tick_df   = pd.concat(dfs, ignore_index=True)
    x         = np.arange(len(tick_df))
    day_starts = tick_df.groupby("Date", sort=False).apply(
        lambda g: g.index[0] - tick_df.index[0], include_groups=False
    )

    ls_label = f"L/S({g_cols[-1]}-g1)"
    if "n_ticks" in daily_df.columns and "long_short" in daily_df.columns:
        total_ticks = daily_df["n_ticks"].iloc[-1]
        total_ls    = daily_df["long_short"].iloc[-1]
        if pd.notna(total_ticks) and pd.notna(total_ls) and total_ticks > 0:
            avg_pnl_bps = total_ls / total_ticks * 1e4
            ls_label = f"L/S({g_cols[-1]}-g1)  avg={avg_pnl_bps:+.3f}bps/tick"

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
    ax.legend(loc="upper left", ncol=min(len(g_cols) + 1, 11), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(parquet_dir, "_chart_tick.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 日内时间段分层图 ────────────────────────────────────────────────────────────

def _build_intraday_slot_charts(parquet_dir: str) -> None:
    """对每个 30 分钟时段，跨日累计各组该时段收益 → _chart_slot_{idx}_{label}.png。"""
    files = _parquet_files(parquet_dir)
    if not files:
        return

    first_df = pd.read_parquet(files[0])
    g_cols = _g_cols(first_df)
    if not g_cols:
        return

    for slot_idx, (label, t_start, t_end) in enumerate(_INTRADAY_SLOTS, start=1):
        daily_rows = []
        for f in files:
            day = os.path.splitext(os.path.basename(f))[0]
            df  = pd.read_parquet(f)
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

        daily    = pd.DataFrame(daily_rows).sort_values("Date").reset_index(drop=True)
        cum_cols = [gc for gc in g_cols if gc in daily.columns] + ["n_ticks"]
        daily[cum_cols] = daily[cum_cols].cumsum()
        daily["long_short"] = daily[g_cols[-1]] - daily["g1"]

        total_ticks = daily["n_ticks"].iloc[-1]
        total_ls    = daily["long_short"].iloc[-1]
        ls_label    = f"L/S({g_cols[-1]}-g1)"
        if pd.notna(total_ticks) and pd.notna(total_ls) and total_ticks > 0:
            avg_pnl_bps = total_ls / total_ticks * 1e4
            ls_label = f"L/S({g_cols[-1]}-g1)  avg={avg_pnl_bps:+.3f}bps/tick"

        x = np.arange(len(daily))
        fig, ax = plt.subplots(figsize=(14, 5))
        for i, gc in enumerate(g_cols):
            if gc in daily.columns:
                ax.plot(x, daily[gc],
                        color=_GROUP_COLORS[i % len(_GROUP_COLORS)],
                        alpha=0.7, linewidth=0.9, label=gc)
        if "long_short" in daily.columns:
            ax.plot(x, daily["long_short"], color="black", linewidth=1.4, label=ls_label)

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
        ax.set_title(f"Multi-Factor Composite  Slot {slot_idx}: {label}  Daily Cumulative Return")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_ax, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper left", ncol=min(len(g_cols) + 1, 11), fontsize=7)
        fig.tight_layout()
        safe_label = label.replace(":", "").replace("-", "_")
        fig.savefig(
            os.path.join(parquet_dir, f"_chart_slot_{slot_idx}_{safe_label}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


# ── 批量入口 ───────────────────────────────────────────────────────────────────

def run_post_compute(out_dirs: dict[str, str]) -> None:
    """多因子计算完成后，对各 ret_horizon 目录生成汇总与图表。"""
    for ret_h, sub_dir in out_dirs.items():
        _build_daily(sub_dir)
        _build_summary(sub_dir)
        _build_cum_daily(sub_dir)
        _build_cum_tick_chart(sub_dir)
        _build_intraday_slot_charts(sub_dir)
        print(f"[{ret_h}] 汇总完成：{sub_dir}")
