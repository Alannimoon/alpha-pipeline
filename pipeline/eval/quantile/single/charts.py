"""
单因子分层汇总与画图模块。

函数列表
--------
_build_daily          → _daily.csv（每日各组均值）
_build_summary        → _summary.csv（跨日整体均值）
_build_cum_daily      → _cum_daily.csv（跨日累计末值，直接从 parquet 计算）
_build_cum_tick_chart → _chart_tick_{fc}.png（跨日 tick 连续曲线）

注：不再生成 _cum_tick_{date}.csv 中间文件；
    画图所需的日内 cumsum 直接从 parquet 实时计算，
    跨日偏移从 _cum_daily.csv 读取。
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

N_GROUPS = 5
_GROUP_COLORS = ["#d62728", "#ff7f0e", "#8c8c8c", "#2ca02c", "#1f77b4"]


def _parquet_files(parquet_dir: str) -> list[str]:
    return sorted(
        f for f in glob.glob(os.path.join(parquet_dir, "*.parquet"))
        if not os.path.basename(f).startswith("_")
    )


# ── 日度均值汇总 ───────────────────────────────────────────────────────────────

def _build_daily(parquet_dir: str) -> None:
    """每天对所有 tick 取均值 → _daily.csv（Date, factor_col, g1..g5）。"""
    files = _parquet_files(parquet_dir)
    if not files:
        return

    rows = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df  = pd.read_parquet(f)
        g_cols = [c for c in df.columns if re.match(r"g\d+_", c)]
        if not g_cols:
            continue
        day_means = df[g_cols].mean()

        for fc in sorted({re.match(r"g\d+_(.*)", c).group(1) for c in g_cols}):
            row = {"Date": day, "factor_col": fc}
            for g in range(1, 6):
                row[f"g{g}"] = day_means.get(f"g{g}_{fc}", np.nan)
            rows.append(row)

    if rows:
        (pd.DataFrame(rows)
         .sort_values(["factor_col", "Date"])
         .reset_index(drop=True)
         .to_csv(os.path.join(parquet_dir, "_daily.csv"), index=False))


def _build_summary(parquet_dir: str) -> None:
    """跨所有日期和 tick 的总体均值 → _summary.csv（factor_col, g1..g5）。"""
    files = _parquet_files(parquet_dir)
    if not files:
        return

    daily_means = []
    for f in files:
        df = pd.read_parquet(f)
        g_cols = [c for c in df.columns if re.match(r"g\d+_", c)]
        if g_cols:
            daily_means.append(df[g_cols].mean())

    if not daily_means:
        return

    overall = pd.DataFrame(daily_means).mean()
    fc_data: dict[str, dict] = {}
    for col, val in overall.items():
        m = re.match(r"g(\d+)_(.*)", col)
        if m:
            fc_data.setdefault(m.group(2), {})[f"g{m.group(1)}"] = val

    rows = [{"factor_col": fc, **vals} for fc, vals in fc_data.items()]
    (pd.DataFrame(rows)[["factor_col", "g1", "g2", "g3", "g4", "g5"]]
     .to_csv(os.path.join(parquet_dir, "_summary.csv"), index=False))


# ── 跨日累计末值（直接从 parquet sum）────────────────────────────────────────

def _build_cum_daily(parquet_dir: str) -> None:
    """
    直接对每日 parquet 求列和（= 日内 cumsum 的最后一行）→ 跨日 cumsum →
    _cum_daily.csv（factor_col, Date, g1..g5, long_short, n_ticks）。
    """
    files = _parquet_files(parquet_dir)
    if not files:
        return

    rows = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df  = pd.read_parquet(f)
        g_cols = [c for c in df.columns if re.match(r"g\d+_", c)]
        if not g_cols:
            continue

        fc_set = sorted({re.match(r"g\d+_(.*)", c).group(1) for c in g_cols})
        for fc in fc_set:
            present = {g: f"g{g}_{fc}" for g in range(1, 6) if f"g{g}_{fc}" in df.columns}
            if not present:
                continue
            row = {"factor_col": fc, "Date": day}
            for g, col in present.items():
                row[f"g{g}"] = float(df[col].sum(skipna=True))
            row["n_ticks"] = int(df[next(iter(present.values()))].notna().sum())
            rows.append(row)

    if not rows:
        return

    all_days = (
        pd.DataFrame(rows)
        .sort_values(["factor_col", "Date"])
        .reset_index(drop=True)
    )
    g_cols   = [f"g{g}" for g in range(1, 6)]
    cum_cols = g_cols + ["n_ticks"]
    all_days[cum_cols] = all_days.groupby("factor_col")[cum_cols].cumsum()
    all_days["long_short"] = all_days["g5"] - all_days["g1"]

    keep = ["factor_col", "Date"] + g_cols + ["long_short", "n_ticks"]
    all_days[keep].to_csv(os.path.join(parquet_dir, "_cum_daily.csv"), index=False)


# ── 跨日 tick 连续曲线图（直接从 parquet 实时 cumsum）────────────────────────

def _build_cum_tick_chart(parquet_dir: str) -> None:
    """
    直接读各日 parquet，日内 cumsum 后以 _cum_daily.csv 中的前一日末值做偏移，
    拼出跨日连续 tick 曲线并画图 → _chart_tick_{factor_col}.png。
    不再依赖 _cum_tick_{date}.csv 中间文件。
    """
    files = _parquet_files(parquet_dir)
    if not files:
        return

    daily_path = os.path.join(parquet_dir, "_cum_daily.csv")
    if not os.path.exists(daily_path):
        return
    daily_df = pd.read_csv(daily_path, dtype={"Date": str})

    fc_list = daily_df["factor_col"].unique().tolist()
    g_cols  = [f"g{g}" for g in range(1, 6)]

    chart_iter = tqdm(fc_list, desc="cum_tick_chart") if tqdm else fc_list
    for fc in chart_iter:
        daily_fc = (
            daily_df[daily_df["factor_col"] == fc]
            .set_index("Date")
            .drop(columns="factor_col")
        )
        all_cols = [c for c in g_cols + ["long_short"] if c in daily_fc.columns]

        dfs      = []
        prev_date = None

        for f in files:
            day = os.path.splitext(os.path.basename(f))[0]
            df  = pd.read_parquet(f)

            present = {g: f"g{g}_{fc}" for g in range(1, 6) if f"g{g}_{fc}" in df.columns}
            if not present:
                prev_date = day
                continue

            sub = df[["SampleTime"] + list(present.values())].copy()
            sub = sub.rename(columns={v: f"g{g}" for g, v in present.items()})
            g_sub = [f"g{g}" for g in range(1, 6) if f"g{g}" in sub.columns]

            sub[g_sub] = sub[g_sub].cumsum()
            sub["long_short"] = sub["g5"] - sub["g1"]

            if prev_date is not None and prev_date in daily_fc.index:
                offset = daily_fc.loc[prev_date, all_cols]
                sub[all_cols] = sub[all_cols] + offset.values

            sub.insert(0, "Date", day)
            dfs.append(sub)
            prev_date = day

        if not dfs:
            continue

        tick_df   = pd.concat(dfs, ignore_index=True)
        x         = np.arange(len(tick_df))
        day_starts = tick_df.groupby("Date", sort=False).apply(
            lambda g: g.index[0] - tick_df.index[0], include_groups=False
        )

        ls_label = "L/S(g5-g1)"
        if "n_ticks" in daily_fc.columns and "long_short" in daily_fc.columns:
            total_ticks = daily_fc["n_ticks"].iloc[-1]
            total_ls    = daily_fc["long_short"].iloc[-1]
            if pd.notna(total_ticks) and pd.notna(total_ls) and total_ticks > 0:
                avg_pnl_bps = total_ls / total_ticks * 1e4
                ls_label = f"L/S(g5-g1)  avg={avg_pnl_bps:+.3f}bps/tick"

        fig, ax = plt.subplots(figsize=(14, 5))
        for g in range(1, 6):
            col = f"g{g}"
            if col in tick_df.columns:
                ax.plot(x, tick_df[col], color=_GROUP_COLORS[g - 1],
                        alpha=0.7, linewidth=0.8, label=f"g{g}")
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
        ax.set_title(f"{fc}  Cross-Day Tick Cumulative Return")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper left", ncol=6, fontsize=8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(parquet_dir, f"_chart_tick_{fc}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def run_post_compute(base_dir: str, ret_horizons: dict) -> None:
    """计算完成后对所有子目录生成汇总与图表。"""
    for h_key in ret_horizons:
        sub_dir = os.path.join(base_dir, h_key)
        if os.path.isdir(sub_dir):
            _build_daily(sub_dir)
            _build_summary(sub_dir)
            _build_cum_daily(sub_dir)
            _build_cum_tick_chart(sub_dir)


def run_cs_quantile_chart(eval_root: str, factor_name: str) -> None:
    """重新生成跨日 tick 图，不重跑分层计算。"""
    base_dir = os.path.join(eval_root, "cs_quantile", factor_name)
    for h_key in ("ret100", "ret200", "ret300"):
        sub_dir = os.path.join(base_dir, h_key)
        if os.path.isdir(sub_dir):
            _build_cum_daily(sub_dir)
            _build_cum_tick_chart(sub_dir)
    print(f"图表重新生成完成：{base_dir}")
