"""
analyze_filter_threshold.py
============================
截面过滤阈值分析 — 基准模型 A500+23 G20 ret300，测试集 14:30-14:42。

两种过滤方法
  std  : 截面内 100 只股票 score 的标准差（越大说明模型分歧越大）
  dist : top5 均值 - bottom5 均值（g20 组均值 - g1 组均值）

Part 1  阈值分布分析
  - 10 分桶柱状图（横轴=分位桶，纵轴=多空收益）
  - Top X% 曲线（X 轴=保留比例，Y 轴=多空收益 + 截面数量）

Part 2  过滤后全量统计（两种方法 × 若干阈值各出一套）
  - 出手截面数量序列（每日）
  - 交易网格图（43天 × 240tick，蓝=出手）
  - 股票被选中次数热图（100×43）
  - g20-g1 和 (g19+g20)/2-(g1+g2)/2 收益
  - 日胜率

输出目录：result/eval/analysis/filter_threshold/
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

# ── 参数 ────────────────────────────────────────────────────────────────────────

_ap = argparse.ArgumentParser()
_ap.add_argument("--pool", default="test", choices=["test", "test2"],
                 help="测试集：test（原 vol100）/ test2（vol_top100_v2）")
_args = _ap.parse_args()

# ── 配置 ────────────────────────────────────────────────────────────────────────

SCORES_ROOT = os.path.join(
    config.EVAL_ROOT,
    f"xgb_quantile_market_state_vol_turnover_{_args.pool}",
    "all", "g20", "ret300",
)
OUT_DIR = os.path.join(config.EVAL_ROOT, "analysis", f"filter_threshold_{_args.pool}")
os.makedirs(OUT_DIR, exist_ok=True)

TIME_START = "14:30:00"
TIME_END   = "14:42:00"   # 不含，最后一个 tick 是 14:41:57
N_GROUPS   = 20
N_PER_GROUP = 5           # 100 只 / 20 组

TOP_X_LIST = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]

# Part 2 跑的 (method, top_pct) 组合
FILTER_CONFIGS = [
    ("std",  10), ("std",  20), ("std",  30),
    ("dist", 10), ("dist", 20), ("dist", 30),
]


# ── 数据加载 ─────────────────────────────────────────────────────────────────────

def load_data():
    dates = sorted(
        f[:-len("_scores.parquet")]
        for f in os.listdir(SCORES_ROOT)
        if f.endswith("_scores.parquet") and not f.startswith("_")
    )
    print(f"Test dates: {len(dates)}  ({dates[0]} ~ {dates[-1]})")

    score_parts, ret_parts = [], []
    for day in dates:
        sp = os.path.join(SCORES_ROOT, f"{day}_scores.parquet")
        rp = os.path.join(SCORES_ROOT, f"{day}.parquet")
        if not os.path.exists(sp) or not os.path.exists(rp):
            print(f"  [SKIP] {day}: missing file")
            continue

        sc = pd.read_parquet(sp)
        sc = sc[(sc["SampleTime"] >= TIME_START) & (sc["SampleTime"] < TIME_END)]
        if not sc.empty:
            score_parts.append(sc)

        rdf = pd.read_parquet(rp)
        rdf = rdf[(rdf["SampleTime"] >= TIME_START) & (rdf["SampleTime"] < TIME_END)]
        if not rdf.empty:
            ret_parts.append(rdf)

    scores_df = pd.concat(score_parts, ignore_index=True)
    ret_df    = pd.concat(ret_parts,   ignore_index=True)

    n_cs = scores_df.groupby(["Date", "SampleTime"]).ngroups
    print(f"Total cross-sections: {n_cs}  (~{n_cs/len(dates):.0f}/day)")
    return scores_df, ret_df, dates


# ── 截面级指标 ───────────────────────────────────────────────────────────────────

def compute_metrics(scores_df: pd.DataFrame) -> pd.DataFrame:
    """每个 (Date, SampleTime) 截面 → std 和 dist 两个指标。"""
    def _agg(grp):
        sc = np.sort(grp["score"].values)
        return pd.Series({
            "std":  float(np.std(sc)),
            "dist": float(sc[-N_PER_GROUP:].mean() - sc[:N_PER_GROUP].mean()),
        })

    metrics = (scores_df
               .groupby(["Date", "SampleTime"], sort=False)
               .apply(_agg)
               .reset_index())
    return metrics


def merge_returns(metrics: pd.DataFrame, ret_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["Date", "SampleTime", "g1", "g2", "g19", "g20"]
    present = [c for c in keep if c in ret_df.columns]
    ret_sub = ret_df[present].copy()
    ret_sub["ls_1_20"] = ret_sub["g20"] - ret_sub["g1"]

    merged = metrics.merge(ret_sub, on=["Date", "SampleTime"], how="left")
    return merged


# ── Part 1: 分桶分析 ─────────────────────────────────────────────────────────────

def plot_decile(merged: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Filter Metric vs Long-Short Return (10-Decile, 14:30-14:42, A500+23 G20 Test)"
    )

    csv_rows = []
    for ax, col, label in [
        (axes[0], "std",  "Score Std"),
        (axes[1], "dist", "Group Distance (top5 - bot5 mean score)"),
    ]:
        df = merged.dropna(subset=[col, "ls_1_20"]).copy()
        df["decile"] = pd.qcut(df[col], 10, labels=False, duplicates="drop")
        gb_ls = df.groupby("decile")["ls_1_20"].mean() * 1e4
        gb_n  = df.groupby("decile").size()
        for d in gb_ls.index:
            csv_rows.append({
                "method": col, "decile": int(d) + 1,
                "avg_ls_bps": round(float(gb_ls[d]), 4),
                "n_cross_sections": int(gb_n[d]),
            })
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in gb_ls.values]
        ax.bar(range(len(gb_ls)), gb_ls.values, color=colors, width=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"{label} decile (0=lowest, 9=highest)")
        ax.set_ylabel("Avg LS Return (bps/tick)")
        ax.set_title(label)
        ax.set_xticks(range(len(gb_ls)))
        ax.set_xticklabels([f"D{i+1}" for i in range(len(gb_ls))])

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "part1_decile.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")
    return pd.DataFrame(csv_rows)


def plot_topk_curve(merged: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Top X% Filter: LS Performance vs Selection Rate (14:30-14:42, A500+23 G20 Test)"
    )

    csv_rows = []
    for ax, col, label in [
        (axes[0], "std",  "Score Std"),
        (axes[1], "dist", "Group Distance"),
    ]:
        df = merged.dropna(subset=[col, "ls_1_20"]).copy()
        xs, ys_ls, ys_n = [], [], []
        for x in TOP_X_LIST:
            thresh = df[col].quantile(1 - x / 100)
            sub = df[df[col] >= thresh]
            xs.append(x)
            ys_ls.append(sub["ls_1_20"].mean() * 1e4)
            ys_n.append(len(sub))
            csv_rows.append({
                "method": col, "top_pct": x,
                "n_selected": len(sub),
                "avg_ls_bps": round(sub["ls_1_20"].mean() * 1e4, 4),
            })

        ax2 = ax.twinx()
        ax2.bar(xs, ys_n, alpha=0.15, color="gray", width=3)
        ax2.set_ylabel("Num cross-sections selected", color="gray")

        ax.plot(xs, ys_ls, "o-", color="#1f77b4", linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"Top X% by {label}")
        ax.set_ylabel("Avg LS Return (bps/tick)")
        ax.set_title(label)
        ax.set_xticks(xs)
        ax.legend(["Avg LS return"], loc="upper right")

        print(f"\n[{label}] Top X% results:")
        print(f"  {'TopX%':>6}  {'N':>7}  {'AvgLS(bps)':>12}")
        for x, n, ls in zip(xs, ys_n, ys_ls):
            print(f"  {x:>5}%  {n:>7}  {ls:>12.4f}")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "part1_topk_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")
    return pd.DataFrame(csv_rows)


# ── Part 2: 过滤后统计 ───────────────────────────────────────────────────────────

def run_filtered_stats(
    merged: pd.DataFrame,
    scores_df: pd.DataFrame,
    dates: list[str],
    method: str,
    top_pct: int,
) -> dict:
    tag = f"{method}_top{top_pct}"
    df  = merged.dropna(subset=[method, "ls_1_20"]).copy()
    thresh = df[method].quantile(1 - top_pct / 100)
    sel   = df[df[method] >= thresh].copy()
    n_total, n_sel = len(df), len(sel)

    print(f"\n{'='*60}")
    print(f"[{tag}]  threshold={thresh:.4f}  selected={n_sel}/{n_total} ({n_sel/n_total*100:.1f}%)")

    # ── 1. 每日出手数量 ──────────────────────────────────────────────────────────
    daily_counts = sel.groupby("Date").size().reindex(dates, fill_value=0)
    print(f"\nDaily cross-sections selected (mean={daily_counts.mean():.1f}, "
          f"min={daily_counts.min()}, max={daily_counts.max()}):")
    print(daily_counts.to_string())

    # ── 2. 交易网格图 ────────────────────────────────────────────────────────────
    all_times = sorted(df["SampleTime"].unique())
    grid = pd.DataFrame(
        False, index=pd.CategoricalIndex(dates), columns=all_times
    )
    for d, t in zip(sel["Date"], sel["SampleTime"]):
        if t in grid.columns:
            grid.loc[d, t] = True

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.imshow(grid.values.astype(float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates, fontsize=7)
    step = max(1, len(all_times) // 24)
    ax.set_xticks(range(0, len(all_times), step))
    ax.set_xticklabels(all_times[::step], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("SampleTime (3s intervals, 14:30-14:42)")
    ax.set_ylabel("Date")
    ax.set_title(
        f"Trade Grid: {tag} | {n_sel} / {n_total} cross-sections | "
        f"avg {daily_counts.mean():.1f}/day"
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"part2_{tag}_grid.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ── 3. 股票被选中次数热图 ─────────────────────────────────────────────────────
    sel_scores = scores_df.merge(
        sel[["Date", "SampleTime"]].drop_duplicates(),
        on=["Date", "SampleTime"],
    )

    # 每个截面内：top N_PER_GROUP = g20（多），bottom N_PER_GROUP = g1（空）
    sel_scores = sel_scores.sort_values(["Date", "SampleTime", "score"])
    grp_size = sel_scores.groupby(["Date", "SampleTime"])["score"].transform("count")
    rank_asc  = sel_scores.groupby(["Date", "SampleTime"]).cumcount()
    sel_scores["in_long"]  = (grp_size - 1 - rank_asc) < N_PER_GROUP   # top N
    sel_scores["in_short"] = rank_asc < N_PER_GROUP                     # bot N
    sel_scores["traded"]   = sel_scores["in_long"] | sel_scores["in_short"]

    all_secids = sorted(sel_scores["SecurityID"].unique())
    heatmap    = pd.DataFrame(0, index=all_secids, columns=dates)

    traded_df = sel_scores[sel_scores["traded"]]
    counts = traded_df.groupby(["SecurityID", "Date"]).size()
    for (sid, d), cnt in counts.items():
        if sid in heatmap.index and d in heatmap.columns:
            heatmap.loc[sid, d] = int(cnt)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(heatmap.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Times selected (long + short)")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=90, fontsize=7)
    ax.set_yticks(range(len(all_secids)))
    ax.set_yticklabels(all_secids, fontsize=7)
    ax.set_title(f"Stock Selection Heatmap: {tag}")
    ax.set_xlabel("Date")
    ax.set_ylabel("SecurityID")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"part2_{tag}_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ── 4. 多空收益 ──────────────────────────────────────────────────────────────
    # ls_1_20        : 多g20 空g1（单组多空差）
    # ls_top2_bot2   : (long g19+g20  +  short g1+g2) / 2，等权多空两组策略收益
    ls_1_20          = sel["ls_1_20"].mean() * 1e4
    ls_top2_bot2_bps = ((sel["g19"] + sel["g20"]) - (sel["g1"] + sel["g2"])).mean() * 1e4 / 2

    print(f"\nLS Performance in filtered cross-sections:")
    print(f"  g20 - g1           : {ls_1_20:.4f} bps/tick")
    print(f"  ls top2/bot2       : {ls_top2_bot2_bps:.4f} bps/tick")

    # ── 5. 日胜率 ─────────────────────────────────────────────────────────────────
    daily_ls      = sel.groupby("Date")["ls_1_20"].mean() * 1e4
    daily_ls      = daily_ls.reindex(dates)       # 无出手天 → NaN
    win_days      = int((daily_ls > 0).sum())
    no_trade_days = int(daily_ls.isna().sum())
    loss_days     = len(dates) - win_days - no_trade_days
    win_rate      = win_days / len(dates)
    print(f"\nDaily Win Rate: {win_days} win / {loss_days} loss / {no_trade_days} no-trade"
          f"  (total {len(dates)} days) = {win_rate:.1%}")
    print(daily_ls.round(4).to_string())

    summary_row = {
        "method":          method,
        "top_pct":         top_pct,
        "threshold":       round(thresh, 6),
        "ls_1_20_bps":       round(ls_1_20,          4),
        "ls_top2_bot2_bps":  round(ls_top2_bot2_bps, 4),
        "win_days":        win_days,
        "loss_days":       loss_days,
        "no_trade_days":   no_trade_days,
        "win_rate":        round(win_rate, 4),
    }

    # daily_counts DataFrame（供外部汇总）
    dc_df = daily_counts.reset_index()
    dc_df.columns = ["date", "n_selected"]
    dc_df.insert(0, "top_pct", top_pct)
    dc_df.insert(0, "method",  method)

    # heatmap DataFrame（供外部汇总）
    hm_df = heatmap.reset_index().melt(id_vars="index", var_name="date", value_name="count")
    hm_df = hm_df.rename(columns={"index": "secid"})
    hm_df.insert(0, "top_pct", top_pct)
    hm_df.insert(0, "method",  method)

    return summary_row, dc_df, hm_df


# ── 主流程 ────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    scores_df, ret_df, dates = load_data()

    print("Computing cross-section metrics ...")
    metrics = compute_metrics(scores_df)
    print(f"std  range: [{metrics['std'].min():.4f},  {metrics['std'].max():.4f}]")
    print(f"dist range: [{metrics['dist'].min():.4f}, {metrics['dist'].max():.4f}]")

    merged = merge_returns(metrics, ret_df)

    # ── Part 1 ────────────────────────────────────────────────────────────────────
    print("\n=== Part 1: Distribution Analysis ===")
    decile_df = plot_decile(merged)
    topk_df   = plot_topk_curve(merged)

    decile_df.to_csv(os.path.join(OUT_DIR, "part1_decile.csv"),    index=False)
    topk_df.to_csv(  os.path.join(OUT_DIR, "part1_topk.csv"),      index=False)
    print(f"Saved: part1_decile.csv, part1_topk.csv")

    # ── Part 2 ────────────────────────────────────────────────────────────────────
    print("\n=== Part 2: Filtered Statistics ===")
    summary_rows, dc_parts, hm_parts = [], [], []
    for method, top_pct in FILTER_CONFIGS:
        row, dc_df, hm_df = run_filtered_stats(merged, scores_df, dates, method, top_pct)
        summary_rows.append(row)
        dc_parts.append(dc_df)
        hm_parts.append(hm_df)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    pd.concat(dc_parts, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "part2_daily_counts.csv"), index=False)

    pd.concat(hm_parts, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "part2_heatmap.csv"), index=False)

    print(f"\n{'='*60}")
    print("CSV saved: summary.csv, part1_decile.csv, part1_topk.csv, "
          "part2_daily_counts.csv, part2_heatmap.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
