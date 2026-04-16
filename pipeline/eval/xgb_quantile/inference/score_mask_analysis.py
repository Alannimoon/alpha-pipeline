#!/usr/bin/env python
"""
score_std 掩码阈值分析脚本

逻辑：
  - score_std 是当前截面各股票模型分数的跨截面标准差，实盘可观测
  - ls_spread = g{n_groups} - g1，是事后才知道的真实多空收益，用于评估效果
  - 分析：score_std 高的截面，ls_spread 是否系统性更高？

只看：
  - 测试集（baseline + market_state）
  - 1430 时段（SampleTime >= 14:30:00）
  - 18 组配置（3 pool × 2 n_groups × 3 ret_horizon）

输出（result/eval/pnl/inference/）：
  - score_mask_bins.csv    : 分箱分析（10分位箱 × 36配置）
  - score_mask_thresh.csv  : 阈值分析（top 10/20/30/50% × 36配置）
  - score_mask/{setting}/{pool}_g{n}_{ret_h}.png : 柱状图

用法：
    python pipeline/eval/xgb_quantile/inference/score_mask_analysis.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

# ── 配置 ──────────────────────────────────────────────────────────────────────
FACTOR_POOLS  = ["union", "intersection", "all"]
N_GROUPS_LIST = [10, 20]
RET_HORIZONS  = ["ret100", "ret200", "ret300"]

SETTINGS = {
    "baseline":     os.path.join(config.EVAL_ROOT, "xgb_quantile_test"),
    "market_state": os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_test"),
}

SLOT_START = "14:30:00"
N_BINS     = 10
THRESHOLDS = [0.10, 0.20, 0.30, 0.50]   # 取 score_std 前 x% 的截面

OUT_DIR = os.path.join(config.EVAL_ROOT, "pnl", "inference")


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_score_std(pred_dir: str) -> pd.DataFrame:
    """
    读取所有 {date}_scores.parquet，只保留 SampleTime >= SLOT_START 的行，
    计算每个 (Date, SampleTime) 的跨股票分数标准差。
    """
    parts = []
    for p in sorted(Path(pred_dir).glob("????????_scores.parquet")):
        date = p.name[: -len("_scores.parquet")]
        if not date.isdigit() or len(date) != 8:
            continue
        try:
            df = pd.read_parquet(p, columns=["SampleTime", "score"])
        except Exception as e:
            print(f"  [WARN] 读取 {p.name} 失败: {e}")
            continue
        df = df[df["SampleTime"] >= SLOT_START]
        if df.empty:
            continue
        grp = (df.groupby("SampleTime")["score"]
                 .std(ddof=1)
                 .reset_index()
                 .rename(columns={"score": "score_std"}))
        grp["Date"] = date
        parts.append(grp[["Date", "SampleTime", "score_std"]])

    if not parts:
        return pd.DataFrame(columns=["Date", "SampleTime", "score_std"])
    return pd.concat(parts, ignore_index=True)


def load_ls_spread(pred_dir: str, n_groups: int) -> pd.DataFrame:
    """
    读取所有 {date}.parquet，计算每个 (Date, SampleTime)
    的多空价差 ls = g{n_groups} - g1（单位：return）。
    """
    g_last = f"g{n_groups}"
    parts  = []
    for p in sorted(Path(pred_dir).glob("????????.parquet")):
        try:
            df = pd.read_parquet(p, columns=["Date", "SampleTime", "g1", g_last])
        except Exception as e:
            print(f"  [WARN] 读取 {p.name} 失败: {e}")
            continue
        df = df[df["SampleTime"] >= SLOT_START]
        if df.empty:
            continue
        df["ls"] = df[g_last] - df["g1"]
        parts.append(df[["Date", "SampleTime", "ls"]])

    if not parts:
        return pd.DataFrame(columns=["Date", "SampleTime", "ls"])
    return pd.concat(parts, ignore_index=True)


def load_and_merge(pred_dir: str, n_groups: int) -> pd.DataFrame:
    """
    合并 score_std 和 ls_spread，筛选 1430 时段，去掉 NaN。
    返回 DataFrame：[Date, SampleTime, score_std, ls_bps]
    """
    score_df = load_score_std(pred_dir)
    ls_df    = load_ls_spread(pred_dir, n_groups)

    if score_df.empty or ls_df.empty:
        return pd.DataFrame()

    merged = score_df.merge(ls_df, on=["Date", "SampleTime"], how="inner")
    merged = merged[merged["SampleTime"] >= SLOT_START].copy()
    merged = merged.dropna(subset=["score_std", "ls"])
    merged["ls_bps"] = merged["ls"] * 1e4
    return merged[["Date", "SampleTime", "score_std", "ls_bps"]].reset_index(drop=True)


# ── 分析函数 ──────────────────────────────────────────────────────────────────

def bin_analysis(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """
    将 score_std 按分位数分成 n_bins 箱，
    返回每箱的 avg_pnl_bps、tick 数、score_std 均值。
    """
    df = df.copy()
    df["bin"] = pd.qcut(df["score_std"], q=n_bins, labels=False, duplicates="drop")
    result = (df.groupby("bin")
                .agg(
                    avg_pnl_bps=("ls_bps", "mean"),
                    n_ticks=("ls_bps", "count"),
                    score_std_mean=("score_std", "mean"),
                )
                .reset_index())
    return result


def threshold_analysis(df: pd.DataFrame, thresholds: list = THRESHOLDS) -> pd.DataFrame:
    """
    对每个阈值 t，只取 score_std >= (1-t) 分位数的截面（top t%），
    计算 avg_pnl_bps、tick 数、覆盖率。
    全量（t=1.0）作为基准行。
    """
    rows = []
    rows.append({
        "top_pct":     1.00,
        "threshold":   df["score_std"].min(),
        "avg_pnl_bps": df["ls_bps"].mean(),
        "n_ticks":     len(df),
        "coverage":    1.00,
    })
    for t in thresholds:
        cutoff = df["score_std"].quantile(1 - t)
        sub    = df[df["score_std"] >= cutoff]
        rows.append({
            "top_pct":     t,
            "threshold":   round(cutoff, 6),
            "avg_pnl_bps": sub["ls_bps"].mean(),
            "n_ticks":     len(sub),
            "coverage":    len(sub) / len(df),
        })
    return pd.DataFrame(rows)


# ── 图表 ──────────────────────────────────────────────────────────────────────

def plot_bins(bin_df: pd.DataFrame, thresh_df: pd.DataFrame,
              tag: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左：分箱 ─────────────────────────────────────────────────────────────
    ax = axes[0]
    bins   = bin_df["bin"].tolist()
    pnl    = bin_df["avg_pnl_bps"].tolist()
    counts = bin_df["n_ticks"].tolist()
    colors = ["#d73027" if v < 0 else "#4575b4" for v in pnl]
    ax.bar(bins, pnl, color=colors, alpha=0.85, width=0.7)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("score_std quantile bin (low -> high)")
    ax.set_ylabel("avg_pnl (bps)")
    ax.set_title(f"{tag}\nscore_std bins vs avg_pnl")
    ax.set_xticks(bins)

    ax2 = ax.twinx()
    ax2.plot(bins, counts, color="orange", marker="o",
             linewidth=1.2, markersize=4, label="n_ticks")
    ax2.set_ylabel("n_ticks", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    for b, v in zip(bins, pnl):
        ax.text(b, v + (0.15 if v >= 0 else -0.4),
                f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # ── 右：阈值过滤 ─────────────────────────────────────────────────────────
    ax = axes[1]
    labels = ["all"] + [f"top {int(t*100)}%" for t in THRESHOLDS]
    pnl_t  = thresh_df["avg_pnl_bps"].tolist()
    cov_t  = thresh_df["coverage"].tolist()
    colors2 = ["#91bfdb"] + ["#4575b4"] * len(THRESHOLDS)
    xpos   = range(len(labels))
    ax.bar(xpos, pnl_t, color=colors2, alpha=0.85, width=0.6)
    ax.axhline(pnl_t[0], color="gray", linewidth=1.0,
               linestyle="--", label=f"all {pnl_t[0]:.2f} bps")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("avg_pnl (bps)")
    ax.set_title(f"{tag}\nscore_std threshold filter avg_pnl")
    ax.legend(fontsize=8)

    ax3 = ax.twinx()
    ax3.plot(xpos, [c * 100 for c in cov_t], color="orange",
             marker="o", linewidth=1.2, markersize=4)
    ax3.set_ylabel("coverage (%)", color="orange")
    ax3.tick_params(axis="y", labelcolor="orange")
    ax3.set_ylim(0, 110)

    for x, v, c in zip(xpos, pnl_t, cov_t):
        ax.text(x, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    all_bin_rows    = []
    all_thresh_rows = []

    for setting, pred_root in SETTINGS.items():
        if not os.path.isdir(pred_root):
            print(f"[SKIP] 目录不存在：{pred_root}")
            continue

        print(f"\n{'='*60}")
        print(f"  设置：{setting}")
        print(f"{'='*60}")

        img_root = os.path.join(OUT_DIR, "score_mask", setting)

        for pool in FACTOR_POOLS:
            for n_groups in N_GROUPS_LIST:
                for ret_h in RET_HORIZONS:
                    pred_dir = os.path.join(pred_root, pool, f"g{n_groups}", ret_h)
                    tag = f"{setting} | {pool} | g{n_groups} | {ret_h}"

                    if not os.path.isdir(pred_dir):
                        print(f"  [SKIP] {tag}：目录不存在")
                        continue

                    df = load_and_merge(pred_dir, n_groups)
                    if df.empty:
                        print(f"  [SKIP] {tag}：无有效数据")
                        continue

                    print(f"  {tag}  n={len(df)}")

                    bin_df = bin_analysis(df)
                    for _, row in bin_df.iterrows():
                        all_bin_rows.append({
                            "setting":        setting,
                            "factor_pool":    pool,
                            "n_groups":       n_groups,
                            "ret_horizon":    ret_h,
                            "bin":            int(row["bin"]),
                            "score_std_mean": round(row["score_std_mean"], 6),
                            "avg_pnl_bps":    round(row["avg_pnl_bps"], 4),
                            "n_ticks":        int(row["n_ticks"]),
                        })

                    thresh_df = threshold_analysis(df)
                    for _, row in thresh_df.iterrows():
                        all_thresh_rows.append({
                            "setting":      setting,
                            "factor_pool":  pool,
                            "n_groups":     n_groups,
                            "ret_horizon":  ret_h,
                            "top_pct":      row["top_pct"],
                            "threshold":    row["threshold"],
                            "avg_pnl_bps":  round(row["avg_pnl_bps"], 4),
                            "n_ticks":      int(row["n_ticks"]),
                            "coverage":     round(row["coverage"], 4),
                        })

                    img_path = os.path.join(img_root, f"{pool}_g{n_groups}_{ret_h}.png")
                    plot_bins(bin_df, thresh_df, tag, img_path)

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    bin_out    = os.path.join(OUT_DIR, "score_mask_bins.csv")
    thresh_out = os.path.join(OUT_DIR, "score_mask_thresh.csv")
    pd.DataFrame(all_bin_rows).to_csv(bin_out, index=False)
    pd.DataFrame(all_thresh_rows).to_csv(thresh_out, index=False)
    print(f"\n已保存：{bin_out}")
    print(f"已保存：{thresh_out}")

    # ── 打印阈值汇总（只看 g20 / ret300，最强配置）────────────────────────────
    thresh_df_all = pd.DataFrame(all_thresh_rows)
    if thresh_df_all.empty:
        return

    print("\n=== 阈值过滤汇总（g20 / ret300）avg_pnl_bps ===")
    sub = thresh_df_all[
        (thresh_df_all["n_groups"] == 20) &
        (thresh_df_all["ret_horizon"] == "ret300")
    ].copy()
    sub["config"] = sub["setting"] + " | " + sub["factor_pool"]
    sub["top_pct_label"] = sub["top_pct"].apply(
        lambda x: "all" if x == 1.0 else f"top{int(x*100)}%"
    )
    pivot = sub.pivot_table(
        index="config",
        columns="top_pct_label",
        values="avg_pnl_bps",
        aggfunc="first",
    )
    col_order = ["all", "top50%", "top30%", "top20%", "top10%"]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(pivot.to_string())

    print(f"\n图表已保存至：{os.path.join(OUT_DIR, 'score_mask/')}")


if __name__ == "__main__":
    main()
