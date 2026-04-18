#!/usr/bin/env python
"""
score_std mask threshold analysis

Metrics:
  - score_std : cross-stock std of model scores at each (Date, SampleTime) tick
  - ls_spread : g{n_groups} - g1 (realised long-short return, post-hoc)

Scope: test set, slot >= 14:30, baseline + market_state, 18 configs each.

Outputs (result/eval/pnl/inference/):
  - score_mask_bins.csv      : bin analysis  (10 quantile bins × 36 configs)
  - score_mask_thresh.csv    : threshold analysis (top 10/20/30/50/100% × 36 configs)
  - score_mask_thresh.png    : 2×3 summary — x: top-pct, lines: 6 pool/g configs
  - score_mask_bins.png      : 2×3 summary — x: bin 0-9,  lines: 6 pool/g configs

Usage:
    python pipeline/eval/xgb_quantile/inference/score_mask.py
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

# ── constants ─────────────────────────────────────────────────────────────────
FACTOR_POOLS  = ["union", "intersection", "all"]
N_GROUPS_LIST = [10, 20]
RET_HORIZONS  = ["ret100", "ret200", "ret300"]

SETTINGS = {
    "baseline":     os.path.join(config.EVAL_ROOT, "xgb_quantile_test"),
    "market_state": os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_test"),
}

SLOT_START = "14:30:00"
N_BINS     = 10
THRESHOLDS = [0.10, 0.20, 0.30, 0.50]

OUT_DIR = os.path.join(config.EVAL_ROOT, "pnl", "inference")

CONFIG_LABELS = [f"{p}/g{n}" for p in FACTOR_POOLS for n in N_GROUPS_LIST]
COLORS        = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_score_std(pred_dir: str) -> pd.DataFrame:
    parts = []
    for p in sorted(Path(pred_dir).glob("????????_scores.parquet")):
        date = p.name[: -len("_scores.parquet")]
        if not date.isdigit() or len(date) != 8:
            continue
        try:
            df = pd.read_parquet(p, columns=["SampleTime", "score"])
        except Exception as e:
            print(f"  [WARN] failed to read {p.name}: {e}")
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
    g_last = f"g{n_groups}"
    parts  = []
    for p in sorted(Path(pred_dir).glob("????????.parquet")):
        try:
            df = pd.read_parquet(p, columns=["Date", "SampleTime", "g1", g_last])
        except Exception as e:
            print(f"  [WARN] failed to read {p.name}: {e}")
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
    score_df = load_score_std(pred_dir)
    ls_df    = load_ls_spread(pred_dir, n_groups)
    if score_df.empty or ls_df.empty:
        return pd.DataFrame()
    merged = score_df.merge(ls_df, on=["Date", "SampleTime"], how="inner")
    merged = merged[merged["SampleTime"] >= SLOT_START].copy()
    merged = merged.dropna(subset=["score_std", "ls"])
    merged["ls_bps"] = merged["ls"] * 1e4
    return merged[["Date", "SampleTime", "score_std", "ls_bps"]].reset_index(drop=True)


# ── analysis ──────────────────────────────────────────────────────────────────

def bin_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = pd.qcut(df["score_std"], q=N_BINS, labels=False, duplicates="drop")
    return (df.groupby("bin")
              .agg(avg_pnl_bps=("ls_bps", "mean"),
                   n_ticks=("ls_bps", "count"),
                   score_std_mean=("score_std", "mean"))
              .reset_index())


def threshold_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"top_pct": 1.00, "threshold": df["score_std"].min(),
             "avg_pnl_bps": df["ls_bps"].mean(), "n_ticks": len(df), "coverage": 1.00}]
    for t in THRESHOLDS:
        cutoff = df["score_std"].quantile(1 - t)
        sub    = df[df["score_std"] >= cutoff]
        rows.append({"top_pct": t, "threshold": round(cutoff, 6),
                     "avg_pnl_bps": sub["ls_bps"].mean(),
                     "n_ticks": len(sub), "coverage": len(sub) / len(df)})
    return pd.DataFrame(rows)


# ── charts ────────────────────────────────────────────────────────────────────

def plot_thresh_summary(thresh_df: pd.DataFrame, out_path: str) -> None:
    """
    2×3 grid (rows=setting, cols=ret_horizon).
    x-axis: all → top50% → top30% → top20% → top10% (tightening filter)
    lines : 6 pool/g{n} configs
    """
    x_labels = ["all", "top50%", "top30%", "top20%", "top10%"]
    x_pcts   = [1.0, 0.5, 0.3, 0.2, 0.1]
    xs       = np.arange(len(x_labels))

    settings     = list(SETTINGS.keys())
    ret_horizons = RET_HORIZONS

    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.3})

    for r, setting in enumerate(settings):
        for c, ret_h in enumerate(ret_horizons):
            ax  = axes[r][c]
            sub = thresh_df[
                (thresh_df["setting"] == setting) &
                (thresh_df["ret_horizon"] == ret_h)
            ]

            for i, (pool, ng) in enumerate(
                [(p, n) for p in FACTOR_POOLS for n in N_GROUPS_LIST]
            ):
                row = sub[(sub["factor_pool"] == pool) & (sub["n_groups"] == ng)]
                if row.empty:
                    continue
                ys = [float(row[row["top_pct"] == pct]["avg_pnl_bps"].iloc[0])
                      if len(row[row["top_pct"] == pct]) > 0 else np.nan
                      for pct in x_pcts]
                ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.6,
                        color=COLORS[i], label=CONFIG_LABELS[i])

            ax.set_xticks(xs)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_title(f"{setting} | {ret_h}", fontsize=9)
            if c == 0:
                ax.set_ylabel("avg_pnl (bps)", fontsize=8)
            ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            if r == 0 and c == 2:
                ax.legend(fontsize=6.5, loc="upper left",
                          bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.suptitle("score_std threshold filter: avg_pnl by top-pct (slot 1430, test set)",
                 fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_bins_summary(bins_df: pd.DataFrame, out_path: str) -> None:
    """
    2×3 grid (rows=setting, cols=ret_horizon).
    x-axis: bin 0-9 (low → high score_std)
    lines : 6 pool/g{n} configs
    """
    xs = np.arange(N_BINS)

    settings     = list(SETTINGS.keys())
    ret_horizons = RET_HORIZONS

    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.3})

    for r, setting in enumerate(settings):
        for c, ret_h in enumerate(ret_horizons):
            ax  = axes[r][c]
            sub = bins_df[
                (bins_df["setting"] == setting) &
                (bins_df["ret_horizon"] == ret_h)
            ]

            for i, (pool, ng) in enumerate(
                [(p, n) for p in FACTOR_POOLS for n in N_GROUPS_LIST]
            ):
                row = sub[(sub["factor_pool"] == pool) & (sub["n_groups"] == ng)]
                if row.empty:
                    continue
                row = row.sort_values("bin")
                ys  = row["avg_pnl_bps"].values
                ax.plot(row["bin"].values, ys, marker="o", markersize=4,
                        linewidth=1.5, color=COLORS[i], label=CONFIG_LABELS[i])

            ax.set_xticks(xs)
            ax.set_xticklabels([str(b) for b in xs], fontsize=8)
            ax.set_xlabel("score_std quantile bin (low → high)", fontsize=7)
            ax.set_title(f"{setting} | {ret_h}", fontsize=9)
            if c == 0:
                ax.set_ylabel("avg_pnl (bps)", fontsize=8)
            ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            if r == 0 and c == 2:
                ax.legend(fontsize=6.5, loc="upper left",
                          bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.suptitle("score_std bin analysis: avg_pnl by quantile bin (slot 1430, test set)",
                 fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_bin_rows    = []
    all_thresh_rows = []

    for setting, pred_root in SETTINGS.items():
        if not os.path.isdir(pred_root):
            print(f"[SKIP] directory not found: {pred_root}")
            continue

        print(f"\n{'='*60}\n  setting: {setting}\n{'='*60}")

        for pool in FACTOR_POOLS:
            for n_groups in N_GROUPS_LIST:
                for ret_h in RET_HORIZONS:
                    pred_dir = os.path.join(pred_root, pool, f"g{n_groups}", ret_h)
                    tag      = f"{setting} | {pool} | g{n_groups} | {ret_h}"

                    if not os.path.isdir(pred_dir):
                        print(f"  [SKIP] {tag}: directory not found")
                        continue

                    df = load_and_merge(pred_dir, n_groups)
                    if df.empty:
                        print(f"  [SKIP] {tag}: no data")
                        continue

                    print(f"  {tag}  n={len(df)}")

                    for _, row in bin_analysis(df).iterrows():
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

                    for _, row in threshold_analysis(df).iterrows():
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

    # ── save CSVs ─────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    bins_df   = pd.DataFrame(all_bin_rows)
    thresh_df = pd.DataFrame(all_thresh_rows)
    bins_df.to_csv(os.path.join(OUT_DIR, "score_mask_bins.csv"),   index=False)
    thresh_df.to_csv(os.path.join(OUT_DIR, "score_mask_thresh.csv"), index=False)
    print(f"\nCSVs saved to {OUT_DIR}/")

    if bins_df.empty or thresh_df.empty:
        return

    # ── summary charts ────────────────────────────────────────────────────────
    plot_thresh_summary(thresh_df, os.path.join(OUT_DIR, "score_mask_thresh.png"))
    plot_bins_summary(bins_df,     os.path.join(OUT_DIR, "score_mask_bins.png"))

    # ── print summary (g20 / ret300) ──────────────────────────────────────────
    print("\n=== threshold filter summary (g20 / ret300) avg_pnl_bps ===")
    sub = thresh_df[(thresh_df["n_groups"] == 20) & (thresh_df["ret_horizon"] == "ret300")].copy()
    sub["config"]      = sub["setting"] + " | " + sub["factor_pool"]
    sub["top_pct_lbl"] = sub["top_pct"].apply(lambda x: "all" if x == 1.0 else f"top{int(x*100)}%")
    pivot = sub.pivot_table(index="config", columns="top_pct_lbl",
                            values="avg_pnl_bps", aggfunc="first")
    col_order = [c for c in ["all", "top50%", "top30%", "top20%", "top10%"] if c in pivot.columns]
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(pivot[col_order].to_string())


if __name__ == "__main__":
    main()
