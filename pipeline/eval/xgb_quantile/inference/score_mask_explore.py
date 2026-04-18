#!/usr/bin/env python
"""
Single-config deep dive: market_state / union / g20 / ret300, test set, slot >= 14:30

Four-panel figure (2×2):
  top-left     : score_std histogram (10320 ticks) + 90th-pct threshold line
  top-right    : ls_bps KDE — all ticks vs top-10% score_std subset
  bottom-left  : scatter score_std vs ls_bps (each dot = one tick) + binned mean
  bottom-right : sorted score curve for two example cross-sections
                 (highest score_std vs median score_std)

Output: result/eval/pnl/inference/score_mask_explore.png

Usage:
    python pipeline/eval/xgb_quantile/inference/score_mask_explore.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────
SETTING    = "market_state"
POOL       = "union"
N_GROUPS   = 20
RET_H      = "ret300"
SLOT_START = "14:30:00"
N_BINS     = 10
TOP_PCT    = 0.10

PRED_DIR = os.path.join(
    config.EVAL_ROOT, "xgb_quantile_market_state_test",
    POOL, f"g{N_GROUPS}", RET_H,
)
OUT_PATH = os.path.join(config.EVAL_ROOT, "pnl", "inference", "score_mask_explore.png")


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(pred_dir: str) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      ticks_df      : one row per (Date, SampleTime), columns: score_std, ls_bps
      cross_sections: scores for the highest and median score_std ticks
    """
    g_last      = f"g{N_GROUPS}"
    score_parts = []
    ls_parts    = []

    for p in sorted(Path(pred_dir).glob("????????.parquet")):
        date = p.stem
        try:
            df = pd.read_parquet(p, columns=["Date", "SampleTime", "g1", g_last])
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}")
            continue
        df = df[df["SampleTime"] >= SLOT_START].copy()
        if df.empty:
            continue
        df["ls_bps"] = (df[g_last] - df["g1"]) * 1e4
        ls_parts.append(df[["Date", "SampleTime", "ls_bps"]])

        sp = Path(pred_dir) / f"{date}_scores.parquet"
        if not sp.exists():
            continue
        try:
            sdf = pd.read_parquet(sp, columns=["SampleTime", "score"])
        except Exception as e:
            print(f"  [WARN] {sp.name}: {e}")
            continue
        sdf = sdf[sdf["SampleTime"] >= SLOT_START].copy()
        if sdf.empty:
            continue
        grp = (sdf.groupby("SampleTime")["score"]
                  .std(ddof=1)
                  .reset_index()
                  .rename(columns={"score": "score_std"}))
        grp["Date"] = date
        score_parts.append(grp[["Date", "SampleTime", "score_std"]])

    if not score_parts or not ls_parts:
        raise RuntimeError("no data found — check PRED_DIR")

    ticks_df = (pd.concat(score_parts, ignore_index=True)
                  .merge(pd.concat(ls_parts, ignore_index=True),
                         on=["Date", "SampleTime"], how="inner")
                  .dropna(subset=["score_std", "ls_bps"])
                  .sort_values("score_std")
                  .reset_index(drop=True))

    # load raw scores for the two representative ticks
    idx_top    = len(ticks_df) - 1
    idx_median = len(ticks_df) // 2
    cross_sections = {}
    for idx, label in [(idx_top, "highest score_std"), (idx_median, "median score_std")]:
        row  = ticks_df.iloc[idx]
        sp   = Path(pred_dir) / f"{row['Date']}_scores.parquet"
        sdf  = pd.read_parquet(sp, columns=["SampleTime", "score"])
        tick_scores = sdf[sdf["SampleTime"] == row["SampleTime"]]["score"].values
        cross_sections[label] = {
            "scores":    tick_scores,
            "score_std": round(row["score_std"], 4),
            "date":      row["Date"],
            "time":      row["SampleTime"],
        }

    return ticks_df, cross_sections


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(ticks_df: pd.DataFrame, cross_sections: dict, out_path: str) -> None:
    score_std = ticks_df["score_std"].values
    ls_bps    = ticks_df["ls_bps"].values
    threshold = np.quantile(score_std, 1 - TOP_PCT)
    mask_top  = score_std >= threshold

    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.35})

    # ── top-left: score_std histogram ────────────────────────────────────────
    ax = axes[0][0]
    ax.hist(score_std, bins=60, color="#4575b4", alpha=0.75, edgecolor="none")
    ax.axvline(threshold, color="#d73027", linewidth=1.8, linestyle="--",
               label=f"top {int(TOP_PCT*100)}% threshold = {threshold:.3f}")
    ax.set_xlabel("score_std")
    ax.set_ylabel("count (ticks)")
    ax.set_title(f"score_std distribution  (n={len(score_std):,})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # ── top-right: ls_bps KDE — all vs top-10% ───────────────────────────────
    ax = axes[0][1]
    for vals, color, label in [
        (ls_bps,           "#aaaaaa", f"all  (n={len(ls_bps):,},  mean={ls_bps.mean():.1f} bps)"),
        (ls_bps[mask_top], "#d73027", f"top {int(TOP_PCT*100)}%  (n={mask_top.sum():,},  mean={ls_bps[mask_top].mean():.1f} bps)"),
    ]:
        kde  = gaussian_kde(vals, bw_method=0.15)
        xs   = np.linspace(vals.min() - 5, vals.max() + 5, 400)
        ax.plot(xs, kde(xs), color=color, linewidth=1.8, label=label)
        ax.axvline(vals.mean(), color=color, linewidth=1.2, linestyle="--", alpha=0.8)

    ax.set_xlabel("ls_bps (bps)")
    ax.set_ylabel("density")
    ax.set_title("ls_bps distribution: all vs top-10% score_std")
    ax.legend(fontsize=8)
    ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # ── bottom-left: scatter score_std vs ls_bps + binned mean ───────────────
    ax = axes[1][0]
    ax.scatter(score_std, ls_bps, s=4, alpha=0.18, color="#4575b4", linewidths=0)

    # binned mean (N_BINS equal-quantile bins)
    ticks_df2 = ticks_df.copy()
    ticks_df2["bin"] = pd.qcut(ticks_df2["score_std"], q=N_BINS,
                                labels=False, duplicates="drop")
    bin_stats = (ticks_df2.groupby("bin")
                           .agg(x=("score_std", "mean"), y=("ls_bps", "mean"))
                           .reset_index())
    ax.plot(bin_stats["x"], bin_stats["y"], color="#d73027",
            marker="o", markersize=6, linewidth=2.0, label="bin mean", zorder=5)

    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.axvline(threshold, color="#d73027", linewidth=1.2, linestyle="--",
               alpha=0.6, label=f"top {int(TOP_PCT*100)}% threshold")
    ax.set_xlabel("score_std")
    ax.set_ylabel("ls_bps (bps)")
    ax.set_title("score_std vs ls_bps  (each dot = one tick)")
    ax.legend(fontsize=8)
    ax.grid(linewidth=0.4, alpha=0.4)

    # ── bottom-right: sorted score curve for two cross-sections ──────────────
    ax = axes[1][1]
    colors = {"highest score_std": "#d73027", "median score_std": "#4575b4"}
    for label, info in cross_sections.items():
        scores = np.sort(info["scores"])
        ax.plot(np.arange(1, len(scores) + 1), scores,
                color=colors[label], linewidth=1.8,
                label=f"{label}\n(std={info['score_std']}, {info['date']} {info['time']})")

    ax.set_xlabel("stock rank (sorted by score, low → high)")
    ax.set_ylabel("model score")
    ax.set_title("cross-section score distribution (2 example ticks)")
    ax.legend(fontsize=7.5)
    ax.grid(linewidth=0.4, alpha=0.4)

    fig.suptitle(
        f"score_std exploration — {SETTING} / {POOL} / g{N_GROUPS} / {RET_H}  "
        f"(slot >= 14:30, test set)",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"loading data from {PRED_DIR} ...")
    ticks_df, cross_sections = load_data(PRED_DIR)
    print(f"  {len(ticks_df):,} ticks loaded")
    print(f"  score_std range: [{ticks_df['score_std'].min():.4f}, "
          f"{ticks_df['score_std'].max():.4f}]")
    print(f"  ls_bps range:    [{ticks_df['ls_bps'].min():.1f}, "
          f"{ticks_df['ls_bps'].max():.1f}]")
    threshold = np.quantile(ticks_df["score_std"].values, 1 - TOP_PCT)
    mask = ticks_df["score_std"] >= threshold
    print(f"  top {int(TOP_PCT*100)}% threshold: {threshold:.4f}  "
          f"({mask.sum()} ticks,  mean ls_bps = {ticks_df.loc[mask, 'ls_bps'].mean():.2f})")
    print("plotting ...")
    plot(ticks_df, cross_sections, OUT_PATH)


if __name__ == "__main__":
    main()
