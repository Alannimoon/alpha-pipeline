"""
Plot time-slot PnL for the 4 test configurations across 3 factor pools.

4 curves per plot:
  A500  +7   (market_state)
  A500  +23  (market_state_vol_turnover)
  vol100 +7  (market_state_vol100)
  vol100 +23 (market_state_vol_turnover_vol100)

X-axis: 8 intraday slots
Y-axis: avg PnL (bps/tick)

Usage:
  python scripts/plot_new_features.py [--input PATH] [--output-dir DIR]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SLOT_COLS = [
    "avg_pnl_0930_1000",
    "avg_pnl_1000_1030",
    "avg_pnl_1030_1100",
    "avg_pnl_1100_1130",
    "avg_pnl_1300_1330",
    "avg_pnl_1330_1400",
    "avg_pnl_1400_1430",
    "avg_pnl_1430_1457",
]
SLOT_LABELS = [
    "09:30", "10:00", "10:30", "11:00",
    "13:00", "13:30", "14:00", "14:30",
]

CURVES = [
    ("market_state",                     "A500 +7"),
    ("market_state_vol_turnover",        "A500 +23"),
    ("market_state_vol100",              "vol100 +7"),
    ("market_state_vol_turnover_vol100", "vol100 +23"),
]

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
MARKERS = ["o", "s", "^", "D"]


def plot_pool(df_test: pd.DataFrame, pool: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for (tag, label), color, marker in zip(CURVES, COLORS, MARKERS):
        row = df_test[(df_test["tag"] == tag) & (df_test["factor_pool"] == pool)]
        if row.empty:
            continue
        vals = row.iloc[0][SLOT_COLS].astype(float).tolist()
        ax.plot(SLOT_LABELS, vals, label=label, color=color,
                marker=marker, linewidth=1.8, markersize=6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Factor pool: {pool}  |  g20 / ret300 / test", fontsize=12)
    ax.set_xlabel("Intraday slot (start time)")
    ax.set_ylabel("Avg long-short PnL (bps/tick)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(out_dir, f"slot_pnl_{pool}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "result/eval/pnl/pnl_xgb_new_features.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "result/eval/pnl/plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df_test = df[
        (df["split"] == "test") &
        (df["n_groups"] == 20) &
        (df["ret_horizon"] == "ret300")
    ].copy()

    os.makedirs(args.output_dir, exist_ok=True)

    for pool in ["all", "intersection", "union"]:
        plot_pool(df_test, pool, args.output_dir)


if __name__ == "__main__":
    main()
