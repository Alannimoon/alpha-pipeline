"""
Draw 100 price line charts for vol100 stocks over the 43-day test set.
Each chart plots tick-level Price as a continuous line.
Y-axis is unified across all charts; 10 and 30 yuan reference lines are drawn.

Outputs:
  result/vol100_price_charts/<secid>.png   (100 files)
  result/vol100_membership.csv             (secid, in_A500)
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import TEST_BASE_ROOT

VOL100_CSV = os.path.join(ROOT, "config", "vol_top100.csv")
A500_CSV   = os.path.join(ROOT, "config", "a500.csv")
OUT_DIR    = os.path.join(ROOT, "result", "vol100_price_charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── membership ────────────────────────────────────────────────────────────────
vol100 = pd.read_csv(VOL100_CSV)["SecurityID"].astype(str).str.zfill(6).tolist()
a500   = set(pd.read_csv(A500_CSV)["SecurityID"].astype(str).str.zfill(6))

in_a500_list  = sorted([s for s in vol100 if s in a500])
not_a500_list = sorted([s for s in vol100 if s not in a500])
print(f"A500 overlap: {len(in_a500_list)} / {len(vol100)}")

# ── load all test base parquets ───────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(TEST_BASE_ROOT, "*.parquet")))
if not files:
    sys.exit(f"No parquet files in {TEST_BASE_ROOT}")
print(f"Loading {len(files)} days ...")

chunks = []
for f in files:
    df = pd.read_parquet(f, columns=["Date", "SampleTime", "SecurityID", "Price", "CanUsePrice"])
    df["SecurityID"] = df["SecurityID"].astype(str).str.zfill(6)
    df = df[df["SecurityID"].isin(vol100)]
    chunks.append(df)

all_data = pd.concat(chunks, ignore_index=True)
all_data = all_data.sort_values(["SecurityID", "Date", "SampleTime"])
print(f"Total rows (vol100 only): {len(all_data):,}")

# ── global y-axis range (use valid prices only) ───────────────────────────────
valid_prices = all_data.loc[all_data["CanUsePrice"], "Price"]
y_min = max(0, valid_prices.min() * 0.95)
y_max = valid_prices.max() * 1.05
print(f"Global price range: {y_min:.2f} - {y_max:.2f}")

# ── find day-boundary tick indices for x-axis labels ─────────────────────────
# We'll build per-stock tick index; day labels from the first stock as reference
ref_stock = vol100[0]
ref = all_data[all_data["SecurityID"] == ref_stock].reset_index(drop=True)
dates = ref["Date"].astype(str).unique()  # sorted already

def get_day_boundaries(stock_df):
    """Return list of (tick_index, date_str) at start of each day."""
    boundaries = []
    for d in stock_df["Date"].astype(str).unique():
        idx = stock_df[stock_df["Date"].astype(str) == d].index[0]
        boundaries.append((idx, d))
    return boundaries

# ── draw charts ───────────────────────────────────────────────────────────────
vol100_set = set(vol100)
grouped = all_data.groupby("SecurityID")

for i, secid in enumerate(vol100):
    if secid not in grouped.groups:
        print(f"  [{i+1}/100] {secid}: no data, skipping")
        continue

    stock_df = grouped.get_group(secid).reset_index(drop=True)
    prices = stock_df["Price"].values  # NaN where CanUsePrice is False

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(prices, linewidth=0.4, color="steelblue", rasterized=True)

    # day boundary vertical lines + date labels
    boundaries = get_day_boundaries(stock_df)
    for tick_idx, date_str in boundaries:
        ax.axvline(tick_idx, color="gray", linewidth=0.3, alpha=0.5)
    # label every ~5th day to avoid crowding
    step = max(1, len(boundaries) // 10)

    # per-stock y range
    valid = prices[~np.isnan(prices)]
    stock_ymin = valid.min() * 0.98 if len(valid) else 0
    stock_ymax = valid.max() * 1.02 if len(valid) else 1

    for tick_idx, date_str in boundaries[::step]:
        ax.text(tick_idx, stock_ymax * 0.99, date_str[4:],  # show MMDD
                fontsize=5, color="gray", va="top", ha="left")

    ax.axhline(10, color="orange", linewidth=0.8, linestyle="--", alpha=0.8, label="10 yuan")

    in_a500 = secid in a500
    ax.set_ylim(stock_ymin, stock_ymax)
    ax.set_xlim(0, len(prices))
    ax.set_xlabel("Tick index")
    ax.set_ylabel("Price (yuan)")
    ax.set_title(f"{secid}  {'[A500]' if in_a500 else '[non-A500]'}", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    out_path = os.path.join(OUT_DIR, f"{secid}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/100 done")

print(f"\nAll charts saved to {OUT_DIR}")

# ── compute mean price per stock and write membership txt ─────────────────────
mean_price = (
    all_data[all_data["CanUsePrice"]]
    .groupby("SecurityID")["Price"]
    .mean()
)

def fmt(secid):
    p = mean_price.get(secid, float("nan"))
    return f"{secid}  avg={p:.2f}"

out_lines = [
    f"=== In A500 ({len(in_a500_list)}) ===",
    *[fmt(s) for s in in_a500_list],
    "",
    f"=== Not in A500 ({len(not_a500_list)}) ===",
    *[fmt(s) for s in not_a500_list],
]
txt_path = os.path.join(ROOT, "result", "vol100_membership.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(out_lines) + "\n")
print(f"Membership txt saved to {txt_path}")
