"""
为 vol100_v2 的三类股票分别生成价格走势图。

输出目录：
  result/vol100_v2_price_charts/added/    — 17 只新加入
  result/vol100_v2_price_charts/removed/  — 17 只被剔除
  result/vol100_v2_price_charts/kept/     — 83 只保留
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

from config import TEST_BASE_ROOT, BASE_ROOT

VOL100_CSV = os.path.join(ROOT, "config", "vol_top100.csv")
OUT_ROOT   = os.path.join(ROOT, "result", "vol100_v2_price_charts")

HIGH_PRICE_STOCKS = [
    "688256", "688027", "300502", "688072", "688012",
    "300476", "301308", "688411", "688615", "603986",
    "001309", "002916", "688195", "688525", "688031",
    "688048", "688347",
]

ADDED_CSV   = os.path.join(ROOT, "result", "eval", "analysis", "vol100_v2", "added.csv")

vol100 = pd.read_csv(VOL100_CSV, dtype=str)["SecurityID"].tolist()
added  = pd.read_csv(ADDED_CSV,  dtype=str)["secid"].tolist()
kept   = [s for s in vol100 if s not in HIGH_PRICE_STOCKS]

# 43 个测试日期
test_dates = sorted(
    os.path.basename(f).replace(".parquet", "")
    for f in glob.glob(os.path.join(TEST_BASE_ROOT, "*.parquet"))
)
print(f"测试日期: {test_dates[0]} ~ {test_dates[-1]}，共 {len(test_dates)} 天")
print(f"added={len(added)}, removed={len(HIGH_PRICE_STOCKS)}, kept={len(kept)}")


def load_prices(secids, base_dir, dates=None):
    """从 base_dir 读取指定股票的价格数据，返回 {secid: DataFrame}。"""
    sid_set = set(secids)
    files   = sorted(glob.glob(os.path.join(base_dir, "*.parquet")))
    if dates is not None:
        date_set = set(dates)
        files = [f for f in files if os.path.basename(f).replace(".parquet", "") in date_set]

    chunks = []
    for i, f in enumerate(files, 1):
        print(f"  loading {os.path.basename(f)} [{i}/{len(files)}]", end="\r", flush=True)
        df = pd.read_parquet(f, columns=["Date", "SampleTime", "SecurityID", "Price", "CanUsePrice"])
        df["SecurityID"] = df["SecurityID"].astype(str).str.zfill(6)
        df = df[df["SecurityID"].isin(sid_set)]
        chunks.append(df)
    print()

    if not chunks:
        return {}
    all_df = pd.concat(chunks, ignore_index=True)
    all_df = all_df.sort_values(["SecurityID", "Date", "SampleTime"])
    return {sid: grp.reset_index(drop=True) for sid, grp in all_df.groupby("SecurityID")}


def draw_charts(stock_data, secids, out_dir, label_map=None):
    """stock_data: {secid: DataFrame}，为每只股票画一张图存到 out_dir。"""
    os.makedirs(out_dir, exist_ok=True)
    total = len(secids)
    for i, secid in enumerate(secids, 1):
        if secid not in stock_data:
            print(f"  [{i:3d}/{total}] {secid}: no data, skip")
            continue

        df     = stock_data[secid]
        prices = df["Price"].values

        # 按天找边界
        boundaries = []
        for d, grp in df.groupby("Date", sort=True):
            boundaries.append((grp.index[0], str(d)))

        valid  = prices[np.isfinite(prices)]
        y_min  = valid.min() * 0.98 if len(valid) else 0
        y_max  = valid.max() * 1.02 if len(valid) else 1

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(prices, linewidth=0.4, color="steelblue", rasterized=True)

        for tick_idx, _ in boundaries:
            ax.axvline(tick_idx, color="gray", linewidth=0.3, alpha=0.5)

        step = max(1, len(boundaries) // 10)
        for tick_idx, date_str in boundaries[::step]:
            ax.text(tick_idx, y_max * 0.99, date_str[4:],
                    fontsize=5, color="gray", va="top", ha="left")

        ax.axhline(10, color="orange", linewidth=0.8, linestyle="--", alpha=0.7, label="10 yuan")

        tag = label_map.get(secid, "") if label_map else ""
        ax.set_title(f"{secid}  {tag}", fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, len(prices))
        ax.set_xlabel("Tick index")
        ax.set_ylabel("Price (yuan)")
        ax.legend(fontsize=7, loc="upper right")

        plt.savefig(os.path.join(out_dir, f"{secid}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

        if i % 5 == 0 or i == total:
            print(f"  [{i:3d}/{total}] {secid} done")


# ── 加载数据 ──────────────────────────────────────────────────────────────────

print("\n[1/2] Loading removed + kept stocks from test base...")
orig_data = load_prices(HIGH_PRICE_STOCKS + kept, TEST_BASE_ROOT)

print("[2/2] Loading added stocks from full base (test dates only)...")
added_data = load_prices(added, BASE_ROOT, dates=test_dates)

# ── 画图 ──────────────────────────────────────────────────────────────────────

print(f"\nDrawing removed ({len(HIGH_PRICE_STOCKS)} stocks)...")
draw_charts(orig_data, HIGH_PRICE_STOCKS,
            os.path.join(OUT_ROOT, "removed"),
            label_map={s: "[removed: high price]" for s in HIGH_PRICE_STOCKS})

print(f"\nDrawing added ({len(added)} stocks)...")
draw_charts(added_data, added,
            os.path.join(OUT_ROOT, "added"),
            label_map={s: "[added: high vol]" for s in added})

print(f"\nDrawing kept ({len(kept)} stocks)...")
draw_charts(orig_data, kept,
            os.path.join(OUT_ROOT, "kept"),
            label_map={s: "[kept]" for s in kept})

print(f"\nAll done. Charts saved to {OUT_ROOT}")

# ── membership txt ────────────────────────────────────────────────────────────
A500_CSV = os.path.join(ROOT, "config", "a500.csv")
a500_set = set(pd.read_csv(A500_CSV, dtype=str)["SecurityID"].tolist())

def mean_price_map(stock_data):
    result = {}
    for sid, df in stock_data.items():
        valid = df.loc[df["CanUsePrice"], "Price"]
        result[sid] = valid.mean() if len(valid) else float("nan")
    return result

mp_orig  = mean_price_map(orig_data)
mp_added = mean_price_map(added_data)
mp_all   = {**mp_orig, **mp_added}

def section(title, secids, mp):
    in_a500  = sorted([s for s in secids if s in a500_set])
    not_a500 = sorted([s for s in secids if s not in a500_set])
    lines = [f"=== {title} ==="]
    lines.append(f"--- In A500 ({len(in_a500)}) ---")
    for s in in_a500:
        lines.append(f"{s}  avg={mp.get(s, float('nan')):.2f}")
    lines.append(f"--- Not in A500 ({len(not_a500)}) ---")
    for s in not_a500:
        lines.append(f"{s}  avg={mp.get(s, float('nan')):.2f}")
    return lines

lines = []
lines += section(f"Removed ({len(HIGH_PRICE_STOCKS)})", HIGH_PRICE_STOCKS, mp_all)
lines.append("")
lines += section(f"Added ({len(added)})", added, mp_all)
lines.append("")
lines += section(f"Kept ({len(kept)})", kept, mp_all)

txt_path = os.path.join(OUT_ROOT, "membership.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Membership txt saved to {txt_path}")
