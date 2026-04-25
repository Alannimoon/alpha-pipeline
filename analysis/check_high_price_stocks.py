"""
检查 vol_top100 中的高价股数量。
高价股定义：688 开头且均价 >= 100，或其他股票均价 >= 200。
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import TEST_BASE_ROOT

VOL100_CSV = os.path.join(ROOT, "config", "vol_top100.csv")

vol100 = pd.read_csv(VOL100_CSV, dtype=str)["SecurityID"].tolist()

# 从所有测试集 parquet 汇总每只股票的均价
files = sorted(glob.glob(os.path.join(TEST_BASE_ROOT, "*.parquet")))
assert files, f"No parquet files found in {TEST_BASE_ROOT}"

prices = {s: [] for s in vol100}

for f in files:
    df = pd.read_parquet(f, columns=["SecurityID", "Price", "CanUsePrice"])
    df = df[df["SecurityID"].isin(vol100) & df["CanUsePrice"]]
    for sid, grp in df.groupby("SecurityID"):
        prices[sid].extend(grp["Price"].tolist())

rows = []
for sid in vol100:
    vals = prices[sid]
    mean_price = np.mean(vals) if vals else np.nan
    is_688 = sid.startswith("688")
    threshold = 100 if is_688 else 200
    is_high = mean_price >= threshold
    rows.append({"secid": sid, "is_688": is_688, "mean_price": round(mean_price, 2),
                 "threshold": threshold, "is_high_price": is_high})

df_res = pd.DataFrame(rows).sort_values("mean_price", ascending=False)

high = df_res[df_res["is_high_price"]]
normal = df_res[~df_res["is_high_price"]]

print(f"\n{'='*55}")
print(f"总计: {len(vol100)} 只  |  高价股: {len(high)} 只  |  正常: {len(normal)} 只")
print(f"{'='*55}")

print(f"\n--- 高价股（需剔除，共 {len(high)} 只）---")
print(f"{'secid':<12} {'均价':>8}  {'阈值':>6}  {'688?':>5}")
for _, r in high.iterrows():
    print(f"{r.secid:<12} {r.mean_price:>8.2f}  {r.threshold:>6}  {'是' if r.is_688 else '否':>5}")

print(f"\n--- 正常股票（保留，共 {len(normal)} 只）---")
print(f"{'secid':<12} {'均价':>8}  {'688?':>5}")
for _, r in normal.sort_values("mean_price", ascending=False).iterrows():
    print(f"{r.secid:<12} {r.mean_price:>8.2f}  {'是' if r.is_688 else '否':>5}")

print(f"\n688 开头合计: {df_res['is_688'].sum()} 只，其中高价: {(high['is_688']).sum()} 只")
print(f"非 688 合计:  {(~df_res['is_688']).sum()} 只，其中高价: {(~high['is_688']).sum()} 只")
