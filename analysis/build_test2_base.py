"""
生成 result/test2/base/：vol_top100_v2 × 43 个测试日的 base 数据。

数据来源：
  83 只 kept 股票 → result/test/base/{date}.parquet
  17 只 added 股票 → result/base/{date}.parquet
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import TEST_BASE_ROOT, BASE_ROOT, TEST2_BASE_ROOT

VOL100_V2_CSV = os.path.join(ROOT, "config", "vol_top100_v2.csv")
ADDED_CSV     = os.path.join(ROOT, "result", "eval", "analysis", "vol100_v2", "added.csv")

v2    = pd.read_csv(VOL100_V2_CSV, dtype=str)["SecurityID"].tolist()
added = pd.read_csv(ADDED_CSV,     dtype=str)["secid"].tolist()
kept  = [s for s in v2 if s not in added]

added_set = set(added)
kept_set  = set(kept)

print(f"kept: {len(kept)} 只，added: {len(added)} 只，total: {len(v2)} 只")

test_dates = sorted(
    os.path.basename(f).replace(".parquet", "")
    for f in glob.glob(os.path.join(TEST_BASE_ROOT, "*.parquet"))
)
print(f"测试日期: {test_dates[0]} ~ {test_dates[-1]}，共 {len(test_dates)} 天")

os.makedirs(TEST2_BASE_ROOT, exist_ok=True)

for i, date in enumerate(test_dates, 1):
    print(f"  [{i:2d}/{len(test_dates)}] {date}", end="\r", flush=True)

    # 83 只 kept → test/base
    test_path = os.path.join(TEST_BASE_ROOT, f"{date}.parquet")
    df_kept = pd.read_parquet(test_path)
    df_kept = df_kept[df_kept["SecurityID"].isin(kept_set)]

    # 17 只 added → base（A500全量）
    base_path = os.path.join(BASE_ROOT, f"{date}.parquet")
    df_added = pd.read_parquet(base_path)
    df_added = df_added[df_added["SecurityID"].isin(added_set)]

    df_day = pd.concat([df_kept, df_added], ignore_index=True)
    df_day = df_day.sort_values(["SecurityID", "SampleTime"]).reset_index(drop=True)

    out_path = os.path.join(TEST2_BASE_ROOT, f"{date}.parquet")
    df_day.to_parquet(out_path, index=False)

print(f"\n完成，共生成 {len(test_dates)} 个 parquet → {TEST2_BASE_ROOT}")

# 简单校验
sample = pd.read_parquet(os.path.join(TEST2_BASE_ROOT, f"{test_dates[0]}.parquet"))
print(f"校验 {test_dates[0]}: {sample['SecurityID'].nunique()} 只股票，{len(sample):,} 行")
missing = set(v2) - set(sample["SecurityID"].unique())
if missing:
    print(f"  警告：缺失 {len(missing)} 只: {missing}")
else:
    print(f"  100 只全部到位 ✓")
