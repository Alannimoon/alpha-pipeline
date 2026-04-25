"""
构建第二测试集股票池 vol_top100_v2。

方法：
- 剔除当前 vol100 中的高价股（688>=100, 其他>=200）
- 从 A500 候选股中按 43 天 ret_fwd_300 标准差排序，补入高波动股至 100 只
- 均价 = 43天所有 CanUsePrice=True 的 tick Price 均值
- 波动率 = 43天所有有效 ret_fwd_300 的标准差
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import TEST_BASE_ROOT, BASE_ROOT

VOL100_CSV = os.path.join(ROOT, "config", "vol_top100.csv")
A500_CSV   = os.path.join(ROOT, "config", "a500.csv")
OUT_CSV    = os.path.join(ROOT, "config", "vol_top100_v2.csv")

HIGH_PRICE_STOCKS = [
    "688256", "688027", "300502", "688072", "688012",
    "300476", "301308", "688411", "688615", "603986",
    "001309", "002916", "688195", "688525", "688031",
    "688048", "688347",
]

vol100     = pd.read_csv(VOL100_CSV, dtype=str)["SecurityID"].tolist()
a500       = pd.read_csv(A500_CSV,   dtype=str)["SecurityID"].tolist()
candidates = [s for s in a500 if s not in vol100]

# 43个测试日期，从 TEST_BASE_ROOT 的文件名读取
test_dates = sorted(
    os.path.basename(f).replace(".parquet", "")
    for f in glob.glob(os.path.join(TEST_BASE_ROOT, "*.parquet"))
)
print(f"测试日期: {test_dates[0]} ~ {test_dates[-1]}，共 {len(test_dates)} 天")
print(f"A500 候选股（不在当前 vol100）: {len(candidates)} 只")

# 按股票累积 Price 和 ret_fwd_300
cand_set   = set(candidates)
prices     = {s: [] for s in candidates}
rets       = {s: [] for s in candidates}

for i, date in enumerate(test_dates, 1):
    print(f"  [{i:2d}/{len(test_dates)}] {date}", end="\r", flush=True)
    path = os.path.join(BASE_ROOT, f"{date}.parquet")
    if not os.path.exists(path):
        print(f"  警告: 缺少 {date}，跳过")
        continue
    df = pd.read_parquet(path, columns=["SecurityID", "Price", "CanUsePrice", "ret_fwd_300"])
    df = df[df["SecurityID"].isin(cand_set)]

    for sid, grp in df.groupby("SecurityID"):
        valid_price = grp.loc[grp["CanUsePrice"], "Price"]
        valid_ret   = grp["ret_fwd_300"].dropna()
        prices[sid].extend(valid_price.tolist())
        rets  [sid].extend(valid_ret.tolist())
print(f"  读取完成，共 {len(test_dates)} 天")

# 汇总
rows = []
for sid in candidates:
    p = prices[sid]
    r = rets  [sid]
    mean_price = np.mean(p) if p else np.nan
    vol        = np.std(r)  if r else np.nan
    is_688     = sid.startswith("688")
    threshold  = 100 if is_688 else 200
    is_high    = np.isfinite(mean_price) and mean_price >= threshold
    rows.append({
        "secid":        sid,
        "is_688":       is_688,
        "mean_price":   round(mean_price, 2) if np.isfinite(mean_price) else np.nan,
        "threshold":    threshold,
        "is_high_price": is_high,
        "vol_ret300":   vol,
    })

df_cand  = pd.DataFrame(rows)
no_data  = df_cand["mean_price"].isna().sum()
if no_data:
    print(f"  注意: {no_data} 只股票无数据（不在A500 base里），已排除")

df_valid = (df_cand[~df_cand["is_high_price"] & df_cand["vol_ret300"].notna()]
            .sort_values("vol_ret300", ascending=False))

df_all = df_cand[df_cand["vol_ret300"].notna()].sort_values("vol_ret300", ascending=False)
print(f"\n候选中有数据: {len(df_all)} 只（其中非高价股 {len(df_valid)} 只）")
print(f"\n--- 全部 {len(df_all)} 只候选股（按 ret_fwd_300 标准差排序）---")
print(f"{'rank':>5}  {'secid':<10} {'均价':>8}  {'vol_ret300':>12}  {'688?':>5}  {'合格':>5}")
for rank, (_, r) in enumerate(df_all.iterrows(), 1):
    ok = "YES" if not r.is_high_price else "NO "
    print(f"{rank:>5}  {r.secid:<10} {r.mean_price:>8.2f}  {r.vol_ret300:>12.6f}  {'是' if r.is_688 else '否':>5}  {ok:>5}")

# 补入 17 只
n_add  = len(HIGH_PRICE_STOCKS)
to_add = df_valid.head(n_add)["secid"].tolist()
keep   = [s for s in vol100 if s not in HIGH_PRICE_STOCKS]
new_list = sorted(keep + to_add)

print(f"\n{'='*55}")
print(f"剔除高价股: {len(HIGH_PRICE_STOCKS)} 只")
print(f"补入高波动股: {len(to_add)} 只")
print(f"新 vol100 共: {len(new_list)} 只")
print(f"{'='*55}")
print(f"\n补入的 {len(to_add)} 只:")
for sid in to_add:
    r = df_valid[df_valid["secid"] == sid].iloc[0]
    print(f"  {sid}  均价={r.mean_price:.2f}  vol={r.vol_ret300:.6f}  688={'是' if r.is_688 else '否'}")

OUT_DIR = os.path.join(ROOT, "result", "eval", "analysis", "vol100_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# vol_top100_v2.csv
pd.DataFrame({"SecurityID": new_list}).to_csv(OUT_CSV, index=False)

# 全部候选股排序结果
df_all_out = df_all.reset_index(drop=True)
df_all_out.insert(0, "rank", range(1, len(df_all_out) + 1))
df_all_out["qualified"] = ~df_all_out["is_high_price"]
df_all_out.to_csv(os.path.join(OUT_DIR, "candidates_ranked.csv"), index=False)

# 被替换列表（剔除的 17 只）—— 从 TEST_BASE_ROOT 读（原 vol100 都在里面）
removed_prices = {s: [] for s in HIGH_PRICE_STOCKS}
hp_set = set(HIGH_PRICE_STOCKS)
for date in test_dates:
    path = os.path.join(TEST_BASE_ROOT, f"{date}.parquet")
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path, columns=["SecurityID", "Price", "CanUsePrice"])
    df = df[df["SecurityID"].isin(hp_set) & df["CanUsePrice"]]
    for sid, grp in df.groupby("SecurityID"):
        removed_prices[sid].extend(grp["Price"].tolist())

removed_rows = []
for sid in HIGH_PRICE_STOCKS:
    p = removed_prices[sid]
    mean_p = round(np.mean(p), 2) if p else np.nan
    is_688 = sid.startswith("688")
    removed_rows.append({
        "secid":       sid,
        "mean_price":  mean_p,
        "is_688":      is_688,
        "threshold":   100 if is_688 else 200,
        "reason":      "high_price",
    })
pd.DataFrame(removed_rows).to_csv(os.path.join(OUT_DIR, "removed.csv"), index=False)

# 替换列表（补入的 17 只）
added_rows = []
for sid in to_add:
    r = df_valid[df_valid["secid"] == sid].iloc[0]
    added_rows.append({
        "secid":       sid,
        "mean_price":  r.mean_price,
        "vol_ret300":  r.vol_ret300,
        "is_688":      r.is_688,
    })
pd.DataFrame(added_rows).to_csv(os.path.join(OUT_DIR, "added.csv"), index=False)

print(f"\n输出目录: {OUT_DIR}")
print(f"  candidates_ranked.csv  — {len(df_all_out)} 只候选股排序")
print(f"  removed.csv            — {len(removed_rows)} 只被剔除高价股")
print(f"  added.csv              — {len(added_rows)} 只补入高波动股")
print(f"  {OUT_CSV}")
print(f"  vol_top100_v2.csv      — 新 100 只股票池")
