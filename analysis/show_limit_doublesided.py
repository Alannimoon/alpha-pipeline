"""
展示三类样本（各取 n_samples 条）：
  1. 漏报涨停：LastPrice==HighLimitPrice 但双边盘口仍存在（我们没检测到）
  2. 漏报跌停：LastPrice==LowLimitPrice  但双边盘口仍存在
  3. 误报涨停：单边无卖盘（我们判断为涨停）但 LastPrice < HighLimitPrice
  4. 误报跌停：单边无买盘（我们判断为跌停）但 LastPrice > LowLimitPrice

通过 --summary-csv 精准定位目标日期，只扫那几天的 ZIP。
"""

import os
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

DEFAULT_DATA_ROOT  = "/home/fund/data"
DEFAULT_A500_CSV   = os.path.join(os.path.dirname(__file__), "..", "config", "a500.csv")
CHUNKSIZE          = 200_000
ENCODING           = "gb18030"
N_SAMPLES          = 5


def norm6(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return "".join(ch for ch in s if ch.isdigit()).zfill(6)


def is_trade_like_sz(phase: pd.Series) -> pd.Series:
    s = phase.astype(str).str.strip()
    return s.str[0].eq("T") & ~s.str[1].eq("1")


def calc_wmid(df):
    num = df["BidPrice1"] * df["AskVolume1"] + df["AskPrice1"] * df["BidVolume1"]
    den = df["BidVolume1"] + df["AskVolume1"]
    return (num / den).round(4)


def read_day(zip_path, sz_codes):
    NEED = ["SecurityID", "UpdateTime", "TradingPhaseCode",
            "LastPrice", "HighLimitPrice", "LowLimitPrice",
            "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"]
    chunks = []
    with zipfile.ZipFile(zip_path) as zf:
        inner = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(inner) as f:
            for ck in pd.read_csv(f, chunksize=CHUNKSIZE, encoding=ENCODING,
                                  encoding_errors="replace", skipinitialspace=True,
                                  index_col=False):
                ck["SecurityID"] = ck["SecurityID"].apply(norm6)
                ck = ck[ck["SecurityID"].isin(sz_codes)]
                if not ck.empty:
                    chunks.append(ck[NEED].copy())
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    for c in ["LastPrice", "HighLimitPrice", "LowLimitPrice",
              "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _process_day_worker(args_tuple):
    zip_path, sz_codes, day, data_root = args_tuple
    df = read_day(zip_path, sz_codes)
    if df is None:
        return day, {}
    tl         = is_trade_like_sz(df["TradingPhaseCode"])
    bid_exists = df["BidPrice1"].gt(0) & df["BidVolume1"].gt(0) & df["BidPrice1"].notna()
    ask_exists = df["AskPrice1"].gt(0) & df["AskVolume1"].gt(0) & df["AskPrice1"].notna()
    hi_valid   = df["HighLimitPrice"].lt(999999999) & df["HighLimitPrice"].notna()
    lo_valid   = df["LowLimitPrice"].gt(0.01) & df["LowLimitPrice"].notna()
    masks = [
        (tl & hi_valid & df["LastPrice"].eq(df["HighLimitPrice"]) & bid_exists & ask_exists,
         "miss_up", "HighLimitPrice"),
        (tl & lo_valid & df["LastPrice"].eq(df["LowLimitPrice"]) & bid_exists & ask_exists,
         "miss_dn", "LowLimitPrice"),
        (tl & hi_valid & ~ask_exists & bid_exists & ~df["LastPrice"].eq(df["HighLimitPrice"]),
         "fp_up", "HighLimitPrice"),
        (tl & lo_valid & ~bid_exists & ask_exists & ~df["LastPrice"].eq(df["LowLimitPrice"]),
         "fp_dn", "LowLimitPrice"),
    ]
    return day, extract_samples(df, day, masks)


def extract_samples(df, day, masks_labels):
    """对多个 mask 各取样本，返回 {label: DataFrame}"""
    results = {}
    for mask, label, limit_col in masks_labels:
        sub = df[mask].copy()
        if sub.empty:
            continue
        sub["Date"]        = day
        sub["wmid"]        = calc_wmid(sub)
        sub["spread"]      = (sub["AskPrice1"] - sub["BidPrice1"]).round(4)
        sub["diff_to_lim"] = (sub["LastPrice"] - sub[limit_col]).round(4)
        results[label] = sub
    return results


def show(samples_list, label, limit_col, n):
    if not samples_list:
        print(f"\n[{label}] 未找到样本")
        return
    df = pd.concat(samples_list, ignore_index=True).head(n)
    cols = ["Date", "UpdateTime", "SecurityID",
            "LastPrice", limit_col, "diff_to_lim",
            "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",
            "wmid", "spread"]
    print(f"\n{'='*70}")
    print(f"【{label}】共展示 {len(df)} 条")
    print('='*70)
    print(df[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",    default=DEFAULT_DATA_ROOT)
    ap.add_argument("--universe-csv", default=DEFAULT_A500_CSV)
    ap.add_argument("--summary-csv",  default=None,
                    help="check_limit 输出的明细 CSV，用于精准定位目标日期")
    ap.add_argument("--days", nargs="*", default=None,
                    help="直接指定要读的日期列表，如 --days 20250113 20250217")
    ap.add_argument("--n-samples",    type=int, default=N_SAMPLES)
    ap.add_argument("--workers",      type=int, default=4)
    args = ap.parse_args()

    sz_codes = set(
        pd.read_csv(args.universe_csv)
        .assign(SecurityID=lambda d: d["SecurityID"].astype(str).str.zfill(6))
        ["SecurityID"]
        .pipe(lambda s: s[~s.str.startswith("6")])
    )

    if args.days:
        all_days     = sorted(args.days)
        days_miss_up = days_miss_dn = days_fp_up = days_fp_dn = all_days
    elif args.summary_csv and os.path.exists(args.summary_csv):
        sm = pd.read_csv(args.summary_csv, dtype={"Date": str})
        days_miss_up = sorted(sm[sm["up_exchange_only"] > 0]["Date"].tolist())
        days_miss_dn = sorted(sm[sm["dn_exchange_only"] > 0]["Date"].tolist())
        days_fp_up   = sorted(sm[sm["up_ours_only"]    > 0]["Date"].tolist())
        days_fp_dn   = sorted(sm[sm["dn_ours_only"]    > 0]["Date"].tolist())
        all_days = sorted(set(days_miss_up + days_miss_dn + days_fp_up + days_fp_dn))
    else:
        raise ValueError("请指定 --days 或 --summary-csv")

    miss_up_samples, miss_dn_samples = [], []
    fp_up_samples,   fp_dn_samples   = [], []

    tasks = [
        (os.path.join(args.data_root, d, "mdl_6_28_0.csv.zip"), sz_codes, d, args.data_root)
        for d in all_days
        if os.path.exists(os.path.join(args.data_root, d, "mdl_6_28_0.csv.zip"))
    ]
    print(f"并行读取 {len(tasks)} 天（workers={args.workers}）...")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_day_worker, t): t[2] for t in tasks}
        for fut in as_completed(futs):
            day, res = fut.result()
            print(f"  完成 {day}")
            if "miss_up" in res: miss_up_samples.append(res["miss_up"])
            if "miss_dn" in res: miss_dn_samples.append(res["miss_dn"])
            if "fp_up"   in res: fp_up_samples.append(res["fp_up"])
            if "fp_dn"   in res: fp_dn_samples.append(res["fp_dn"])

    n = args.n_samples
    show(miss_up_samples, "漏报涨停：已涨停但盘口双边（我们没检测到）", "HighLimitPrice", n)
    show(miss_dn_samples, "漏报跌停：已跌停但盘口双边（我们没检测到）", "LowLimitPrice",  n)
    show(fp_up_samples,   "误报涨停：单边无卖盘但未达涨停价",           "HighLimitPrice", n)
    show(fp_dn_samples,   "误报跌停：单边无买盘但未达跌停价",           "LowLimitPrice",  n)


if __name__ == "__main__":
    main()
