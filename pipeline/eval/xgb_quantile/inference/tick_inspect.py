#!/usr/bin/env python
"""
Single cross-section traceback
  config : market_state / union / g20 / ret300, test set

Finds the tick with the highest ls_bps (g20 - g1) in the 14:30+ slot,
then traces back through:
  _scores.parquet  → top-5 and bottom-5 stocks by model score
  TEST_BASE         → ret_fwd_300, Price (wmid), BidPrice1/AskPrice1/BidVolume1/AskVolume1
                      at tick t and at tick t+300

Output: printed table + result/eval/pnl/inference/traceback.csv

Usage:
    python pipeline/eval/xgb_quantile/inference/traceback.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────
POOL       = "union"
N_GROUPS   = 20
RET_H      = "ret300"
SLOT_START = "14:30:00"
TOP_N      = 5       # top-N and bottom-N stocks to show

PRED_DIR  = os.path.join(
    config.EVAL_ROOT, "xgb_quantile_market_state_test",
    POOL, f"g{N_GROUPS}", RET_H,
)
BASE_ROOT = config.TEST_BASE_ROOT
OUT_CSV   = os.path.join(config.EVAL_ROOT, "pnl", "inference", "traceback.csv")

# base data columns to retrieve
BASE_COLS = [
    "Date", "SampleTime", "SecurityID",
    "Price", "PriceType",
    "BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1",
    "ret_fwd_300",
]


# ── step 1: find the tick with highest ls_bps ─────────────────────────────────

def find_best_tick(pred_dir: str) -> tuple[str, str, float]:
    """Return (date, sample_time, ls_bps) for the tick with highest g20-g1."""
    g_last  = f"g{N_GROUPS}"
    best    = (-np.inf, None, None)

    for p in sorted(Path(pred_dir).glob("????????.parquet")):
        try:
            df = pd.read_parquet(p, columns=["Date", "SampleTime", "g1", g_last])
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}")
            continue
        df = df[df["SampleTime"] >= SLOT_START].copy()
        if df.empty:
            continue
        df["ls"] = df[g_last] - df["g1"]
        idx = df["ls"].idxmax()
        row = df.loc[idx]
        val = float(row["ls"])
        if val > best[0]:
            best = (val, str(row["Date"]), str(row["SampleTime"]))

    ls_bps, date, stime = best[0] * 1e4, best[1], best[2]
    return date, stime, ls_bps


# ── step 2: get top-N / bottom-N stocks from scores ──────────────────────────

def get_top_bottom_stocks(pred_dir: str, date: str, stime: str) -> pd.DataFrame:
    """Return DataFrame with SecurityID, score, group (top/bottom)."""
    sp = Path(pred_dir) / f"{date}_scores.parquet"
    sdf = pd.read_parquet(sp, columns=["SampleTime", "SecurityID", "score"])
    sdf = sdf[sdf["SampleTime"] == stime].copy()
    sdf = sdf.sort_values("score").reset_index(drop=True)

    n = len(sdf)
    top    = sdf.tail(TOP_N).copy();    top["group"]    = f"top{TOP_N}"
    bottom = sdf.head(TOP_N).copy();    bottom["group"] = f"bottom{TOP_N}"
    return pd.concat([top, bottom], ignore_index=True)


# ── step 3: load base data for the 10 stocks ─────────────────────────────────

def load_base_for_tick(date: str, stime: str,
                       security_ids: list[str]) -> pd.DataFrame:
    """
    Load base parquet for `date`, filter to `stime` and the given stocks.
    Also loads the row at t+300 ticks to show the future price.
    """
    base_path = os.path.join(BASE_ROOT, f"{date}.parquet")
    cols = [c for c in BASE_COLS if c != "ret_fwd_300"] + ["ret_fwd_300"]

    df = pd.read_parquet(base_path, columns=cols)
    df = df[df["SecurityID"].astype(str).isin([str(s) for s in security_ids])]

    # all sample times sorted → find index of target time and t+300
    all_times = sorted(df["SampleTime"].unique())
    try:
        t_idx   = all_times.index(stime)
        t300_st = all_times[t_idx + 300] if t_idx + 300 < len(all_times) else None
    except ValueError:
        t300_st = None

    tick_t = df[df["SampleTime"] == stime].copy()
    tick_t = tick_t.rename(columns={
        "Price":       "Price_t",
        "PriceType":   "PriceType_t",
        "BidPrice1":   "BidPrice1_t",
        "AskPrice1":   "AskPrice1_t",
        "BidVolume1":  "BidVolume1_t",
        "AskVolume1":  "AskVolume1_t",
        "ret_fwd_300": "ret_fwd_300",
    })

    if t300_st:
        tick_t300 = (df[df["SampleTime"] == t300_st]
                     [["SecurityID", "Price", "BidPrice1", "AskPrice1",
                       "BidVolume1", "AskVolume1"]]
                     .rename(columns={
                         "Price":       "Price_t300",
                         "BidPrice1":   "BidPrice1_t300",
                         "AskPrice1":   "AskPrice1_t300",
                         "BidVolume1":  "BidVolume1_t300",
                         "AskVolume1":  "AskVolume1_t300",
                     }))
        tick_t = tick_t.merge(tick_t300, on="SecurityID", how="left")
    else:
        for col in ["Price_t300", "BidPrice1_t300", "AskPrice1_t300",
                    "BidVolume1_t300", "AskVolume1_t300"]:
            tick_t[col] = np.nan

    return tick_t.reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. find best tick
    print("scanning inference parquets ...")
    date, stime, ls_bps = find_best_tick(PRED_DIR)
    print(f"\n  best tick: {date}  {stime}  ls_bps = {ls_bps:.2f}")

    # 2. top / bottom stocks
    stocks = get_top_bottom_stocks(PRED_DIR, date, stime)
    print(f"  top {TOP_N} stocks (highest score): "
          f"{stocks[stocks['group']==f'top{TOP_N}']['SecurityID'].tolist()}")
    print(f"  bottom {TOP_N} stocks (lowest score): "
          f"{stocks[stocks['group']==f'bottom{TOP_N}']['SecurityID'].tolist()}")

    # 3. base data
    print("\nloading base data ...")
    base = load_base_for_tick(date, stime,
                              stocks["SecurityID"].astype(str).tolist())

    # 4. merge
    result = stocks.merge(base, on="SecurityID", how="left")
    result["ret_fwd_300_bps"] = (result["ret_fwd_300"] * 1e4).round(2)
    result["Price_t300_check"] = (result["Price_t"] *
                                  (1 + result["ret_fwd_300"])).round(4)

    # display columns
    show_cols = [
        "group", "SecurityID", "score",
        "Price_t", "BidPrice1_t", "AskPrice1_t", "BidVolume1_t", "AskVolume1_t",
        "ret_fwd_300_bps", "Price_t300", "Price_t300_check",
        "BidPrice1_t300", "AskPrice1_t300", "BidVolume1_t300", "AskVolume1_t300",
    ]
    show_cols = [c for c in show_cols if c in result.columns]
    result_show = result[show_cols].sort_values(["group", "score"],
                                                ascending=[True, False])

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n=== cross-section traceback: {date}  {stime} ===")
    print(f"    reported ls_bps = {ls_bps:.2f}")
    print(result_show.to_string(index=False))

    # 5. verify ls_bps
    top_mean    = result[result["group"] == f"top{TOP_N}"]["ret_fwd_300"].mean()
    bottom_mean = result[result["group"] == f"bottom{TOP_N}"]["ret_fwd_300"].mean()
    calc_ls     = (top_mean - bottom_mean) * 1e4
    print(f"\n  verification:")
    print(f"    top{TOP_N} avg ret_fwd_300     = {top_mean*1e4:.2f} bps")
    print(f"    bottom{TOP_N} avg ret_fwd_300  = {bottom_mean*1e4:.2f} bps")
    print(f"    calculated ls_bps             = {calc_ls:.2f}")
    print(f"    reported   ls_bps             = {ls_bps:.2f}")
    print(f"    difference                    = {abs(calc_ls - ls_bps):.2f} bps"
          f"  (should be small — rounding from full 20-group average)")

    # 6. save
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    result[show_cols].to_csv(OUT_CSV, index=False)
    print(f"\nsaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
