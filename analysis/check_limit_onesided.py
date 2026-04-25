"""
验证深交所数据中：
  1. 判断条件等价性：(TradeLike & 单边缺失) 与 (LastPrice == HighLimitPrice/LowLimitPrice) 是否等价
  2. 价格正确性：涨停时 BidPrice1 == HighLimitPrice，跌停时 AskPrice1 == LowLimitPrice

直接读原始 ZIP 文件（mdl_6_28_0.csv.zip），不经过 extract/sample/clean 流程。
"""

import os
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── 参数 ─────────────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT = "/home/fund/data"
DEFAULT_A500_CSV  = os.path.join(os.path.dirname(__file__), "..", "config", "a500.csv")
DEFAULT_OUT       = "result/eval/analysis/check_limit_onesided.csv"
DEFAULT_N_DAYS    = None   # None = 全量扫描
CHUNKSIZE         = 200_000
ENCODING          = "gb18030"
TOL               = 1e-3   # 价格比较容差（元，仅用于 BidPrice1/AskPrice1 vs 限价的数值校验）


# ── 工具 ─────────────────────────────────────────────────────────────────────

def load_sz_codes(a500_csv: str) -> set[str]:
    df = pd.read_csv(a500_csv)
    col = "SecurityID" if "SecurityID" in df.columns else "code"
    codes = df[col].astype(str).str.zfill(6)
    return {c for c in codes if not c.startswith("6")}   # 深交所：非6开头


def is_trade_like_sz(phase: pd.Series) -> pd.Series:
    """深交所 TradingPhaseCode：第一位 T 且第二位非 1。"""
    s = phase.astype(str).str.strip()
    return s.str[0].eq("T") & ~s.str[1].eq("1")


def norm6(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return "".join(ch for ch in s if ch.isdigit()).zfill(6)


# ── 顶层 worker（可被 ProcessPoolExecutor pickle）────────────────────────────

def _day_worker(args_tuple):
    zip_path, sz_codes, day = args_tuple
    res = process_day(zip_path, sz_codes)
    if res:
        res["Date"] = day
    return day, res


# ── 单日处理 ──────────────────────────────────────────────────────────────────

NEED_COLS = [
    "SecurityID", "TradingPhaseCode",
    "LastPrice",
    "HighLimitPrice", "LowLimitPrice",
    "BidPrice1", "BidVolume1",
    "AskPrice1", "AskVolume1",
]


def process_day(zip_path: str, sz_codes: set[str]) -> dict:
    """
    返回单日统计结果 dict：
      涨停侧（HighLimit 有效）：
        up_by_onesided    : 我们判断为涨停的 tick 数（TradeLike & 无卖盘）
        up_by_lastprice   : 交易所涨停 tick 数（LastPrice == HighLimitPrice）
        up_both           : 两者都认为是涨停
        up_ours_only      : 只有我们判断（漏报）
        up_exchange_only  : 只有交易所判断（误报）
        up_price_match    : 我们判断为涨停时 BidPrice1 == HighLimitPrice 的比例
      跌停侧同理（dn_*）
    """
    if not os.path.exists(zip_path):
        return {}

    chunks = []
    with zipfile.ZipFile(zip_path) as zf:
        inner = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(inner) as f:
            for ck in pd.read_csv(
                f, chunksize=CHUNKSIZE, encoding=ENCODING,
                encoding_errors="replace", skipinitialspace=True,
                index_col=False,
            ):
                ck["SecurityID"] = ck["SecurityID"].apply(norm6)
                ck = ck[ck["SecurityID"].isin(sz_codes)]
                if ck.empty:
                    continue
                miss = [c for c in NEED_COLS if c not in ck.columns]
                if miss:
                    raise ValueError(f"ZIP 缺少列: {miss}")
                chunks.append(ck[NEED_COLS].copy())

    if not chunks:
        return {}

    df = pd.concat(chunks, ignore_index=True)

    # 数值转换
    for c in ["LastPrice", "HighLimitPrice", "LowLimitPrice",
              "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    trade_like = is_trade_like_sz(df["TradingPhaseCode"])

    # 盘口存在性
    bid_exists = df["BidPrice1"].gt(0) & df["BidVolume1"].gt(0) & df["BidPrice1"].notna()
    ask_exists = df["AskPrice1"].gt(0) & df["AskVolume1"].gt(0) & df["AskPrice1"].notna()

    # 有效涨跌停限价（排除无限制的哨兵值）
    hi_valid = df["HighLimitPrice"].lt(999999999) & df["HighLimitPrice"].notna()
    lo_valid = df["LowLimitPrice"].gt(0.01)       & df["LowLimitPrice"].notna()

    tl = trade_like   # 简写

    # ── 涨停侧 ────────────────────────────────────────────────────────────────
    # up_ours：完全复现 base.py 逻辑，不加 hi_valid（base.py 里没有这个条件）
    up_ours     = tl & ~ask_exists & bid_exists
    # up_exchange：ground truth，需要 hi_valid 排除无涨停限制品种
    up_exchange = tl & hi_valid & df["LastPrice"].eq(df["HighLimitPrice"])

    up_both          = (up_ours & up_exchange).sum()
    up_ours_only     = (up_ours & ~up_exchange).sum()
    up_exchange_only = (~up_ours & up_exchange).sum()
    up_total_ours    = up_ours.sum()

    # 价格正确性：在我们判断涨停的 tick 上，BidPrice1 是否 == HighLimitPrice
    # 仅对 hi_valid 的 tick 有意义（无涨停限制的品种没有 HighLimitPrice 可比）
    up_ours_hi = up_ours & hi_valid
    up_price_ok = ((df.loc[up_ours_hi, "BidPrice1"] - df.loc[up_ours_hi, "HighLimitPrice"]).abs() <= TOL).sum() \
                  if up_ours_hi.any() else np.nan

    # ── 跌停侧 ────────────────────────────────────────────────────────────────
    dn_ours     = tl & ~bid_exists & ask_exists
    dn_exchange = tl & lo_valid & df["LastPrice"].eq(df["LowLimitPrice"])

    dn_both          = (dn_ours & dn_exchange).sum()
    dn_ours_only     = (dn_ours & ~dn_exchange).sum()
    dn_exchange_only = (~dn_ours & dn_exchange).sum()
    dn_total_ours    = dn_ours.sum()

    dn_ours_lo  = dn_ours & lo_valid
    dn_price_ok = ((df.loc[dn_ours_lo, "AskPrice1"] - df.loc[dn_ours_lo, "LowLimitPrice"]).abs() <= TOL).sum() \
                  if dn_ours_lo.any() else np.nan

    return {
        "trade_like_ticks":   int(tl.sum()),
        "up_ours":            int(up_total_ours),
        "up_exchange":        int(up_exchange.sum()),
        "up_both":            int(up_both),
        "up_ours_only":       int(up_ours_only),
        "up_exchange_only":   int(up_exchange_only),
        "up_price_match":     int(up_price_ok) if not np.isnan(up_price_ok) else 0,
        "dn_ours":            int(dn_total_ours),
        "dn_exchange":        int(dn_exchange.sum()),
        "dn_both":            int(dn_both),
        "dn_ours_only":       int(dn_ours_only),
        "dn_exchange_only":   int(dn_exchange_only),
        "dn_price_match":     int(dn_price_ok) if not np.isnan(dn_price_ok) else 0,
    }


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",     default=DEFAULT_DATA_ROOT)
    ap.add_argument("--universe-csv",  default=DEFAULT_A500_CSV,
                    help="股票池 CSV（含 SecurityID 列），默认 config/a500.csv")
    ap.add_argument("--out",           default=DEFAULT_OUT)
    ap.add_argument("--n-days",        type=int, default=DEFAULT_N_DAYS,
                    help="只跑前 N 天（调试用），不指定则全量")
    ap.add_argument("--start-date",    default=None, help="起始日期（含），如 20250102")
    ap.add_argument("--end-date",      default=None, help="截止日期（含），如 20260131")
    ap.add_argument("--workers",       type=int, default=4, help="并行进程数")
    args = ap.parse_args()

    sz_codes = load_sz_codes(args.universe_csv)
    print(f"深交所股票 {len(sz_codes)} 只")

    days = sorted(
        d for d in os.listdir(args.data_root)
        if len(d) == 8 and d.isdigit()
        and os.path.isdir(os.path.join(args.data_root, d))
    )
    if args.start_date:
        days = [d for d in days if d >= args.start_date]
    if args.end_date:
        days = [d for d in days if d <= args.end_date]
    if args.n_days:
        days = days[:args.n_days]
    print(f"扫描 {len(days)} 天: {days[0]} ~ {days[-1]}")

    rows = []
    pbar = tqdm(total=len(days), desc="scanning") if tqdm else None
    tasks = [(os.path.join(args.data_root, d, "mdl_6_28_0.csv.zip"), sz_codes, d) for d in days]

    if args.workers == 1:
        for t in tasks:
            day, res = _day_worker(t)
            if res:
                rows.append(res)
            if pbar:
                pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_day_worker, t): t[2] for t in tasks}
            for fut in as_completed(futs):
                try:
                    day, res = fut.result()
                    if res:
                        rows.append(res)
                except Exception as e:
                    print(f"  {futs[fut]} FAIL: {e}")
                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()

    if not rows:
        print("无数据")
        return

    df = pd.DataFrame(rows)

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    total = df.sum(numeric_only=True)
    print("\n===== 汇总 =====")
    for side, label in [("up", "涨停"), ("dn", "跌停")]:
        ours      = int(total[f"{side}_ours"])
        exchange  = int(total[f"{side}_exchange"])
        both      = int(total[f"{side}_both"])
        ours_only = int(total[f"{side}_ours_only"])
        exch_only = int(total[f"{side}_exchange_only"])
        pm        = int(total[f"{side}_price_match"])

        cond_ok   = "100%" if ours_only == 0 and exch_only == 0 else \
                    f"误报={ours_only} 漏报={exch_only}"
        price_pct = f"{100*pm/ours:.1f}%" if ours > 0 else "N/A"

        print(f"[{label}] 我们={ours} 交易所={exchange} 条件吻合={cond_ok}  价格吻合={price_pct}({pm}/{ours})")

    # ── 保存明细 ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n明细已保存: {args.out}")


if __name__ == "__main__":
    main()
