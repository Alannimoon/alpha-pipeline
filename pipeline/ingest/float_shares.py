"""
Prepare float shares lookup table for feature computation.

Step 1 (auto): if data/equ_free_shares.parquet is missing, pull from MySQL:
  host=192.168.1.11  port=3306  user=datayesread  db=datayes
  table: equ_free_shares  columns: TICKER_SYMBOL, CHANGE_DATE, FREE_SHARES

Step 2: asof-fill sparse data into a daily lookup table per pool.

Reads:
  data/equ_free_shares.parquet       -- sparse raw data (auto-fetched if absent)
  result/base/                       -- to derive A500 trading dates
  result/test/base/                  -- to derive vol_top100 trading dates
  config/a500.csv
  config/vol_top100.csv

Writes:
  data/equ_free_shares.parquet       -- sparse raw data from MySQL (if fetched)
  data/float_shares_filled.parquet   -- (pool, SecurityID, Date, FREE_SHARES)
  data/float_shares_stats.csv        -- change frequency per stock per pool
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MYSQL_CONFIG = dict(
    host="192.168.1.11",
    port=3306,
    user="datayesread",
    password="33#7MNSqq",
    database="datayes",
)


# ── 路径 ──────────────────────────────────────────────────────────────────────

SPARSE_PATH  = os.path.join(ROOT, "data", "equ_free_shares.parquet")
BASE_ROOT    = os.path.join(ROOT, "result", "base")
TEST_BASE    = os.path.join(ROOT, "result", "test", "base")
A500_CSV     = os.path.join(ROOT, "config", "a500.csv")
VOL100_CSV   = os.path.join(ROOT, "config", "vol_top100.csv")
OUT_FILLED   = os.path.join(ROOT, "data", "float_shares_filled.parquet")
OUT_STATS    = os.path.join(ROOT, "data", "float_shares_stats.csv")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def scan_dates(base_dir: str) -> list[str]:
    """扫描 base 目录，返回排好序的交易日列表（YYYYMMDD 字符串）。"""
    dates = []
    for f in os.listdir(base_dir):
        stem, ext = os.path.splitext(f)
        if ext == ".parquet" and stem.isdigit() and len(stem) == 8:
            dates.append(stem)
    return sorted(dates)


def load_stocks(csv_path: str) -> list[str]:
    return pd.read_csv(csv_path)["SecurityID"].astype(str).str.zfill(6).tolist()


def asof_fill(
    sparse: pd.DataFrame,
    stocks: list[str],
    dates: list[str],
    pool: str,
) -> pd.DataFrame:
    """
    对 (stocks × dates) 做 asof 填充，返回 (pool, SecurityID, Date, FREE_SHARES)。
    sparse 列：TICKER_SYMBOL, CHANGE_DATE (datetime64), FREE_SHARES
    """
    date_index = pd.to_datetime(dates, format="%Y%m%d")
    rows = []

    for sid in stocks:
        sub = sparse[sparse["TICKER_SYMBOL"] == sid].sort_values("CHANGE_DATE")
        if sub.empty:
            # 无记录：全部填 NaN
            fs_values = np.full(len(dates), np.nan)
        else:
            # merge_asof：对每个 date 取 <= date 的最新 FREE_SHARES
            target = pd.DataFrame({"Date": date_index})
            merged = pd.merge_asof(
                target,
                sub[["CHANGE_DATE", "FREE_SHARES"]].rename(columns={"CHANGE_DATE": "Date"}),
                on="Date",
            )
            fs_values = merged["FREE_SHARES"].to_numpy(np.float64)

        df_stock = pd.DataFrame({
            "pool":       pool,
            "SecurityID": sid,
            "Date":       dates,
            "FREE_SHARES": fs_values,
        })
        rows.append(df_stock)

    return pd.concat(rows, ignore_index=True)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def fetch_from_mysql() -> pd.DataFrame:
    """从 MySQL 拉取 equ_free_shares 全表，返回 DataFrame 并保存到 SPARSE_PATH。"""
    import pymysql
    print("[1] equ_free_shares.parquet not found, fetching from MySQL...")
    conn = pymysql.connect(**MYSQL_CONFIG)
    sql = "SELECT TICKER_SYMBOL, CHANGE_DATE, FREE_SHARES FROM equ_free_shares"
    df = pd.read_sql(sql, conn)
    conn.close()
    df["CHANGE_DATE"] = pd.to_datetime(df["CHANGE_DATE"])
    df["TICKER_SYMBOL"] = df["TICKER_SYMBOL"].astype(str).str.zfill(6)
    df["FREE_SHARES"] = df["FREE_SHARES"].astype(float)
    os.makedirs(os.path.dirname(SPARSE_PATH), exist_ok=True)
    df.to_parquet(SPARSE_PATH, index=False)
    print(f"    Saved {len(df)} rows -> {SPARSE_PATH}")
    return df


def main():
    # 1. 加载稀疏原始数据（不存在则从 MySQL 提取）
    if os.path.exists(SPARSE_PATH):
        print("[1] Loading sparse data from parquet...")
        sparse = pd.read_parquet(SPARSE_PATH)
        sparse["CHANGE_DATE"] = pd.to_datetime(sparse["CHANGE_DATE"])
        sparse["TICKER_SYMBOL"] = sparse["TICKER_SYMBOL"].astype(str).str.zfill(6)
    else:
        sparse = fetch_from_mysql()
    print(f"    {len(sparse)} rows, {sparse['TICKER_SYMBOL'].nunique()} tickers")

    # 2. 扫描交易日
    print("[2] Scanning trading dates...")
    a500_dates  = scan_dates(BASE_ROOT)
    vol100_dates = scan_dates(TEST_BASE)
    print(f"    A500:    {len(a500_dates)} dates  ({a500_dates[0]} ~ {a500_dates[-1]})")
    print(f"    vtop100: {len(vol100_dates)} dates  ({vol100_dates[0]} ~ {vol100_dates[-1]})")

    # 3. 加载股票池
    a500_stocks  = load_stocks(A500_CSV)
    vol100_stocks = load_stocks(VOL100_CSV)
    print(f"    A500:    {len(a500_stocks)} stocks")
    print(f"    vtop100: {len(vol100_stocks)} stocks")

    # 4. asof 展开
    print("[3] Filling A500 pool...")
    df_a500 = asof_fill(sparse, a500_stocks, a500_dates, pool="A500")

    print("[4] Filling vtop100 pool...")
    df_vol100 = asof_fill(sparse, vol100_stocks, vol100_dates, pool="vtop100")

    filled = pd.concat([df_a500, df_vol100], ignore_index=True)

    # NaN 统计
    n_nan = filled["FREE_SHARES"].isna().sum()
    if n_nan > 0:
        print(f"    WARNING: {n_nan} rows have no float shares data (NaN)")
        missing = filled[filled["FREE_SHARES"].isna()][["pool", "SecurityID"]].drop_duplicates()
        print(missing.to_string(index=False))

    print(f"    Total rows: {len(filled)}")

    # 5. 保存 filled
    filled.to_parquet(OUT_FILLED, index=False)
    print(f"[5] Saved filled parquet -> {OUT_FILLED}")

    # 6. 统计变动次数
    print("[6] Computing change stats...")

    def count_changes(stocks, dates, pool):
        d_min = pd.to_datetime(dates[0],  format="%Y%m%d")
        d_max = pd.to_datetime(dates[-1], format="%Y%m%d")
        window = sparse[
            (sparse["TICKER_SYMBOL"].isin(stocks)) &
            (sparse["CHANGE_DATE"] >= d_min) &
            (sparse["CHANGE_DATE"] <= d_max)
        ]
        counts = (
            window.groupby("TICKER_SYMBOL")["CHANGE_DATE"]
            .agg(n_changes="count", change_dates=lambda x: ",".join(x.dt.strftime("%Y-%m-%d").sort_values()))
            .reset_index()
            .rename(columns={"TICKER_SYMBOL": "SecurityID"})
        )
        # 补入变动次数为0的股票
        all_stocks = pd.DataFrame({"SecurityID": stocks})
        counts = all_stocks.merge(counts, on="SecurityID", how="left")
        counts["n_changes"]   = counts["n_changes"].fillna(0).astype(int)
        counts["change_dates"] = counts["change_dates"].fillna("")
        counts.insert(0, "pool", pool)
        return counts

    stats_a500   = count_changes(a500_stocks,   a500_dates,   "A500")
    stats_vol100 = count_changes(vol100_stocks, vol100_dates, "vtop100")
    stats = pd.concat([stats_a500, stats_vol100], ignore_index=True)
    stats = stats.sort_values(["pool", "n_changes"], ascending=[True, False]).reset_index(drop=True)

    stats.to_csv(OUT_STATS, index=False)
    print(f"[7] Saved stats -> {OUT_STATS}")

    # 简要统计输出
    for pool, grp in stats.groupby("pool"):
        vc = grp["n_changes"].value_counts().sort_index()
        print(f"\n    [{pool}] change count distribution:")
        for k, v in vc.items():
            print(f"      {k} changes: {v} stocks")


if __name__ == "__main__":
    main()
