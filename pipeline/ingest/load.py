"""
分钟线数据加载模块。

读取 data_min/ 下的 {secid}.csv 文件（OHLCV 格式，GBK 编码），
按日期拆分后写入 result/min_base/{date}/{secid}.csv。

该步骤合并了快照流水线的 sample + clean + base 三个阶段：
  - 分钟线已是固定 1 分钟间隔，无需重采样
  - 成交量为 0 的分钟保留，不自动删除
  - 价格直接使用收盘价（Close），无盘口数据

输出列
------
Date, SampleTime (HH:MM:00 格式), SecurityID, Market,
Open, High, Low, Close, Volume, Amount,
Price (= Close), CumVolume (当日累计成交量),
CanUsePrice (Close > 0), CanUseDoubleSideBook (= CanUsePrice),
CanUseFiveLevelBook (= False, 无五档盘口)

说明
----
CanUseDoubleSideBook 设为与 CanUsePrice 相同，
是因为因子层使用 is_limit_tick = CanUsePrice & ~CanUseDoubleSideBook 标记涨跌停。
分钟线无涨跌停概念，将两者设为相同可保证 is_limit_tick 始终返回 False，
从而 has_limit 不污染因子值。
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 列名映射（中文 → 英文）
_COL_MAP = {
    "日期": "Date",
    "时间": "Time",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
    "成交额": "Amount",
}

# 时段边界（与快照流水线保持一致，SampleTime 格式为 HH:MM:00）
_AM_START = "09:30:00"
_AM_END   = "11:30:00"  # 不含
_PM_START = "13:00:00"
_PM_END   = "14:57:00"  # 不含


def _parse_market(secid: str) -> str:
    """从证券代码前缀推断交易所。支持纯数字代码和带交易所前缀（如 bj920000）。"""
    s = secid.lower()
    if s.startswith("bj") or s.startswith("nq"):
        return "BJ"
    if s.startswith("sh"):
        return "SH"
    if s.startswith("sz"):
        return "SZ"
    # 纯数字代码
    digits = secid.lstrip("abcdefghijklmnopqrstuvwxyz")
    if digits.startswith(("6", "5")):
        return "SH"
    if digits.startswith(("0", "3")):
        return "SZ"
    if digits.startswith(("8", "4", "9")):
        return "BJ"
    return "UNK"


def _process_one_stock(in_path: str, out_root: str) -> list[dict]:
    """
    读取单只股票的分钟线 CSV，按日期拆分并写出 base 文件。
    返回各日汇总信息列表。
    """
    secid  = os.path.splitext(os.path.basename(in_path))[0]
    market = _parse_market(secid)

    df = pd.read_csv(in_path, encoding="gbk", dtype={"日期": str, "时间": str})
    df = df.rename(columns=_COL_MAP)

    for c in ("Open", "High", "Low", "Close", "Volume", "Amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # SampleTime: "HH:MM" → "HH:MM:00"
    df["SampleTime"] = df["Time"].str.strip().apply(
        lambda t: t if len(t) == 8 else t + ":00"
    )
    df["SecurityID"] = secid
    df["Market"]     = market

    # 时段过滤（保留 AM + PM，排除午休和收盘后）
    st    = df["SampleTime"]
    in_am = (st >= _AM_START) & (st < _AM_END)
    in_pm = (st >= _PM_START) & (st < _PM_END)
    df    = df[in_am | in_pm].copy()

    if df.empty:
        return []

    # 掩码
    close_valid = df["Close"].notna() & (df["Close"] > 0)
    df["CanUsePrice"]          = close_valid
    df["CanUseDoubleSideBook"] = close_valid   # 见模块说明：保证 is_limit_tick=False
    df["CanUseFiveLevelBook"]  = False
    df["Price"]                = df["Close"]

    results = []
    for date, day_df in df.groupby("Date"):
        day_df = day_df.copy()
        day_df["CumVolume"] = day_df["Volume"].cumsum()

        out_cols = [
            "Date", "SampleTime", "SecurityID", "Market",
            "Open", "High", "Low", "Close", "Volume", "Amount",
            "Price", "CumVolume",
            "CanUsePrice", "CanUseDoubleSideBook", "CanUseFiveLevelBook",
        ]
        out = day_df[[c for c in out_cols if c in day_df.columns]]

        date_str = str(date).replace("-", "")
        out_dir  = os.path.join(out_root, date_str)
        out_path = os.path.join(out_dir, f"{secid}.csv")
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(out_path, index=False)

        results.append({
            "Date":        date_str,
            "SecurityID":  secid,
            "Status":      "OK",
            "Rows":        len(out),
            "CanUseTicks": int(day_df["CanUsePrice"].sum()),
        })

    return results


def _worker(args) -> list[dict]:
    in_path, out_root = args
    secid = os.path.splitext(os.path.basename(in_path))[0]
    try:
        return _process_one_stock(in_path, out_root)
    except Exception as e:
        return [{"SecurityID": secid, "Status": f"FAIL: {e}"}]


def run_load(
    data_root: str,
    base_root: str,
    max_workers: int | None = None,
):
    """
    批量加载分钟线数据，生成 base 文件。

    Parameters
    ----------
    data_root   : 分钟线原始数据目录（每只股票一个 CSV 文件）
    base_root   : 输出根目录（结果写入 base_root/{date}/{secid}.csv）
    max_workers : 并行进程数；None 表示使用 CPU 核数
    """
    os.makedirs(base_root, exist_ok=True)

    stock_files = sorted(f for f in os.listdir(data_root) if f.endswith(".csv"))
    if not stock_files:
        raise FileNotFoundError(f"data_min 目录下无 CSV 文件：{data_root}")

    tasks = [(os.path.join(data_root, f), base_root) for f in stock_files]

    all_results: list[dict] = []
    if max_workers == 1:
        iter_ = tqdm(tasks, desc="load_min") if tqdm else tasks
        for t in iter_:
            all_results.extend(_worker(t))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs  = [pool.submit(_worker, t) for t in tasks]
            iter_ = tqdm(as_completed(futs), total=len(futs), desc="load_min") \
                    if tqdm else as_completed(futs)
            for f in iter_:
                all_results.extend(f.result())

    summary_path = os.path.join(base_root, "_summary.csv")
    pd.DataFrame(all_results).sort_values(["Date", "SecurityID"]).reset_index(drop=True)\
      .to_csv(summary_path, index=False)
    print(f"分钟线 base 生成完成，汇总：{summary_path}")
