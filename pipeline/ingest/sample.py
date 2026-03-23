"""
重采样模块：将原始快照（UpdateTime 驱动）映射到固定时间网格。

核心逻辑
--------
对每个采样时刻 t，取原始序列中 <= t 的最近一条记录（backward asof merge）。

采样网格
--------
上午 09:30:00 – 11:29:57
下午 13:00:00 – 14:56:57
间隔 3 秒（共 ~2400 个采样点），均不含两端边界 tick。

输出列
------
Date, SampleTime, SecurityID,
PreCloPrice, LastPrice, Turnover, TradVolume/Volume, InstruStatus/TradingPhaseCode,
AskPrice1-5, AskVolume1-5, BidPrice1-5, BidVolume1-5,
UpdateTime（原始 tick 时间），GapSec（原始 tick 与采样时刻的间隔秒数）
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, time

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def build_sampling_grid(day: str, freq: str, am_start: str, am_end: str,
                        pm_start: str, pm_end: str) -> pd.DatetimeIndex:
    """
    生成当天的采样时间网格。
    使用左闭右开区间（inclusive="left"），自动排除两端边界 tick。
    """
    d = datetime.strptime(day, "%Y%m%d").date()

    def to_dt(t_str):
        h, m, s = map(int, t_str.split(":"))
        return datetime.combine(d, time(h, m, s))

    am = pd.date_range(to_dt(am_start), to_dt(am_end), freq=freq, inclusive="left")
    pm = pd.date_range(to_dt(pm_start), to_dt(pm_end), freq=freq, inclusive="left")
    return am.append(pm)


def _parse_update_time(day: str, s) -> "datetime | pd.NaT":
    """把原始 UpdateTime 字符串解析为带日期的 datetime。"""
    s = str(s).strip()
    if ":" not in s:
        return pd.NaT
    d = datetime.strptime(day, "%Y%m%d").date()
    fmt = "%H:%M:%S.%f" if "." in s else "%H:%M:%S"
    try:
        return datetime.combine(d, datetime.strptime(s, fmt).time())
    except ValueError:
        return pd.NaT


# ── 单文件重采样 ──────────────────────────────────────────────────────────────

def resample_one_file(in_path: str, out_path: str, day: str,
                      freq: str, am_start: str, am_end: str,
                      pm_start: str, pm_end: str) -> dict:
    """
    对单只股票的原始快照文件做重采样，结果写入 out_path。
    返回一行 summary dict（无论成功还是失败）。
    """
    secid = os.path.splitext(os.path.basename(in_path))[0]
    base = {"Date": day, "SecurityID": secid}

    df = pd.read_csv(in_path)
    if len(df) == 0:
        return {**base, "RawRows": 0, "SampleRows": 0, "Status": "NO_RAW_DATA"}

    # 解析原始时间
    df["_ts"] = df["UpdateTime"].apply(lambda x: _parse_update_time(day, x))
    df = df.dropna(subset=["_ts"])
    if len(df) == 0:
        return {**base, "RawRows": len(df), "SampleRows": 0, "Status": "NO_VALID_TIME"}

    # 同一时刻保留最后一条，按时间排序
    df = df.sort_values("_ts").drop_duplicates("_ts", keep="last").reset_index(drop=True)

    # 生成采样网格并做 backward asof merge
    grid = pd.DataFrame({"SampleTime": build_sampling_grid(
        day, freq, am_start, am_end, pm_start, pm_end
    )})
    sampled = pd.merge_asof(
        grid,
        df.rename(columns={"_ts": "SourceTime"}),
        left_on="SampleTime",
        right_on="SourceTime",
        direction="backward",
    )

    # GapSec：原始 tick 距采样时刻的间隔（越大说明数据越"陈旧"）
    sampled["GapSec"] = (sampled["SampleTime"] - sampled["SourceTime"]).dt.total_seconds()

    # 整理输出列顺序
    date_str = datetime.strptime(day, "%Y%m%d").strftime("%Y-%m-%d")
    sampled["Date"]       = date_str
    sampled["SampleTime"] = sampled["SampleTime"].dt.strftime("%H:%M:%S")
    sampled["SecurityID"] = sampled["SecurityID"].astype(str).str.zfill(6)

    vol_col    = "TradVolume" if "TradVolume" in sampled.columns else \
                 ("Volume"    if "Volume"    in sampled.columns else None)
    status_col = "InstruStatus"    if "InstruStatus"    in sampled.columns else \
                 ("TradingPhaseCode" if "TradingPhaseCode" in sampled.columns else None)

    ordered = ["Date", "SampleTime", "SecurityID", "PreCloPrice", "LastPrice", "Turnover"]
    if vol_col:    ordered.append(vol_col)
    if status_col: ordered.append(status_col)
    ordered += [
        "AskPrice1",  "AskPrice2",  "AskPrice3",  "AskPrice4",  "AskPrice5",
        "AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5",
        "BidPrice1",  "BidPrice2",  "BidPrice3",  "BidPrice4",  "BidPrice5",
        "BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5",
        "UpdateTime", "GapSec",
    ]
    ordered = [c for c in ordered if c in sampled.columns]
    out_df = sampled[ordered]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    max_gap = sampled["GapSec"].max()
    return {
        **base,
        "RawRows":    len(df),
        "SampleRows": len(out_df),
        "MaxGapSec":  round(max_gap, 1) if pd.notna(max_gap) else None,
        "MeanGapSec": round(sampled["GapSec"].mean(), 1),
        "PctGapGt60": round((sampled["GapSec"] > 60).mean(), 4),
        "Status":     "OK" if (pd.notna(max_gap) and max_gap <= 60) else "LARGE_GAP",
    }


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def _worker(args):
    """ProcessPoolExecutor 的 worker，捕获异常确保不中断整体进度。"""
    in_path, out_path, day, freq, am_start, am_end, pm_start, pm_end = args
    secid = os.path.splitext(os.path.basename(in_path))[0]
    try:
        return resample_one_file(in_path, out_path, day,
                                 freq, am_start, am_end, pm_start, pm_end)
    except Exception as e:
        return {"Date": day, "SecurityID": secid, "Status": f"FAIL: {e}"}


def run_sample(raw_root: str, sampled_root: str, dates: list[str] | None = None,
               freq: str = "3s",
               am_start: str = "09:30:00", am_end: str = "11:30:00",
               pm_start: str = "13:00:00", pm_end: str = "14:57:00",
               max_workers: int | None = None):
    """
    批量重采样。

    Parameters
    ----------
    raw_root     : 原始数据根目录，子目录按日期命名（如 20250102/）
    sampled_root : 输出根目录
    dates        : 指定日期列表；为 None 时自动扫描 raw_root 下所有日期目录
    freq         : 采样频率，默认 "3s"
    max_workers  : 并行进程数；None 表示使用 CPU 核数
    """
    os.makedirs(sampled_root, exist_ok=True)

    if dates is None:
        dates = sorted(
            d for d in os.listdir(raw_root)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(raw_root, d))
        )

    all_tasks = []
    for day in dates:
        in_dir  = os.path.join(raw_root, day)
        out_dir = os.path.join(sampled_root, day)
        stock_files = sorted(
            f for f in os.listdir(in_dir)
            if f.endswith(".csv") and not f.startswith("_")
        )
        if not stock_files:
            continue
        for f in stock_files:
            all_tasks.append((
                os.path.join(in_dir, f),
                os.path.join(out_dir, f),
                day, freq, am_start, am_end, pm_start, pm_end,
            ))

    if max_workers == 1:
        iter_ = tqdm(all_tasks, desc="sample") if tqdm else all_tasks
        results = [_worker(t) for t in iter_]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs  = [pool.submit(_worker, t) for t in all_tasks]
            iter_ = tqdm(as_completed(futs), total=len(futs), desc="sample") \
                    if tqdm else as_completed(futs)
            results = [f.result() for f in iter_]

    day_map = defaultdict(list)
    for r in results:
        day_map[r["Date"]].append(r)
    for day, day_results in sorted(day_map.items()):
        out_dir = os.path.join(sampled_root, day)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(day_results).sort_values("SecurityID").reset_index(drop=True) \
          .to_csv(os.path.join(out_dir, "_summary.csv"), index=False)

    summary_path = os.path.join(sampled_root, "_summary.csv")
    pd.DataFrame(results).sort_values(["Date", "SecurityID"]).reset_index(drop=True) \
      .to_csv(summary_path, index=False)
    print(f"重采样完成，汇总：{summary_path}")
