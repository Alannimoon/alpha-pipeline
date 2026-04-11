"""
Base数据生成模块。

功能
----
在清洗后数据的基础上，为每个 tick 生成：
  - Price / PriceType：当前时刻的"可用价格"
  - CanUsePrice：该 tick 的价格是否可用
  - CanUseDoubleSideBook：双边一档盘口是否有效（可算 wmid）
  - CanUseFiveLevelBook：五档盘口是否完整有效（含梯度校验）

价格定义
--------
TradeLike 状态下：
  双边有效（bid 和 ask 均存在且 bid ≤ ask）
      → NORMAL_WMID：反向加权中间价
        = (BidPrice1 × AskVolume1 + AskPrice1 × BidVolume1) / (BidVolume1 + AskVolume1)

  仅卖单（无 bid，有 ask）
      → LIMIT_DOWN_ONE_SIDED：Price = AskPrice1

  仅买单（无 ask，有 bid）
      → LIMIT_UP_ONE_SIDED：Price = BidPrice1

  其他（双边均缺、spread 倒挂等）
      → INVALID：Price = NaN

非 TradeLike 状态 → Price = NaN（不赋 PriceType）

掩码定义
--------
CanUsePrice          = TradeLike AND Price 有效（非 NaN）
CanUseDoubleSideBook = TradeLike AND 双边一档有效（bid & ask 均存在且 bid ≤ ask）
CanUseFiveLevelBook  = TradeLike AND 五档完整 AND ask/bid 梯度单调 AND bid ≤ ask

输出列
------
Date, SampleTime, SecurityID, Market,
PreCloPrice, OpenPrice,
CumVolume, GapSec,
PriceType, Price,
CanUsePrice, CanUseDoubleSideBook, CanUseFiveLevelBook,
BidPrice1-5, AskPrice1-5, BidVolume1-5, AskVolume1-5
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ── 列名常量 ──────────────────────────────────────────────────────────────────

_ASK_P = [f"AskPrice{i}"  for i in range(1, 6)]
_BID_P = [f"BidPrice{i}"  for i in range(1, 6)]
_ASK_V = [f"AskVolume{i}" for i in range(1, 6)]
_BID_V = [f"BidVolume{i}" for i in range(1, 6)]
_5L    = _ASK_P + _ASK_V + _BID_P + _BID_V

_NEEDED = [
    "Date", "SampleTime", "SecurityID",
    "PreCloPrice", "OpenPrice", "Volume", "TradVolume",
    "InstruStatus", "TradingPhaseCode",
    "GapSec",
    "BidPrice1", "BidPrice2", "BidPrice3", "BidPrice4", "BidPrice5",
    "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5",
    "BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5",
    "AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5",
]


# ── 시장 판별 ─────────────────────────────────────────────────────────────────

def _detect_market(df: pd.DataFrame) -> tuple[str, str | None]:
    if "InstruStatus" in df.columns:
        return "SH", "InstruStatus"
    if "TradingPhaseCode" in df.columns:
        return "SZ", "TradingPhaseCode"
    return "UNK", None


def _build_trade_like(series: pd.Series, market: str) -> pd.Series:
    """True 表示该 tick 处于连续竞价（TradeLike）状态。"""
    s = series.astype(str).str.strip()
    if market == "SH":
        return s.eq("TRADE")
    elif market == "SZ":
        # 第二位不为 "1"（非全天停牌）且第一位为 "T"（连续竞价）
        flag  = s.str[1:2]
        phase = s.str[0:1]
        return phase.eq("T") & ~flag.eq("1")
    return pd.Series(False, index=series.index)


# ── 盘口有效性 ─────────────────────────────────────────────────────────────────

def _build_book_masks(df: pd.DataFrame):
    """
    返回：
      double_side_valid  — 双边一档有效（bid & ask 均存在且 bid ≤ ask）
      five_level_valid   — 五档完整有效（含梯度校验）
      bid_exists         — 买一存在
      ask_exists         — 卖一存在
    """
    bp1, ap1 = df["BidPrice1"], df["AskPrice1"]
    bv1, av1 = df["BidVolume1"], df["AskVolume1"]

    bid_exists = bp1.notna() & bv1.notna() & bp1.gt(0) & bv1.gt(0)
    ask_exists = ap1.notna() & av1.notna() & ap1.gt(0) & av1.gt(0)
    spread_ok  = bp1.le(ap1)                              # bid ≤ ask

    double_side_valid = bid_exists & ask_exists & spread_ok

    five_complete = df[_5L].notna().all(axis=1)
    ask_ladder = (
        df["AskPrice1"].le(df["AskPrice2"]) &
        df["AskPrice2"].le(df["AskPrice3"]) &
        df["AskPrice3"].le(df["AskPrice4"]) &
        df["AskPrice4"].le(df["AskPrice5"])
    )
    bid_ladder = (
        df["BidPrice1"].ge(df["BidPrice2"]) &
        df["BidPrice2"].ge(df["BidPrice3"]) &
        df["BidPrice3"].ge(df["BidPrice4"]) &
        df["BidPrice4"].ge(df["BidPrice5"])
    )
    five_level_valid = double_side_valid & five_complete & ask_ladder & bid_ladder

    return double_side_valid, five_level_valid, bid_exists, ask_exists


# ── 价格计算 ──────────────────────────────────────────────────────────────────

def _compute_wmid(df: pd.DataFrame) -> pd.Series:
    """反向加权中间价（仅对 double_side_valid 的 tick 有意义）。"""
    denom = df["BidVolume1"] + df["AskVolume1"]
    return (df["BidPrice1"] * df["AskVolume1"] + df["AskPrice1"] * df["BidVolume1"]) / denom


# ── 单文件处理 ────────────────────────────────────────────────────────────────

def process_one_file(in_path: str) -> tuple[dict, pd.DataFrame]:
    """
    处理单只股票日，返回 (summary_dict, output_df)。
    output_df 由调用方按天汇总后写入 base/{date}.parquet。
    """
    secid = os.path.splitext(os.path.basename(in_path))[0]

    df = pd.read_parquet(in_path)

    # parquet 已保留数值类型，保留此步作为防御性保障
    num_cols = ["PreCloPrice", "OpenPrice"] + _5L
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    market, status_col = _detect_market(df)
    if status_col is None:
        raise ValueError("状态码列缺失")

    trade_like = _build_trade_like(df[status_col], market)

    double_side_valid, five_level_valid, bid_exists, ask_exists = _build_book_masks(df)

    # ── 价格类型与价格 ────────────────────────────────────────────────────────
    # 优先级：NORMAL_WMID > LIMIT_DOWN > LIMIT_UP > INVALID
    is_normal   = trade_like & double_side_valid
    is_lim_down = trade_like & ~bid_exists & ask_exists   # 仅卖单 → 跌停
    is_lim_up   = trade_like & ~ask_exists & bid_exists   # 仅买单 → 涨停

    price_type = np.where(
        is_normal,   "NORMAL_WMID",
        np.where(
            is_lim_down, "LIMIT_DOWN_ONE_SIDED",
            np.where(
                is_lim_up, "LIMIT_UP_ONE_SIDED",
                "INVALID"
            )
        )
    )

    price = np.where(
        is_normal,   _compute_wmid(df),
        np.where(
            is_lim_down, df["AskPrice1"],
            np.where(
                is_lim_up, df["BidPrice1"],
                np.nan
            )
        )
    )

    # ── 掩码 ──────────────────────────────────────────────────────────────────
    can_use_price     = trade_like & pd.notna(price)
    can_use_double    = trade_like & double_side_valid
    can_use_five      = trade_like & five_level_valid

    # ── 成交量列（沪深字段名不同）────────────────────────────────────────────
    vol_col = "TradVolume" if "TradVolume" in df.columns else \
              ("Volume"    if "Volume"    in df.columns else None)

    # ── 组装输出 ──────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "Date":                df["Date"],
        "SampleTime":          df["SampleTime"],
        "SecurityID":          df["SecurityID"],
        "Market":              market,
        "PreCloPrice":         df["PreCloPrice"] if "PreCloPrice" in df.columns else np.nan,
        "OpenPrice":           df["OpenPrice"]   if "OpenPrice"   in df.columns else np.nan,
        "CumVolume":           df[vol_col] if vol_col else np.nan,
        "GapSec":              df["GapSec"] if "GapSec" in df.columns else np.nan,
        "PriceType":           price_type,
        "Price":               price,
        "CanUsePrice":         can_use_price,
        "CanUseDoubleSideBook": can_use_double,
        "CanUseFiveLevelBook":  can_use_five,
    })

    # 保留五档盘口列（供因子计算使用）
    for c in _BID_P + _ASK_P + _BID_V + _ASK_V:
        if c in df.columns:
            out[c] = df[c].values

    # 前向收益率（与 _core.load_data 逻辑一致）
    price_s = pd.Series(price, index=df.index, dtype=float)
    for h in (100, 200, 300):
        fut_price   = price_s.shift(-h)
        fut_can_use = can_use_price.shift(-h).to_numpy(dtype=bool, na_value=False)
        valid       = can_use_price & fut_can_use
        out[f"ret_fwd_{h}"] = np.where(valid, fut_price / price_s - 1, np.nan)

    summary = {
        "SecurityID":          secid,
        "Rows":                len(out),
        "NormalWmidTicks":     int(is_normal.sum()),
        "LimitDownTicks":      int(is_lim_down.sum()),
        "LimitUpTicks":        int(is_lim_up.sum()),
        "CanUsePriceTicks":    int(can_use_price.sum()),
        "CanUseDoubleTicks":   int(can_use_double.sum()),
        "CanUseFiveTicks":     int(can_use_five.sum()),
    }
    return summary, out


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def _worker(args) -> tuple[dict, pd.DataFrame | None]:
    in_path, day = args
    secid = os.path.splitext(os.path.basename(in_path))[0]
    try:
        summary, df = process_one_file(in_path)
        return {"Date": day, "Status": "OK", **summary}, df
    except Exception as e:
        return {"Date": day, "SecurityID": secid, "Status": f"FAIL: {e}"}, None


def run_base(
    cleaned_root: str,
    base_root: str,
    dates: list[str] | None = None,
    max_workers: int | None = None,
):
    """
    批量生成 base 数据。

    Parameters
    ----------
    cleaned_root : 清洗后数据根目录
    base_root    : 输出根目录
    dates        : 指定日期列表；None 时自动扫描
    max_workers  : 并行进程数
    """
    os.makedirs(base_root, exist_ok=True)

    if dates is None:
        dates = sorted(
            d for d in os.listdir(cleaned_root)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(cleaned_root, d))
        )

    # 按天组织任务，避免一次性提交全量 futures
    tasks_by_day: dict[str, list] = {}
    for day in dates:
        in_dir = os.path.join(cleaned_root, day)
        stock_files = sorted(
            f for f in os.listdir(in_dir)
            if f.endswith(".parquet") and not f.startswith("_")
        )
        if not stock_files:
            continue
        tasks_by_day[day] = [
            (os.path.join(in_dir, f), day)
            for f in stock_files
        ]

    total       = sum(len(v) for v in tasks_by_day.values())
    all_results: list[dict] = []

    def _consolidate(day: str, pairs: list[tuple[dict, pd.DataFrame | None]]) -> None:
        """合并单日所有股票结果 → 写 base/{day}.parquet，收集 summary。"""
        summaries = [s for s, _ in pairs]
        dfs       = [df for _, df in pairs if df is not None]
        if dfs:
            combined = (pd.concat(dfs, ignore_index=True)
                        .sort_values(["SampleTime", "SecurityID"])
                        .reset_index(drop=True))
            combined.to_parquet(os.path.join(base_root, f"{day}.parquet"), index=False)
        all_results.extend(summaries)

    if max_workers == 1:
        pbar = tqdm(total=total, desc="base") if tqdm else None
        for day, day_tasks in tasks_by_day.items():
            pairs = []
            for t in day_tasks:
                pairs.append(_worker(t))
                if pbar: pbar.update(1)
            _consolidate(day, pairs)
        if pbar: pbar.close()
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            pbar = tqdm(total=total, desc="base") if tqdm else None
            for day, day_tasks in tasks_by_day.items():
                futs = [pool.submit(_worker, t) for t in day_tasks]
                pairs = []
                for f in as_completed(futs):
                    pairs.append(f.result())
                    if pbar: pbar.update(1)
                _consolidate(day, pairs)
            if pbar: pbar.close()

    summary_path = os.path.join(base_root, "_summary.csv")
    pd.DataFrame(all_results).sort_values(["Date", "SecurityID"]).reset_index(drop=True) \
      .to_csv(summary_path, index=False)
    print(f"Base 生成完成，汇总：{summary_path}")
