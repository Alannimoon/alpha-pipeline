"""
OFD —— 委托密度比率因子（Order Flow Density）。

定义
----
第一步：逐 tick 计算原始值（仅 CanUseFiveLevelBook=True 时有效）

  avg_bid_spread = mean(BidPrice_1-BidPrice_2, BidPrice_2-BidPrice_3,
                        BidPrice_3-BidPrice_4, BidPrice_4-BidPrice_5)
  avg_ask_spread = mean(AskPrice_2-AskPrice_1, AskPrice_3-AskPrice_2,
                        AskPrice_4-AskPrice_3, AskPrice_5-AskPrice_4)

  bid_density = Σ BidVol_1:5 / avg_bid_spread   （单位价差内的买单量）
  ask_density = Σ AskVol_1:5 / avg_ask_spread   （单位价差内的卖单量）

  OFD_raw = bid_density / ask_density

第二步：rolling 均值
  ofd_{W}m(t) = mean(OFD_raw, W)   mask: CanUseFiveLevelBook

有效性条件
----------
  过去 W×20 个 tick 中 OFD_raw 非 NaN 的比例 < 90% → 因子记 NaN
  avg_bid/ask_spread ≤ 0 时 OFD_raw = NaN（挂单价格异常或全为同价位）

附加输出
--------
  has_limit(t)：过去 W×20 个 tick 内是否出现过涨跌停（实际恒为 False）

窗口
----
  [15, 30, 45, 60, 75] 分钟
"""

import numpy as np
import pandas as pd

from ._core import (
    is_limit_tick, window_valid_mask, rolling_mean_masked, rolling_any, TICKS_PER_MIN,
    ASK_PRICE_COLS, ASK_VOL_COLS, BID_PRICE_COLS, BID_VOL_COLS,
)

WINDOWS_MIN       = [15, 30, 45, 60, 75]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：ofd_15m, ofd_15m_has_limit, ofd_30m, ...
    """
    can_use_l5 = df["CanUseFiveLevelBook"].to_numpy(bool)
    limit      = is_limit_tick(df)

    bidp = df[BID_PRICE_COLS].to_numpy(np.float64)   # (n, 5)
    askp = df[ASK_PRICE_COLS].to_numpy(np.float64)
    bidv = df[BID_VOL_COLS].to_numpy(np.float64)
    askv = df[ASK_VOL_COLS].to_numpy(np.float64)

    # ── 逐 tick 计算 OFD_raw ──────────────────────────────────────────────────
    # 相邻档位价差（4 个差值取均值）
    avg_bid_spread = np.nanmean(bidp[:, :-1] - bidp[:, 1:], axis=1)   # B1-B2, B2-B3, ...
    avg_ask_spread = np.nanmean(askp[:, 1:]  - askp[:, :-1], axis=1)  # A2-A1, A3-A2, ...

    total_bidv = np.nansum(bidv, axis=1)
    total_askv = np.nansum(askv, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        bid_density = np.where(avg_bid_spread > EPS, total_bidv / avg_bid_spread, np.nan)
        ask_density = np.where(avg_ask_spread > EPS, total_askv / avg_ask_spread, np.nan)
        ofd_raw     = np.where(
            can_use_l5 & np.isfinite(bid_density) & np.isfinite(ask_density) & (ask_density > EPS),
            bid_density / ask_density,
            np.nan,
        )

    out = {}

    for m in WINDOWS_MIN:
        w      = m * TICKS_PER_MIN
        w_ok   = window_valid_mask(np.isfinite(ofd_raw), w, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, w), False)

        val = rolling_mean_masked(ofd_raw, can_use_l5, w)
        val = np.where(w_ok, val, np.nan)

        out[f"ofd_{m}m"]           = val
        out[f"ofd_{m}m_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
