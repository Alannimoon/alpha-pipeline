"""
RSRS —— 阻力支撑相对强度因子（Resistance Support Relative Strength）。

定义
----
第一步：计算五档量加权买卖价格
  bid_w = Σ(BidPrice_i × BidVol_i) / Σ(BidVol_i)   i=1..5
  ask_w = Σ(AskPrice_i × AskVol_i) / Σ(AskVol_i)   i=1..5
  有效：CanUseFiveLevelBook=True 且 bid_w / ask_w 均有限

第二步：rolling 线性回归（窗口 W）
  ask_w = α + β × bid_w
  rsrs(t) = β（斜率）

  β 用滚动统计量直接向量化：
    β = (n×Σxy - Σx×Σy) / (n×Σx² - (Σx)²)
  窗口内至少需要 2 个有效 tick。

有效性条件
----------
  window_ok：过去 W×20 个 tick 中 CanUseFiveLevelBook=False 的比例 < 10%

附加输出
--------
  has_limit(t)：过去 W×20 个 tick 内是否出现过涨跌停（实际恒为 False）

窗口
----
  [30, 45, 60, 75, 105] 分钟
"""

import numpy as np
import pandas as pd

from ._core import (
    is_limit_tick, window_valid_mask, rolling_any, TICKS_PER_MIN,
    ASK_PRICE_COLS, ASK_VOL_COLS, BID_PRICE_COLS, BID_VOL_COLS,
)

WINDOWS_MIN       = [30, 45, 60, 75, 105]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：rsrs_30m, rsrs_30m_has_limit, rsrs_45m, ...
    """
    can_use_l5 = df["CanUseFiveLevelBook"].to_numpy(bool)
    limit      = is_limit_tick(df)

    # ── 五档量加权买/卖价格 ───────────────────────────────────────────────────
    bidp = df[BID_PRICE_COLS].to_numpy(np.float64)
    askp = df[ASK_PRICE_COLS].to_numpy(np.float64)
    bidv = df[BID_VOL_COLS].to_numpy(np.float64)
    askv = df[ASK_VOL_COLS].to_numpy(np.float64)

    bid_num = np.nansum(bidp * bidv, axis=1)
    bid_den = np.nansum(bidv,        axis=1)
    ask_num = np.nansum(askp * askv, axis=1)
    ask_den = np.nansum(askv,        axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        bid_w = np.where(bid_den > EPS, bid_num / bid_den, np.nan)
        ask_w = np.where(ask_den > EPS, ask_num / ask_den, np.nan)

    # 仅 CanUseFiveLevelBook=True 且双侧均有限时有效
    valid = can_use_l5 & np.isfinite(bid_w) & np.isfinite(ask_w)
    x = np.where(valid, bid_w, np.nan)
    y = np.where(valid, ask_w, np.nan)

    xs = pd.Series(x)
    ys = pd.Series(y)

    out = {}

    for m in WINDOWS_MIN:
        w      = m * TICKS_PER_MIN
        w_ok   = window_valid_mask(can_use_l5, w, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, w), False)

        # ── rolling 回归统计量（向量化）────────────────────────────────────────
        roll_x  = xs.rolling(w, min_periods=2)
        roll_y  = ys.rolling(w, min_periods=2)
        roll_xy = (xs * ys).rolling(w, min_periods=2)
        roll_x2 = (xs * xs).rolling(w, min_periods=2)

        n_valid = xs.notna().rolling(w, min_periods=2).sum().to_numpy()
        sum_x   = roll_x.sum().to_numpy()
        sum_y   = roll_y.sum().to_numpy()
        sum_xy  = roll_xy.sum().to_numpy()
        sum_x2  = roll_x2.sum().to_numpy()

        denom = n_valid * sum_x2 - sum_x ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(
                np.abs(denom) > EPS,
                (n_valid * sum_xy - sum_x * sum_y) / denom,
                np.nan,
            )

        out[f"rsrs_{m}m"]           = np.where(w_ok, beta, np.nan)
        out[f"rsrs_{m}m_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
