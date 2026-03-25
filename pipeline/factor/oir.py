"""
OIR —— 加权委托失衡率（Order Imbalance Ratio，五档衰减权重版）。

定义
----
第一步：逐 tick 计算原始值（仅 CanUseFiveLevelBook=True 时有效）
  weights = [1.0, 0.8, 0.6, 0.4, 0.2]
  OIR_raw = (Σ w_i × BidVol_i - Σ w_i × AskVol_i)
           / (Σ w_i × BidVol_i + Σ w_i × AskVol_i)

  与 BAP 的区别：BAP 只用一档，OIR 用五档并对远端档位给予衰减。

第二步：rolling 均值
  oir_{W}m(t) = mean(OIR_raw, W)   mask: CanUseFiveLevelBook

有效性条件
----------
  过去 W×20 个 tick 中 OIR_raw 非 NaN 的比例 < 90% → 因子记 NaN

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
    ASK_VOL_COLS, BID_VOL_COLS,
)

WINDOWS_MIN       = [15, 30, 45, 60, 75]
MAX_INVALID_RATIO = 0.10
OIR_WEIGHTS       = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
EPS               = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：oir_15m, oir_15m_has_limit, oir_30m, ...
    """
    can_use_l5 = df["CanUseFiveLevelBook"].to_numpy(bool)
    limit      = is_limit_tick(df)

    bidv = df[BID_VOL_COLS].to_numpy(np.float64)
    askv = df[ASK_VOL_COLS].to_numpy(np.float64)

    # ── 逐 tick 计算 OIR_raw ──────────────────────────────────────────────────
    wb  = bidv * OIR_WEIGHTS          # (n, 5) 广播
    wa  = askv * OIR_WEIGHTS
    num = np.nansum(wb - wa, axis=1)
    den = np.nansum(wb + wa, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        oir_raw = np.where(
            can_use_l5 & (den > EPS),
            num / den,
            np.nan,
        )

    out = {}

    for m in WINDOWS_MIN:
        w      = m * TICKS_PER_MIN
        w_ok   = window_valid_mask(np.isfinite(oir_raw), w, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, w), False)

        val = rolling_mean_masked(oir_raw, can_use_l5, w)
        val = np.where(w_ok, val, np.nan)

        out[f"oir_{m}m"]           = val
        out[f"oir_{m}m_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
