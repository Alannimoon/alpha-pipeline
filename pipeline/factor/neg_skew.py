"""
NegSkew（分钟线版）—— 负偏度因子。

与快照版逻辑相同；TICKS_PER_MIN = 1，R1M_LAG = 1（1 tick = 1 分钟收益率）。
"""

import numpy as np
import pandas as pd

from ._core import is_limit_tick, window_valid_mask, rolling_any

TICKS_PER_MIN     = 1
WINDOWS_MIN       = [15, 30, 45, 60]
MAX_INVALID_RATIO = 0.10
R1M_LAG           = TICKS_PER_MIN   # 1 分钟 = 1 tick


def compute(df: pd.DataFrame) -> pd.DataFrame:
    can_use = df["CanUsePrice"].to_numpy(bool)
    price   = df["Price"].to_numpy(np.float64)
    limit   = is_limit_tick(df)
    n       = len(df)

    can_use_lag = np.zeros(n, dtype=bool)
    price_lag   = np.full(n, np.nan)
    can_use_lag[R1M_LAG:] = can_use[:-R1M_LAG]
    price_lag[R1M_LAG:]   = price[:-R1M_LAG]

    r_1m_valid = can_use & can_use_lag
    r_1m = np.where(r_1m_valid, price / price_lag - 1, np.nan)
    r_1m_series = pd.Series(r_1m)

    out = {}

    for m in WINDOWS_MIN:
        w        = m * TICKS_PER_MIN
        w_ok_len = w + R1M_LAG

        w_ok = window_valid_mask(can_use, w_ok_len, MAX_INVALID_RATIO)

        skew = r_1m_series.rolling(w).skew().to_numpy()
        val  = np.where(w_ok, -skew, np.nan)

        has_limit = np.where(w_ok, rolling_any(limit, w_ok_len), False)

        out[f"neg_skew_{m}m"]           = val
        out[f"neg_skew_{m}m_has_limit"] = has_limit.astype(bool)

    return pd.DataFrame(out, index=df.index)
