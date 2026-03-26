"""
Mom（分钟线版）—— 价格动量因子。

与快照版完全相同的逻辑，窗口单位改为 1 min/tick。
TICKS_PER_MIN = 1，所以窗口 tick 数 = 分钟数。
"""

import numpy as np
import pandas as pd

from ._core import is_limit_tick

TICKS_PER_MIN = 1
WINDOWS_MIN   = [5, 10, 20, 30, 45, 60, 90]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    can_use = df["CanUsePrice"].to_numpy(bool)
    log_p   = np.where(can_use, np.log(df["Price"].to_numpy(np.float64)), np.nan)
    limit   = is_limit_tick(df)

    out = {}

    for m in WINDOWS_MIN:
        w = m * TICKS_PER_MIN

        can_use_lag = np.empty(len(can_use), dtype=bool)
        can_use_lag[:w] = False
        can_use_lag[w:] = can_use[:-w]

        log_p_lag = np.empty(len(log_p), dtype=np.float64)
        log_p_lag[:w] = np.nan
        log_p_lag[w:] = log_p[:-w]

        valid = can_use & can_use_lag
        val   = np.where(valid, log_p - log_p_lag, np.nan)

        limit_lag = np.zeros(len(limit), dtype=bool)
        limit_lag[w:] = limit[:-w]
        has_limit = (limit | limit_lag) & valid

        out[f"mom_{m}m"]           = val
        out[f"mom_{m}m_has_limit"] = has_limit

    return pd.DataFrame(out, index=df.index)
