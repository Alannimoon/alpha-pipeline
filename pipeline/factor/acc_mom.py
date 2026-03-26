"""
AccMom（分钟线版）—— 加速度动量因子。

与快照版逻辑相同；窗口对调整为分钟级别（1 tick = 1 min）。

快照版 PAIRS（3s tick）：(25,50)...(600,1200)，对应 (1.25,2.5)...(30,60) min。
分钟版 PAIRS（1min tick）：对应相似时间跨度，取整为常用分钟数。
"""

import numpy as np
import pandas as pd

from ._core import is_limit_tick

# (short_tick, long_tick)，单位：分钟（= tick）
PAIRS = [
    (2,  4),
    (5,  10),
    (10, 20),
    (15, 30),
    (20, 40),
    (30, 60),
    (45, 90),
    (60, 120),
]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    can_use = df["CanUsePrice"].to_numpy(bool)
    limit   = is_limit_tick(df)
    log_p   = np.where(can_use, np.log(df["Price"].to_numpy(np.float64)), np.nan)
    n       = len(df)

    out = {}

    for short, long in PAIRS:
        can_short = np.zeros(n, dtype=bool)
        can_long  = np.zeros(n, dtype=bool)
        can_short[short:] = can_use[:-short]
        can_long[long:]   = can_use[:-long]
        valid = can_use & can_short & can_long

        log_p_short = np.full(n, np.nan)
        log_p_long  = np.full(n, np.nan)
        log_p_short[short:] = log_p[:-short]
        log_p_long[long:]   = log_p[:-long]

        raw = (log_p - log_p_short) - (log_p_short - log_p_long)
        val = np.where(valid, raw, np.nan)

        limit_short = np.zeros(n, dtype=bool)
        limit_long  = np.zeros(n, dtype=bool)
        limit_short[short:] = limit[:-short]
        limit_long[long:]   = limit[:-long]
        has_limit = (limit | limit_short | limit_long) & valid

        col = f"acc_mom_{short}_{long}t"
        out[col]                = val
        out[f"{col}_has_limit"] = has_limit

    return pd.DataFrame(out, index=df.index)
