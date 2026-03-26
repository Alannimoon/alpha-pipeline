"""
AccMom —— 加速度动量因子（Acceleration Momentum）。

定义
----
  acc_mom = ln(P(t) / P(t-short)) - ln(P(t-short) / P(t-long))
          = 近期动量 - 远期动量

有效性条件
----------
  三个 anchor tick 的 CanUsePrice 均须为 True：
    CanUsePrice(t) & CanUsePrice(t-short) & CanUsePrice(t-long)
  否则为 NaN。

附加输出
--------
  has_limit(t)：三个 anchor tick 中是否有涨跌停
    limit(t) | limit(t-short) | limit(t-long)，仅当 valid 时有意义。

窗口（short_tick, long_tick）对
--------------------------------
  (25,50), (50,100), (100,200), (150,300), (200,400),
  (300,600), (400,800), (500,1000), (600,1200)
"""

import numpy as np
import pandas as pd

from ._core import (
    TICKS_PER_MIN,
    is_limit_tick,
)

PAIRS = [
    (25, 50), (50, 100), (100, 200), (150, 300), (200, 400),
    (300, 600), (400, 800), (500, 1000), (600, 1200),
]
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：acc_mom_25_50t, acc_mom_25_50t_has_limit, acc_mom_50_100t, ...
    """
    can_use = df["CanUsePrice"].to_numpy(bool)
    limit   = is_limit_tick(df)
    log_p   = np.where(can_use, np.log(df["Price"].to_numpy(np.float64)), np.nan)
    n       = len(df)

    out = {}

    for short, long in PAIRS:
        # ── anchor tick 有效性 ───────────────────────────────────────────────
        can_short = np.zeros(n, dtype=bool)
        can_long  = np.zeros(n, dtype=bool)
        can_short[short:] = can_use[:-short]
        can_long[long:]   = can_use[:-long]
        valid = can_use & can_short & can_long

        # ── 取 lag 价格 ──────────────────────────────────────────────────────
        log_p_short = np.full(n, np.nan)
        log_p_long  = np.full(n, np.nan)
        log_p_short[short:] = log_p[:-short]
        log_p_long[long:]   = log_p[:-long]

        # ── 因子值 ───────────────────────────────────────────────────────────
        raw = (log_p - log_p_short) - (log_p_short - log_p_long)
        val = np.where(valid, raw, np.nan)

        # ── has_limit ────────────────────────────────────────────────────────
        limit_short = np.zeros(n, dtype=bool)
        limit_long  = np.zeros(n, dtype=bool)
        limit_short[short:] = limit[:-short]
        limit_long[long:]   = limit[:-long]
        has_limit = (limit | limit_short | limit_long) & valid

        col = f"acc_mom_{short}_{long}t"
        out[col]                = val
        out[f"{col}_has_limit"] = has_limit

    return pd.DataFrame(out, index=df.index)
