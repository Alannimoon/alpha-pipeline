"""
Mom —— 价格动量因子（Price Momentum）。

定义
----
  mom_raw(t) = ln(P(t)) - ln(P(t-W))

有效性条件
----------
  1. CanUsePrice(t)   = True
  2. CanUsePrice(t-W) = True
  两者同时满足则计算，否则 NaN。
  无窗口质量检查，允许跨午休。

附加输出
--------
  mom_{W}m_has_limit(t)：当前 tick 或 W 个 tick 前是否为涨跌停 tick
    （IsLimitTick = CanUsePrice=True & CanUseDoubleSideBook=False）
    仅检查两个端点，中间 tick 不考虑。
    与 BAP 不同，涨跌停 tick 的 CanUsePrice=True，动量值可以算出，
    但价格受限制，信号失真，has_limit=True 供下游（ts_ic）掩码使用。

窗口
----
  [5, 10, 20, 30, 45, 60, 90] 分钟，即 [100, 200, 400, 600, 900, 1200, 1800] tick
"""

import numpy as np
import pandas as pd

from ._core import TICKS_PER_MIN, is_limit_tick

WINDOWS_MIN = [5, 10, 20, 30, 45, 60, 90]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：mom_5m, mom_5m_has_limit, mom_10m, mom_10m_has_limit, ...
    """
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
