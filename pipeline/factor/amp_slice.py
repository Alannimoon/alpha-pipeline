"""
AmpSlice（分钟线版）—— 振幅分层因子。

与快照版逻辑相同；参数调整为分钟级别：
  - LOOKBACK_TICKS：快照版 [300,600,900,1200,1500] tick = [15,30,45,60,75] min
                    分钟版 [15,30,45,60,75] tick = [15,30,45,60,75] min
  - GROUP_LEN：快照版 10 tick = 30s；分钟版 5 tick = 5 min
  每个 lookback 的分组数 = lookback / GROUP_LEN 须为整数。
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ._core import is_limit_tick, window_valid_mask, rolling_any

LOOKBACK_TICKS    = [15, 30, 45, 60, 75]
GROUP_LEN         = 5       # 5 分钟/组
MAX_INVALID_RATIO = 0.10
QUANTILE_FRAC     = 0.20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    can_use = df["CanUsePrice"].to_numpy(bool)
    price   = df["Price"].to_numpy(np.float64)
    limit   = is_limit_tick(df)
    n       = len(df)

    masked_price = np.where(can_use, price, np.nan)

    out = {}

    for lookback in LOOKBACK_TICKS:
        group_count = lookback // GROUP_LEN
        k           = max(1, int(np.floor(group_count * QUANTILE_FRAC)))

        w_ok   = window_valid_mask(can_use, lookback, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, lookback), False)
        val    = np.full(n, np.nan)

        if n < lookback:
            out[f"amp_slice_{lookback}t"]           = val
            out[f"amp_slice_{lookback}t_has_limit"] = hl_arr.astype(bool)
            continue

        p_win      = sliding_window_view(masked_price, lookback)
        active     = w_ok[lookback - 1:]
        active_idx = np.where(active)[0]

        if len(active_idx) == 0:
            out[f"amp_slice_{lookback}t"]           = val
            out[f"amp_slice_{lookback}t_has_limit"] = hl_arr.astype(bool)
            continue

        p_active = p_win[active_idx].copy()
        p_groups = p_active.reshape(len(active_idx), group_count, GROUP_LEN)

        avg_prices = np.nanmean(p_groups, axis=2)
        hi         = np.nanmax(p_groups, axis=2)
        lo         = np.nanmin(p_groups, axis=2)

        lo_ok = (lo > 0).all(axis=1)
        amps  = hi / lo - 1.0

        sorted_idx = np.argsort(avg_prices, axis=1)
        rows       = np.arange(len(active_idx))[:, None]

        low_amp  = amps[rows, sorted_idx[:, :k]].mean(axis=1)
        high_amp = amps[rows, sorted_idx[:, -k:]].mean(axis=1)

        result = np.where(lo_ok, high_amp - low_amp, np.nan)
        val[active_idx + lookback - 1] = result

        out[f"amp_slice_{lookback}t"]           = val
        out[f"amp_slice_{lookback}t_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
