"""
AmpSlice —— 振幅分层因子（Amplitude Slice，align 版）。

定义
----
将过去 lookback 个 tick 按每 10 tick（30 秒）分为若干组，对每组计算：
  avg_price = 组内价格均值
  amplitude = 组内最高价 / 组内最低价 - 1

按 avg_price 升序排列各组，取最高 20% 和最低 20% 的组：
  amp_slice(t) = mean(高价格组的振幅) - mean(低价格组的振幅)

捕捉含义：高价位时段的波动幅度与低价位时段的差异。

有效性条件
----------
  window_ok：过去 lookback 个 tick 中 CanUsePrice=False 的比例 < 10%
  组内无效 tick（CanUsePrice=False）价格置 NaN，由 nanmean/nanmax/nanmin 跳过；
  若某组最低价 ≤ 0 则整个 tick 的因子值为 NaN。

窗口（lookback ticks）
----------------------
  [300, 600, 900, 1200, 1500]
  对应 30/60/90/120/150 组，每组 10 tick（30 秒）

性能说明
--------
  预先对 GROUP_LEN=10 的小滑窗计算 mean/max/min，供所有 lookback 共用；
  通过花式索引组合出各组统计量，避免复制 (m × lookback) 大数组。
  内存从 O(m × lookback) 降为 O(n × GROUP_LEN)，速度提升约 10-50x。
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ._core import window_valid_mask

LOOKBACK_TICKS     = [300, 600, 900, 1200, 1500]
GROUP_INTERVAL_SEC = 30
GROUP_LEN          = GROUP_INTERVAL_SEC // 3   # 10 ticks / 组
MAX_INVALID_RATIO  = 0.10
QUANTILE_FRAC      = 0.20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：amp_slice_300t, amp_slice_600t, ...
    """
    can_use = df["CanUsePrice"].to_numpy(bool)
    price   = df["Price"].to_numpy(np.float64)
    n       = len(df)

    # CanUsePrice=False 的 tick 价格置 NaN，后续 nan 系列函数跳过
    masked_price = np.where(can_use, price, np.nan)

    # ── 预计算 GROUP_LEN 滑窗统计（所有 lookback 共用，计算一次）──────────────
    # wins_gl[j] = masked_price[j : j+GROUP_LEN]，形状 (n-GROUP_LEN+1, GROUP_LEN)
    if n >= GROUP_LEN:
        wins_gl = sliding_window_view(masked_price, GROUP_LEN)
        with np.errstate(all="ignore"):
            wins_mean = np.nanmean(wins_gl, axis=1)   # (n-GROUP_LEN+1,)
            wins_hi   = np.nanmax(wins_gl,  axis=1)
            wins_lo   = np.nanmin(wins_gl,  axis=1)
    else:
        wins_mean = wins_hi = wins_lo = np.empty(0)

    out = {}

    for lookback in LOOKBACK_TICKS:
        group_count = lookback // GROUP_LEN                            # 30/60/90/120/150
        k           = max(1, int(np.floor(group_count * QUANTILE_FRAC)))

        w_ok = window_valid_mask(can_use, lookback, MAX_INVALID_RATIO)
        val  = np.full(n, np.nan)

        if n < lookback or wins_mean.size == 0:
            out[f"amp_slice_{lookback}t"] = val
            continue

        active     = w_ok[lookback - 1:]
        active_idx = np.where(active)[0]

        if len(active_idx) == 0:
            out[f"amp_slice_{lookback}t"] = val
            continue

        # active_idx[i] = 该窗口在 masked_price 中的起始偏移，对应 tick = active_idx[i]+lookback-1
        # 第 g 组的 GROUP_LEN 窗口起始 = active_idx[i] + g*GROUP_LEN
        # 即 wins_* 的下标 = active_idx[i] + g*GROUP_LEN
        go    = np.arange(group_count) * GROUP_LEN      # (G,) 组偏移
        g_idx = active_idx[:, None] + go[None, :]       # (m, G)

        avg_prices = wins_mean[g_idx]   # (m, G)
        hi         = wins_hi[g_idx]     # (m, G)
        lo         = wins_lo[g_idx]     # (m, G)

        lo_ok = (lo > 0).all(axis=1)   # (m,)：任一组 lo≤0 则整 tick 无效
        amps  = hi / lo - 1.0          # (m, G)

        sorted_idx = np.argsort(avg_prices, axis=1)       # (m, G)
        rows_i     = np.arange(len(active_idx))[:, None]

        low_amp  = amps[rows_i, sorted_idx[:, :k]].mean(axis=1)   # (m,)
        high_amp = amps[rows_i, sorted_idx[:, -k:]].mean(axis=1)  # (m,)

        result = np.where(lo_ok, high_amp - low_amp, np.nan)

        # sliding_window index i → tick position i+lookback-1
        val[active_idx + lookback - 1] = result

        out[f"amp_slice_{lookback}t"] = val

    return pd.DataFrame(out, index=df.index)
