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

附加输出
--------
  has_limit(t)：过去 lookback 个 tick 内是否出现过涨跌停

窗口（lookback ticks）
----------------------
  [300, 600, 900, 1200, 1500]
  对应 30/60/90/120/150 组，每组 10 tick（30 秒）
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ._core import is_limit_tick, window_valid_mask, rolling_any

LOOKBACK_TICKS     = [300, 600, 900, 1200, 1500]
GROUP_INTERVAL_SEC = 30
GROUP_LEN          = GROUP_INTERVAL_SEC // 3   # 10 ticks / 组
MAX_INVALID_RATIO  = 0.10
QUANTILE_FRAC      = 0.20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：amp_slice_300t, amp_slice_300t_has_limit, amp_slice_600t, ...
    """
    can_use = df["CanUsePrice"].to_numpy(bool)
    price   = df["Price"].to_numpy(np.float64)
    limit   = is_limit_tick(df)
    n       = len(df)

    # CanUsePrice=False 的 tick 价格置 NaN，后续用 nan 系列函数跳过
    masked_price = np.where(can_use, price, np.nan)

    out = {}

    for lookback in LOOKBACK_TICKS:
        group_count = lookback // GROUP_LEN                            # 30/60/90/120/150
        k           = max(1, int(np.floor(group_count * QUANTILE_FRAC)))

        w_ok   = window_valid_mask(can_use, lookback, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, lookback), False)
        val    = np.full(n, np.nan)

        if n < lookback:
            out[f"amp_slice_{lookback}t"]           = val
            out[f"amp_slice_{lookback}t_has_limit"] = hl_arr.astype(bool)
            continue

        # ── 滑动窗口视图（向量化，避免逐 tick 循环）──────────────────────────
        p_win = sliding_window_view(masked_price, lookback)   # (n-lookback+1, lookback)

        # 仅 window_ok 作为外层门控（允许窗口内最多 10% 无效 tick）
        active     = w_ok[lookback - 1:]
        active_idx = np.where(active)[0]

        if len(active_idx) == 0:
            out[f"amp_slice_{lookback}t"]           = val
            out[f"amp_slice_{lookback}t_has_limit"] = hl_arr.astype(bool)
            continue

        # ── 分组计算 ──────────────────────────────────────────────────────────
        p_active = p_win[active_idx].copy()                                   # (m, lookback)
        p_groups = p_active.reshape(len(active_idx), group_count, GROUP_LEN)  # (m, G, 10)

        avg_prices = np.nanmean(p_groups, axis=2)    # (m, G)，跳过 NaN tick
        hi         = np.nanmax(p_groups, axis=2)     # (m, G)
        lo         = np.nanmin(p_groups, axis=2)     # (m, G)

        lo_ok = (lo > 0).all(axis=1)          # (m,)：任一组 lo≤0 则整 tick 无效
        amps  = hi / lo - 1.0                 # (m, G)

        # ── 按均价排序，取最高 / 最低 k 组 ──────────────────────────────────
        sorted_idx = np.argsort(avg_prices, axis=1)        # (m, G)
        rows       = np.arange(len(active_idx))[:, None]

        low_amp  = amps[rows, sorted_idx[:, :k]].mean(axis=1)    # (m,)
        high_amp = amps[rows, sorted_idx[:, -k:]].mean(axis=1)   # (m,)

        result = np.where(lo_ok, high_amp - low_amp, np.nan)

        # ── 写回原始 tick 位置（sliding_window index i → tick i+lookback-1）─
        val[active_idx + lookback - 1] = result

        out[f"amp_slice_{lookback}t"]           = val
        out[f"amp_slice_{lookback}t_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
