"""
NegSkew —— 负偏度因子（Negative Skewness）。

定义
----
第一步：每个 tick 计算 1 分钟简单收益率（lag=20 ticks）
  r_1m(t) = P(t) / P(t-20) - 1
  有效条件：CanUsePrice(t) & CanUsePrice(t-20) 均为 True，否则 NaN。

第二步：对过去 W 分钟（W×20 ticks）内的 r_1m 序列做 rolling 偏度，取负
  neg_skew(t) = -skew(r_1m(t-W×20+1 : t))
  仅统计非 NaN 的 r_1m 值，至少需要 3 个有效值。

有效性条件
----------
  window_ok：过去 W×20+20 个 tick 中 CanUsePrice=False 的比例 < 10%
  （多出的 20 覆盖 r_1m 的 lag）

附加输出
--------
  has_limit(t)：过去 W×20+20 个 tick 内是否出现过涨跌停

窗口
----
  [15, 30, 45, 60] 分钟
"""

import numpy as np
import pandas as pd

from ._core import is_limit_tick, window_valid_mask, rolling_any, TICKS_PER_MIN

WINDOWS_MIN       = [15, 30, 45, 60]
MAX_INVALID_RATIO = 0.10
R1M_LAG           = TICKS_PER_MIN          # 1 分钟 = 20 ticks


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：neg_skew_15m, neg_skew_15m_has_limit, neg_skew_30m, ...
    """
    can_use = df["CanUsePrice"].to_numpy(bool)
    price   = df["Price"].to_numpy(np.float64)
    limit   = is_limit_tick(df)
    n       = len(df)

    # ── 第一步：逐 tick 计算 r_1m ─────────────────────────────────────────────
    can_use_lag = np.zeros(n, dtype=bool)
    price_lag   = np.full(n, np.nan)
    can_use_lag[R1M_LAG:] = can_use[:-R1M_LAG]
    price_lag[R1M_LAG:]   = price[:-R1M_LAG]

    r_1m_valid = can_use & can_use_lag
    r_1m = np.where(r_1m_valid, price / price_lag - 1, np.nan)

    # r_1m 为 NaN 时无需额外掩码，rolling.skew() 自动跳过 NaN
    r_1m_series = pd.Series(r_1m)

    out = {}

    for m in WINDOWS_MIN:
        w        = m * TICKS_PER_MIN           # skew 的滚动窗口（ticks）
        w_ok_len = w + R1M_LAG                  # window_ok / has_limit 的检查跨度

        # ── window_ok ─────────────────────────────────────────────────────────
        w_ok = window_valid_mask(can_use, w_ok_len, MAX_INVALID_RATIO)

        # ── 第二步：rolling 偏度，min_periods=3 ──────────────────────────────
        skew = r_1m_series.rolling(w, min_periods=3).skew().to_numpy()
        val  = np.where(w_ok, -skew, np.nan)

        # ── has_limit ─────────────────────────────────────────────────────────
        has_limit = np.where(w_ok, rolling_any(limit, w_ok_len), False)

        out[f"neg_skew_{m}m"]           = val
        out[f"neg_skew_{m}m_has_limit"] = has_limit.astype(bool)

    return pd.DataFrame(out, index=df.index)
