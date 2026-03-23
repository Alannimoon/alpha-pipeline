"""
BAP —— 买卖盘压力因子（Bid-Ask Pressure）。

定义
----
第一步：逐 tick 计算原始值（仅 CanUseDoubleSideBook=True 时有效）
  BAP_raw(t) = (BidVolume1(t) - AskVolume1(t)) / (BidVolume1(t) + AskVolume1(t))

第二步：滚动均值
  bap_{W}m(t) = mean(BAP_raw(i), i ∈ [t-W+1, t], BAP_raw(i) 非 NaN)

有效性条件
----------
  过去 W 个 tick 中 BAP_raw 非 NaN 的比例 < 90% → 因子记 NaN
  （BAP_raw 非 NaN 当且仅当 CanUseDoubleSideBook=True 且分母 > 0，
   故直接用 BAP_raw 的有效占比即可，无需单独看 CanUseDoubleSideBook）

附加输出
--------
  bap_{W}m_has_limit(t)：过去 W 个 tick 内是否存在涨跌停 tick
    （IsLimitTick = CanUsePrice=True & CanUseDoubleSideBook=False）
    对 BAP 而言 has_limit 恒为 False（涨跌停 tick 的 BAP_raw 本身就是 NaN，
    不会参与均值计算），保留此列仅为与其他因子输出格式对齐。

窗口
----
  [15, 30, 45, 60, 75] 分钟，即 [300, 600, 900, 1200, 1500] tick
"""

import numpy as np
import pandas as pd

from ._core import (
    TICKS_PER_MIN,
    is_limit_tick,
    window_valid_mask,
    rolling_mean_masked,
    rolling_any,
)

WINDOWS_MIN       = [15, 30, 45, 60, 75]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：bap_15m, bap_15m_has_limit, bap_30m, bap_30m_has_limit, ...
    """
    can_use_l1    = df["CanUseDoubleSideBook"].to_numpy(bool)
    limit_tick    = is_limit_tick(df)

    bidv1 = df["BidVolume1"].to_numpy(np.float64)
    askv1 = df["AskVolume1"].to_numpy(np.float64)

    # ── 第一步：BAP_raw（仅双边盘口有效时计算）────────────────────────────────
    denom   = bidv1 + askv1
    bap_raw = np.where(
        can_use_l1 & (denom > EPS),
        (bidv1 - askv1) / denom,
        np.nan,
    )

    out = {}

    for m in WINDOWS_MIN:
        w = m * TICKS_PER_MIN

        # ── 第二步：窗口有效性（基于 bap_raw 自身的有效占比）────────────────────
        w_valid = window_valid_mask(np.isfinite(bap_raw), w, MAX_INVALID_RATIO)

        # ── 第三步：滚动均值（仅对 CanUseDoubleSideBook=True 的 tick）──────────
        val = rolling_mean_masked(bap_raw, can_use_l1, w)
        # 窗口无效 → NaN
        val = np.where(w_valid, val, np.nan)

        # ── 第四步：涨跌停 bool（窗口内是否出现过涨跌停 tick）────────────────
        has_limit = rolling_any(limit_tick, w)
        # 因子本身无效时，has_limit 无意义，置 False
        has_limit = np.where(w_valid, has_limit, False)

        out[f"bap_{m}m"]           = val
        out[f"bap_{m}m_has_limit"] = has_limit.astype(bool)

    return pd.DataFrame(out, index=df.index)
