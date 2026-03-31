"""
因子计算共享工具。

包含：
- 窗口有效性检查（基于 CanUsePrice 比例）
- 滚动均值（跳过无效 tick）

数据加载由 compute.py 统一负责：每天读一次 base/{date}.parquet，
按 SecurityID 分组后将单股 DataFrame 传入各因子 compute() 函数。
"""

import numpy as np
import pandas as pd

# ── 列名常量（供各因子模块 import 使用）────────────────────────────────────────

ASK_PRICE_COLS = [f"AskPrice{i}" for i in range(1, 6)]
ASK_VOL_COLS   = [f"AskVolume{i}" for i in range(1, 6)]
BID_PRICE_COLS = [f"BidPrice{i}" for i in range(1, 6)]
BID_VOL_COLS   = [f"BidVolume{i}" for i in range(1, 6)]

TICKS_PER_MIN = 20   # 3 秒/tick


# ── 窗口级工具 ────────────────────────────────────────────────────────────────

def window_valid_mask(can_use_price: np.ndarray, window: int,
                      max_invalid_ratio: float = 0.10) -> np.ndarray:
    """
    滚动窗口有效性检查。

    对每个 tick t，检查过去 window 个 tick（含 t）中 CanUsePrice=False 的比例。
    比例 > max_invalid_ratio → False（因子记 NaN）；否则 → True。
    前 window-1 个 tick 因历史不足，始终返回 False。

    Parameters
    ----------
    can_use_price     : CanUsePrice bool 数组
    window            : 窗口大小（tick 数）
    max_invalid_ratio : 无效 tick 比例上限，默认 0.10（即 10%）
    """
    invalid = (~can_use_price).astype(np.float64)
    # min_periods=window 保证历史不足时返回 NaN，NaN / window → NaN，NaN <= 0.10 → False
    invalid_sum = pd.Series(invalid).rolling(window, min_periods=window).sum().to_numpy()
    return invalid_sum / window <= max_invalid_ratio


def rolling_mean_masked(values: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    """
    滚动均值，仅对 mask=True 的 tick 取均值。

    min_periods=1：窗口内只要有一个有效 tick 就计算均值。
    窗口有效性（是否满足 max_invalid_ratio）由调用方通过 window_valid_mask 控制，
    不在此函数内判断。
    """
    return pd.Series(values).where(mask).rolling(window, min_periods=1).mean().to_numpy()


