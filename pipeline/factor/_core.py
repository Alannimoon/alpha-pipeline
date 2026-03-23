"""
因子计算共享工具。

包含：
- 数据加载（合并 base + returns）
- 窗口有效性检查（基于 CanUsePrice 比例）
- 滚动均值（跳过无效 tick）
- 滚动 OR（窗口内是否存在涨跌停）
- IsLimitTick 判断
"""

import numpy as np
import pandas as pd

# ── 列名常量 ──────────────────────────────────────────────────────────────────

ASK_PRICE_COLS = [f"AskPrice{i}" for i in range(1, 6)]
ASK_VOL_COLS   = [f"AskVolume{i}" for i in range(1, 6)]
BID_PRICE_COLS = [f"BidPrice{i}" for i in range(1, 6)]
BID_VOL_COLS   = [f"BidVolume{i}" for i in range(1, 6)]

_BASE_COLS = [
    "Date", "SampleTime", "SecurityID", "Market",
    "Price", "CumVolume",
    "CanUsePrice", "CanUseDoubleSideBook", "CanUseFiveLevelBook",
    *ASK_PRICE_COLS, *ASK_VOL_COLS, *BID_PRICE_COLS, *BID_VOL_COLS,
]

TICKS_PER_MIN = 20   # 3 秒/tick


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_data(base_path: str, horizons: list[int] = None) -> pd.DataFrame:
    """
    读取单只股票的 base 文件，内联计算前向收益率，返回完整 DataFrame。

    收益率定义：ret_fwd_{h} = P(t+h)/P(t) - 1
    仅当 CanUsePrice(t) 和 CanUsePrice(t+h) 均为 True 时有效，否则为 NaN。
    """
    if horizons is None:
        horizons = [100, 200, 300]

    df = pd.read_csv(
        base_path,
        dtype={"Date": str, "SecurityID": str, "SampleTime": str},
        usecols=lambda c: c in _BASE_COLS,
    )

    num_cols = [
        "Price", "CumVolume",
        *ASK_PRICE_COLS, *ASK_VOL_COLS, *BID_PRICE_COLS, *BID_VOL_COLS,
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("CanUsePrice", "CanUseDoubleSideBook", "CanUseFiveLevelBook"):
        df[c] = df[c].astype(bool)

    # 内联计算前向收益率（逻辑与原 returns.py 完全相同）
    price   = df["Price"]
    can_use = df["CanUsePrice"]
    for h in horizons:
        fut_price   = price.shift(-h)
        fut_can_use = can_use.shift(-h).fillna(False)
        valid = can_use & fut_can_use
        df[f"ret_fwd_{h}"] = np.where(valid, fut_price / price - 1, np.nan)

    return df


# ── 逐 tick 工具 ──────────────────────────────────────────────────────────────

def is_limit_tick(df: pd.DataFrame) -> np.ndarray:
    """
    判断每个 tick 是否为涨跌停：
      CanUsePrice=True（有效价格）且 CanUseDoubleSideBook=False（单边盘口）
    返回 bool 数组。
    """
    return df["CanUsePrice"].to_numpy(bool) & ~df["CanUseDoubleSideBook"].to_numpy(bool)


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


def rolling_any(bool_arr: np.ndarray, window: int) -> np.ndarray:
    """
    滚动 OR：窗口内任意一个 tick 为 True 则返回 True。

    min_periods=window：历史不足时返回 False（与因子有效性对齐）。
    """
    s = pd.Series(bool_arr.astype(np.float64)).rolling(window, min_periods=window).max().to_numpy()
    # max=1.0 → True；max=0.0 → False；NaN（历史不足）→ False
    return s == 1.0
