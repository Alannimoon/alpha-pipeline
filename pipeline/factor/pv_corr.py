"""
PvCorr —— 价量相关性因子（Price-Volume Correlation）。

定义（基础窗口 w ticks）
------------------------
第一步：每 tick 算简单收益率（lag = w）
  r_price(t) = P(t) / P(t-w) - 1      有效：CanUsePrice(t) & CanUsePrice(t-w)
  r_vol(t)   = V(t) / V(t-w) - 1      同上，且 V(t-w) > 0

第二步：对 r_price / r_vol 各自做 rolling z-score（窗口 2w）
  z_price(t) = (r_price - mean(r_price, 2w)) / std(r_price, 2w, ddof=0)
  z_vol(t)   = (r_vol   - mean(r_vol,   2w)) / std(r_vol,   2w, ddof=0)
  raw(t)     = z_price(t) * sign(z_vol(t))

  含义：价涨量增或价跌量减 → raw 为正；方向相反 → raw 为负。

第三步：对 raw 做 rolling mean（窗口 w）
  pv_corr(t) = mean(raw, w)    mask: CanUsePrice & isfinite(raw)

有效性条件
----------
  window_ok：过去 4×w 个 tick 中 CanUsePrice=False 的比例 < 10%
  （涵盖完整数据链：1w 算原始收益 + 2w 做标准化 + 1w 做平滑）

附加输出
--------
  has_limit(t)：过去 4×w 个 tick 内是否出现过涨跌停

窗口
----
  [100, 200, 300] ticks
"""

import numpy as np
import pandas as pd

from ._core import is_limit_tick, window_valid_mask, rolling_mean_masked, rolling_any

WINDOWS_TICK      = [100, 200, 300]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：pv_corr_100t, pv_corr_100t_has_limit, pv_corr_200t, ...
    """
    can_use = df["CanUsePrice"].to_numpy(bool)
    price   = df["Price"].to_numpy(np.float64)
    cumvol  = df["CumVolume"].to_numpy(np.float64)
    limit   = is_limit_tick(df)
    n       = len(df)

    out = {}

    for w in WINDOWS_TICK:
        w_ok   = window_valid_mask(can_use, 4 * w, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, 4 * w), False)

        # ── 第一步：lag-w 简单收益率 ─────────────────────────────────────────
        can_lag = np.zeros(n, dtype=bool)
        can_lag[w:] = can_use[:-w]

        price_lag = np.full(n, np.nan)
        vol_lag   = np.full(n, np.nan)
        price_lag[w:] = price[:-w]
        vol_lag[w:]   = cumvol[:-w]

        r_price_valid = can_use & can_lag & (price_lag > EPS)
        r_vol_valid   = r_price_valid & (vol_lag > EPS)

        r_price = np.where(r_price_valid, price   / price_lag - 1, np.nan)
        r_vol   = np.where(r_vol_valid,   cumvol  / vol_lag   - 1, np.nan)

        # ── 第二步：rolling z-score（窗口 2w，ddof=0）────────────────────────
        rp_s = pd.Series(r_price)
        rv_s = pd.Series(r_vol)

        rp_roll = rp_s.where(np.isfinite(r_price)).rolling(2 * w, min_periods=2 * w)
        rv_roll = rv_s.where(np.isfinite(r_vol)  ).rolling(2 * w, min_periods=2 * w)

        rp_mean = rp_roll.mean().to_numpy()
        rp_std  = rp_roll.std(ddof=0).to_numpy()
        rv_mean = rv_roll.mean().to_numpy()
        rv_std  = rv_roll.std(ddof=0).to_numpy()

        with np.errstate(invalid="ignore", divide="ignore"):
            z_price = (r_price - rp_mean) / rp_std
            z_vol   = (r_vol   - rv_mean) / rv_std

        z_price = np.where(np.isfinite(z_price), z_price, np.nan)
        z_vol   = np.where(np.isfinite(z_vol),   z_vol,   np.nan)

        raw = z_price * np.sign(z_vol)

        # ── 第三步：rolling mean（窗口 w）────────────────────────────────────
        raw_mask = can_use & np.isfinite(raw)
        val_raw  = rolling_mean_masked(raw, raw_mask, w)
        val      = np.where(w_ok, val_raw, np.nan)

        out[f"pv_corr_{w}t"]           = val
        out[f"pv_corr_{w}t_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
