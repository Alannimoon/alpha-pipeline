"""
PvCorr（分钟线版）—— 价量相关性因子。

与快照版逻辑相同；窗口调整为分钟级别。
快照版 WINDOWS_TICK = [100, 200, 300]（= 5/10/15 min at 3s）。
分钟版 WINDOWS_TICK = [5, 10, 15]（= 5/10/15 min at 1min）。
"""

import numpy as np
import pandas as pd

from ._core import is_limit_tick, window_valid_mask, rolling_mean_masked, rolling_any

WINDOWS_TICK      = [5, 10, 15]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    can_use = df["CanUsePrice"].to_numpy(bool)
    price   = df["Price"].to_numpy(np.float64)
    cumvol  = df["CumVolume"].to_numpy(np.float64)
    limit   = is_limit_tick(df)
    n       = len(df)

    out = {}

    for w in WINDOWS_TICK:
        w_ok   = window_valid_mask(can_use, 4 * w, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, 4 * w), False)

        can_lag = np.zeros(n, dtype=bool)
        can_lag[w:] = can_use[:-w]

        price_lag = np.full(n, np.nan)
        vol_lag   = np.full(n, np.nan)
        price_lag[w:] = price[:-w]
        vol_lag[w:]   = cumvol[:-w]

        r_price_valid = can_use & can_lag & (price_lag > EPS)
        r_vol_valid   = r_price_valid & (vol_lag > EPS)

        with np.errstate(divide="ignore", invalid="ignore"):
            r_price = np.where(r_price_valid, price  / price_lag - 1, np.nan)
            r_vol   = np.where(r_vol_valid,   cumvol / vol_lag   - 1, np.nan)

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

        raw_mask = can_use & np.isfinite(raw)
        val_raw  = rolling_mean_masked(raw, raw_mask, w)
        val      = np.where(w_ok, val_raw, np.nan)

        out[f"pv_corr_{w}t"]           = val
        out[f"pv_corr_{w}t_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
