"""
Rigidity（分钟线版）—— 价格刚性因子。

与快照版逻辑完全相同；TICKS_PER_MIN = 1，窗口直接为分钟数。
快照版 WINDOWS_MIN = [10,20,30,45,60,90,105] × 20 tick/min。
分钟版 WINDOWS_MIN = [10,20,30,45,60,90,105] × 1 tick/min。
"""

import math

import numpy as np
import pandas as pd

from ._core import is_limit_tick, window_valid_mask, rolling_any

TICKS_PER_MIN     = 1
WINDOWS_MIN       = [10, 20, 30, 45, 60, 90, 105]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-8


def _rigidity_window_impl(price, valid_tick, w_ok, w, eps):
    n   = len(price)
    val = np.empty(n)
    for i in range(n):
        val[i] = np.nan

    for t in range(w - 1, n):
        if not w_ok[t]:
            continue

        start = t - w + 1

        sn = 0.0; sx = 0.0; sx2 = 0.0; sx3 = 0.0; sx4 = 0.0
        sy = 0.0; sxy = 0.0; sx2y = 0.0

        for j in range(start, t + 1):
            if valid_tick[j]:
                xi  = float(j - start)
                p   = price[j]
                xi2 = xi * xi
                sn   += 1.0
                sx   += xi
                sx2  += xi2
                sx3  += xi2 * xi
                sx4  += xi2 * xi2
                sy   += p
                sxy  += xi * p
                sx2y += xi2 * p

        if sn < 3.0:
            continue

        det_A = (sx4 * (sx2 * sn  - sx  * sx )
               - sx3 * (sx3 * sn  - sx  * sx2)
               + sx2 * (sx3 * sx  - sx2 * sx2))

        if abs(det_A) <= 1e-20:
            continue

        det_b = (sx4  * (sxy * sn  - sx  * sy )
               - sx2y * (sx3 * sn  - sx  * sx2)
               + sx2  * (sx3 * sy  - sxy * sx2))

        det_c = (sx2y * (sx2 * sn  - sx  * sx )
               - sx3  * (sxy * sn  - sx  * sy )
               + sx2  * (sxy * sx  - sx2 * sy ))

        det_a = (sx4  * (sx2 * sy  - sxy * sx )
               - sx3  * (sx3 * sy  - sxy * sx2)
               + sx2y * (sx3 * sx  - sx2 * sx2))

        b_coef = det_b / det_A
        c_coef = det_c / det_A
        a_coef = det_a / det_A

        y_bar  = sy / sn
        ss_res = 0.0
        ss_tot = 0.0

        for j in range(start, t + 1):
            if valid_tick[j]:
                xi    = float(j - start)
                p     = price[j]
                y_hat = a_coef + b_coef * xi + c_coef * xi * xi
                ss_res += (p - y_hat) ** 2
                ss_tot += (p - y_bar)  ** 2

        if ss_tot < 1e-12:
            continue

        r2 = 1.0 - ss_res / ss_tot
        if r2 != r2:
            continue

        val[t] = b_coef * r2 * math.log(1.0 / (abs(c_coef) + eps))

    return val


_rigidity_window_compiled = None

def _rigidity_window(price, valid_tick, w_ok, w, eps):
    global _rigidity_window_compiled
    if _rigidity_window_compiled is None:
        import numba
        _rigidity_window_compiled = numba.njit(cache=True)(_rigidity_window_impl)
    return _rigidity_window_compiled(price, valid_tick, w_ok, w, eps)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    can_use    = df["CanUsePrice"].to_numpy(bool)
    price      = df["Price"].to_numpy(np.float64)
    limit      = is_limit_tick(df)
    n          = len(df)
    valid_tick = can_use & np.isfinite(price)

    out = {}

    for m in WINDOWS_MIN:
        w      = m * TICKS_PER_MIN
        w_ok   = window_valid_mask(can_use, w, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, w), False)

        if n < w:
            val = np.full(n, np.nan)
        else:
            val = _rigidity_window(price, valid_tick, w_ok, w, EPS)

        out[f"rigidity_{m}m"]           = val
        out[f"rigidity_{m}m_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
