"""
Rigidity —— 价格刚性因子（Price Rigidity）。

定义
----
对窗口内有效价格序列拟合二次多项式：
  P = a + b*t + c*t²
其中 t 为有效 tick 在窗口内的实际相对位置（0-based，跳过无效 tick）。

因子值：
  rigidity = b * R² * ln(1 / (|c| + ε))

含义：b 是线性趋势斜率，R² 衡量拟合质量，ln(1/|c|) 惩罚曲率。
三者合一：方向明确、拟合好、曲率低的趋势才有高值。

有效性条件
----------
  window_ok：过去 W×20 个 tick 中 CanUsePrice=False 的比例 < 10%
  窗口内有效 tick（CanUsePrice=True 且价格有限）须 ≥ 3，否则 NaN。

附加输出
--------
  has_limit(t)：过去 W×20 个 tick 内是否出现过涨跌停

窗口
----
  [10, 20, 30, 45, 60, 90, 105] 分钟

实现说明
--------
  用 numba.njit(cache=True) JIT 编译内层循环：保留逐 tick 遍历的
  cache 友好结构，同时消除 Python 解释开销，达到接近 C 的速度。
  Cramer 法则在循环内直接计算，无需构造矩阵或调用 BLAS。
"""

import math

import numpy as np
import pandas as pd

from ._core import is_limit_tick, window_valid_mask, rolling_any, TICKS_PER_MIN

WINDOWS_MIN       = [10, 20, 30, 45, 60, 90, 105]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-8


def _rigidity_window_impl(price, valid_tick, w_ok, w, eps):
    """
    对单只股票单日数据计算单个窗口大小的 rigidity 序列。

    逐 tick 遍历，每次在窗口内累加统计量，用 Cramer 法则解三元方程组，
    再算 R²，最终得到因子值。全程无大数组，对 CPU cache 友好。
    """
    n   = len(price)
    val = np.empty(n)
    for i in range(n):
        val[i] = np.nan

    for t in range(w - 1, n):
        if not w_ok[t]:
            continue

        start = t - w + 1

        # ── 累加统计量 ────────────────────────────────────────────────────────
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

        # ── Cramer 法则解 A·[c, b, a]ᵀ = B ──────────────────────────────────
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

        # ── 计算 R² ───────────────────────────────────────────────────────────
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
        if r2 != r2:    # isnan
            continue

        val[t] = b_coef * r2 * math.log(1.0 / (abs(c_coef) + eps))

    return val


# numba JIT 编译版本，第一次调用时才初始化，避免主进程 fork 前加载 LLVM
_rigidity_window_compiled = None

def _rigidity_window(price, valid_tick, w_ok, w, eps):
    global _rigidity_window_compiled
    if _rigidity_window_compiled is None:
        import numba
        _rigidity_window_compiled = numba.njit(cache=True)(_rigidity_window_impl)
    return _rigidity_window_compiled(price, valid_tick, w_ok, w, eps)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：rigidity_10m, rigidity_10m_has_limit, rigidity_20m, ...
    """
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
