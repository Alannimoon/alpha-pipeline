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

窗口
----
  [10, 20, 30, 45, 60, 90, 105] 分钟

实现说明
--------
  预先建立 9 条前缀和（sn, sja, sj2a, sj3a, sj4a, sy, sjy, sj2y, sy2），
  每条 O(n)，所有窗口尺寸共用。每个窗口用 O(1) 差分取出绝对坐标统计量，
  通过二项展开转换为相对坐标（xi = j − start），再向量化 Cramer 法则解
  三元正规方程，彻底消除原 numba 双重循环的 O(n × w) 开销。
  R² 利用正规方程恒等式 ss_res = sy2 − (a·sy + b·sxy + c·sx2y) 化简，
  无需第二次遍历窗口。
"""

import numpy as np
import pandas as pd

from ._core import window_valid_mask, TICKS_PER_MIN

WINDOWS_MIN       = [10, 20, 30, 45, 60, 90, 105]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-8


def _pfx(arr: np.ndarray) -> np.ndarray:
    """长度 n+1 的前缀和（P[0]=0, P[i]=sum(arr[:i])）。"""
    out = np.empty(len(arr) + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(arr, out=out[1:])
    return out


def _rigidity_window_np(pfxs, w_ok, w, n, eps):
    """
    向量化前缀和版本，替代原 numba 双重循环。

    pfxs : (P_sn, P_sja, P_sj2a, P_sj3a, P_sj4a, P_sy, P_sjy, P_sj2y, P_sy2)
           各长度 n+1，由外层 compute() 一次性计算后传入。
    """
    P_sn, P_sja, P_sj2a, P_sj3a, P_sj4a, P_sy, P_sjy, P_sj2y, P_sy2 = pfxs

    # 窗口端点：t ∈ [w-1, n-1]，长度 n-w+1
    t_arr = np.arange(w - 1, n)
    ep1   = t_arr + 1                       # P 右端（开区间）
    si    = t_arr - w + 1                   # 窗口左端（整数索引）
    s     = si.astype(np.float64)           # 同一数据，浮点算术用

    # O(1) 差分取绝对坐标窗口统计量
    sn   = P_sn[ep1]   - P_sn[si]
    sja  = P_sja[ep1]  - P_sja[si]
    sj2a = P_sj2a[ep1] - P_sj2a[si]
    sj3a = P_sj3a[ep1] - P_sj3a[si]
    sj4a = P_sj4a[ep1] - P_sj4a[si]
    sy   = P_sy[ep1]   - P_sy[si]
    sjy  = P_sjy[ep1]  - P_sjy[si]
    sj2y = P_sj2y[ep1] - P_sj2y[si]
    sy2  = P_sy2[ep1]  - P_sy2[si]

    # 二项展开：绝对坐标 → 相对坐标 xi = j − s
    s2 = s * s;  s3 = s2 * s;  s4 = s3 * s

    sx   = sja  - s * sn
    sx2  = sj2a - 2.0*s*sja  + s2*sn
    sx3  = sj3a - 3.0*s*sj2a + 3.0*s2*sja  - s3*sn
    sx4  = sj4a - 4.0*s*sj3a + 6.0*s2*sj2a - 4.0*s3*sja + s4*sn
    sxy  = sjy  - s * sy
    sx2y = sj2y - 2.0*s*sjy  + s2*sy

    # 向量化 Cramer 法则，解 A·[c,b,a]ᵀ = [sx2y, sxy, sy]
    # A = [[sx4,sx3,sx2],[sx3,sx2,sx],[sx2,sx,sn]]
    det_A = (sx4 * (sx2*sn  - sx *sx )
           - sx3 * (sx3*sn  - sx *sx2)
           + sx2 * (sx3*sx  - sx2*sx2))

    det_b = (sx4  * (sxy*sn  - sx *sy )
           - sx2y * (sx3*sn  - sx *sx2)
           + sx2  * (sx3*sy  - sxy*sx2))

    det_c = (sx2y * (sx2*sn  - sx *sx )
           - sx3  * (sxy*sn  - sx *sy )
           + sx2  * (sxy*sx  - sx2*sy ))

    det_a = (sx4  * (sx2*sy  - sxy*sx )
           - sx3  * (sx3*sy  - sxy*sx2)
           + sx2y * (sx3*sx  - sx2*sx2))

    with np.errstate(invalid="ignore", divide="ignore"):
        inv_det = np.where(np.abs(det_A) > 1e-20, 1.0 / det_A, np.nan)
        b_coef  = det_b * inv_det
        c_coef  = det_c * inv_det
        a_coef  = det_a * inv_det

        # R²：利用正规方程恒等式 ss_res = sy2 − (a·sy + b·sxy + c·sx2y)
        ss_tot = sy2 - sy * sy / sn
        ss_res = sy2 - (a_coef*sy + b_coef*sxy + c_coef*sx2y)
        r2     = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)

        result = b_coef * r2 * np.log(1.0 / (np.abs(c_coef) + eps))

    # 有效性掩码：w_ok & sn≥3 & |det_A|>1e-20 & r2/result 有限
    valid = (w_ok[w - 1:]
             & (sn >= 3.0)
             & (np.abs(det_A) > 1e-20)
             & np.isfinite(r2)
             & np.isfinite(result))
    result = np.where(valid, result, np.nan)

    val = np.full(n, np.nan)
    val[w - 1:] = result
    return val


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：rigidity_10m, rigidity_20m, ...
    """
    can_use    = df["CanUsePrice"].to_numpy(bool)
    price      = df["Price"].to_numpy(np.float64)
    n          = len(df)
    valid_tick = can_use & np.isfinite(price)

    # 预计算 9 条前缀和，所有窗口尺寸共用
    vt = valid_tick.astype(np.float64)
    p  = np.where(valid_tick, price, 0.0)
    j  = np.arange(n, dtype=np.float64)

    pfxs = (
        _pfx(vt),
        _pfx(j * vt),
        _pfx(j * j * vt),
        _pfx(j * j * j * vt),
        _pfx(j * j * j * j * vt),
        _pfx(p),           # p 已将无效 tick 置 0，与 p*vt 等价
        _pfx(j * p),
        _pfx(j * j * p),
        _pfx(p * p),
    )

    out = {}

    for m in WINDOWS_MIN:
        w    = m * TICKS_PER_MIN
        w_ok = window_valid_mask(can_use, w, MAX_INVALID_RATIO)

        if n < w:
            val = np.full(n, np.nan)
        else:
            val = _rigidity_window_np(pfxs, w_ok, w, n, EPS)

        out[f"rigidity_{m}m"] = val

    return pd.DataFrame(out, index=df.index)
