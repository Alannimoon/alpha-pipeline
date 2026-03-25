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
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ._core import is_limit_tick, window_valid_mask, rolling_any, TICKS_PER_MIN

WINDOWS_MIN       = [10, 20, 30, 45, 60, 90, 105]
MAX_INVALID_RATIO = 0.10
EPS               = 1e-8


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日的完整 DataFrame（由 _core.load_data 加载）
    输出：只含因子列的 DataFrame，index 与输入对齐

    列名：rigidity_10m, rigidity_10m_has_limit, rigidity_20m, ...

    实现说明
    --------
    用 sliding_window_view 一次性取出所有滑动窗口，通过矩阵点积批量计算
    二次多项式法方程的 8 个统计量（sn, sx, sx2, sx3, sx4, sy, sxy, sx2y），
    再用 Cramer 法则对所有有效窗口同时解方程，向量化计算 R²。
    全程零 Python 循环，较原逐 tick 实现有 100-500x 加速。
    """
    can_use = df["CanUsePrice"].to_numpy(bool)
    price   = df["Price"].to_numpy(np.float64)
    limit   = is_limit_tick(df)
    n       = len(df)

    # 预计算全局有效掩码（CanUsePrice & 价格有限）
    valid_tick = can_use & np.isfinite(price)

    out = {}

    for m in WINDOWS_MIN:
        w      = m * TICKS_PER_MIN
        w_ok   = window_valid_mask(can_use, w, MAX_INVALID_RATIO)
        hl_arr = np.where(w_ok, rolling_any(limit, w), False)
        val    = np.full(n, np.nan)

        if n < w:
            out[f"rigidity_{m}m"]           = val
            out[f"rigidity_{m}m_has_limit"] = hl_arr.astype(bool)
            continue

        # ── 滑动窗口视图（零拷贝）────────────────────────────────────────────
        p_win = sliding_window_view(price,      w)   # (N, w)  view
        vw    = sliding_window_view(valid_tick, w)   # (N, w)  bool view

        # ── x 幂次（每窗口大小只算一次）──────────────────────────────────────
        x_arr  = np.arange(w, dtype=np.float64)
        x2_arr = x_arr * x_arr

        # ── 向量化统计量（矩阵点积代替逐行循环）─────────────────────────────
        # v_f: 0/1 掩码，p_m: 无效处置零，避免 NaN 污染求和
        v_f  = vw.astype(np.float64)             # (N, w)
        p_m  = np.where(vw, p_win, 0.0)          # (N, w)

        sn   = v_f.sum(axis=1)                   # (N,)  有效 tick 数
        sx   = v_f.dot(x_arr)                    # (N,)  Σ xᵢ
        sx2  = v_f.dot(x2_arr)                   # (N,)  Σ xᵢ²
        sx3  = v_f.dot(x2_arr * x_arr)           # (N,)  Σ xᵢ³
        sx4  = v_f.dot(x2_arr * x2_arr)          # (N,)  Σ xᵢ⁴
        sy   = p_m.sum(axis=1)                   # (N,)  Σ yᵢ
        sxy  = p_m.dot(x_arr)                    # (N,)  Σ xᵢ yᵢ
        sx2y = p_m.dot(x2_arr)                   # (N,)  Σ xᵢ² yᵢ

        del v_f, p_m

        # ── Cramer 法则（全向量化，零 Python 循环）───────────────────────────
        # 法方程 A·[c, b, a]ᵀ = B，其中
        # A = [[sx4, sx3, sx2],      B = [sx2y]
        #      [sx3, sx2, sx ],          [sxy ]
        #      [sx2, sx,  sn ]]          [sy  ]
        det_A = (sx4 * (sx2 * sn  - sx  * sx )
               - sx3 * (sx3 * sn  - sx  * sx2)
               + sx2 * (sx3 * sx  - sx2 * sx2))

        # b 的 Cramer 行列式（替换第 1 列为 B）
        det_b = (sx4  * (sxy * sn  - sx  * sy )
               - sx2y * (sx3 * sn  - sx  * sx2)
               + sx2  * (sx3 * sy  - sxy * sx2))

        # c 的 Cramer 行列式（替换第 0 列为 B）
        det_c = (sx2y * (sx2 * sn  - sx  * sx )
               - sx3  * (sxy * sn  - sx  * sy )
               + sx2  * (sxy * sx  - sx2 * sy ))

        # a 的 Cramer 行列式（替换第 2 列为 B）
        det_a = (sx4  * (sx2 * sy  - sxy * sx )
               - sx3  * (sx3 * sy  - sxy * sx2)
               + sx2y * (sx3 * sx  - sx2 * sx2))

        # ── 有效窗口筛选 ──────────────────────────────────────────────────────
        # w_ok[w-1:] 对应窗口索引 0..N-1
        active = w_ok[w - 1:] & (sn >= 3) & (np.abs(det_A) > 1e-20)

        if not active.any():
            out[f"rigidity_{m}m"]           = val
            out[f"rigidity_{m}m_has_limit"] = hl_arr.astype(bool)
            continue

        win_idx = np.where(active)[0]   # 有效窗口索引（0-based in [0, N-1]）
        dA      = det_A[win_idx]

        b_coef = det_b[win_idx] / dA    # (K,) 线性斜率
        c_coef = det_c[win_idx] / dA    # (K,) 二次系数
        a_coef = det_a[win_idx] / dA    # (K,) 截距

        # ── 计算 R²（只对有效窗口）────────────────────────────────────────────
        pw_a = p_win[win_idx]           # (K, w)  实际拷贝
        vw_a = vw[win_idx]              # (K, w)  bool

        # 拟合值 y_hat = a + b·x + c·x²（广播）
        y_hat = (a_coef[:, None]
                 + b_coef[:, None] * x_arr[None, :]
                 + c_coef[:, None] * x2_arr[None, :])   # (K, w)

        sn_a  = sn[win_idx]                             # (K,)
        y_bar = (np.where(vw_a, pw_a, 0.0).sum(axis=1)
                 / sn_a)                                # (K,) 有效点均值

        ss_res = np.where(vw_a, (pw_a - y_hat)          ** 2, 0.0).sum(axis=1)
        ss_tot = np.where(vw_a, (pw_a - y_bar[:, None]) ** 2, 0.0).sum(axis=1)

        r2       = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
        rigidity = b_coef * r2 * np.log(1.0 / (np.abs(c_coef) + EPS))
        rigidity = np.where(np.isfinite(r2), rigidity, np.nan)

        # ── 写回原始 tick 位置（窗口索引 i → tick i+w-1）─────────────────────
        val[win_idx + w - 1] = rigidity

        out[f"rigidity_{m}m"]           = val
        out[f"rigidity_{m}m_has_limit"] = hl_arr.astype(bool)

    return pd.DataFrame(out, index=df.index)
