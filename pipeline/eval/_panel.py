"""
IC 计算核心工具（供 cs_ic.py 使用）。

get_factor_cols  : 从 DataFrame 列名中提取指定因子的值列（去掉 _has_limit）
compute_ic_pair  : 对一对宽表计算 Pearson IC 和 Spearman RankIC
"""

import numpy as np
import pandas as pd

# 不属于因子值的列
_META_COLS = {"Date", "SampleTime", "SecurityID", "Market"}


def get_factor_cols(df: pd.DataFrame, factor_name: str) -> list[str]:
    """
    返回 df 中属于 factor_name 的因子值列（排除 _has_limit 列）。

    例：factor_name="bap" → ["bap_15m", "bap_30m", "bap_45m", "bap_60m", "bap_75m"]
    """
    prefix = f"{factor_name}_"
    return [
        c for c in df.columns
        if c.startswith(prefix) and not c.endswith("_has_limit")
    ]


def _pearson(f: np.ndarray, r: np.ndarray, axis: int) -> np.ndarray:
    """
    沿 axis 方向计算 Pearson 相关系数，NaN 位置联合排除。

    axis=1 → 逐行（CS-IC：每个时间点跨股票）
    """
    valid = np.isfinite(f) & np.isfinite(r)
    f = np.where(valid, f, np.nan)
    r = np.where(valid, r, np.nan)

    f_mean = np.nanmean(f, axis=axis, keepdims=True)
    r_mean = np.nanmean(r, axis=axis, keepdims=True)
    f_dm   = f - f_mean
    r_dm   = r - r_mean

    numer = np.nansum(f_dm * r_dm, axis=axis)
    denom = (
        np.sqrt(np.nansum(f_dm ** 2, axis=axis))
        * np.sqrt(np.nansum(r_dm ** 2, axis=axis))
    )
    return np.where(denom > 1e-12, numer / denom, np.nan)


def _nanrank_2d(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    纯 numpy 实现的二维 nanrank，NaN 保持为 NaN，有效值在各自行/列内排名。

    用 argsort(argsort(...)) 实现，比 pd.DataFrame.rank() 快约 2-3 倍。
    对于几乎无重复值的浮点因子，tie-handling 的差异可忽略不计。
    """
    nan_mask = ~np.isfinite(arr)
    tmp = arr.copy()
    tmp[nan_mask] = np.inf          # NaN 替换为 +inf，排序后落在最末位
    ranks = (
        np.argsort(np.argsort(tmp, axis=axis), axis=axis) + 1
    ).astype(np.float64)
    ranks[nan_mask] = np.nan        # 还原 NaN 位置
    return ranks


def compute_ic_pair(
    f_wide: pd.DataFrame,
    r_wide: pd.DataFrame,
    axis: int,
) -> tuple[pd.Series, pd.Series]:
    """
    对一对宽表计算 IC（Pearson）和 RankIC（Spearman）。

    Parameters
    ----------
    f_wide : 因子宽表，行 = 时间点，列 = SecurityID
    r_wide : 收益率宽表，行 = 时间点，列 = SecurityID
    axis   : 1 → 逐行（CS-IC）

    Returns
    -------
    ic, rankic : 两个 Series，index 为 SampleTime
    """
    f_arr = f_wide.to_numpy(dtype=np.float64)
    r_arr = r_wide.to_numpy(dtype=np.float64)

    ic_arr     = _pearson(f_arr, r_arr, axis)
    f_ranked   = _nanrank_2d(f_arr, axis)
    r_ranked   = _nanrank_2d(r_arr, axis)
    rankic_arr = _pearson(f_ranked, r_ranked, axis)

    idx = f_wide.index if axis == 1 else f_wide.columns
    return pd.Series(ic_arr, index=idx), pd.Series(rankic_arr, index=idx)
