"""
因子评分与IC权重模块。

包含：
  - IC 权重加载（load_ic_weights）：读取 cs_ic_stats.csv，按阈值或白名单筛选，IC 归一化加权
  - 截面评分函数（_percentile_scores / _zscore_scores / _minmax_scores）：将因子值矩阵转换为可比得分
  - 合成得分与分组（_composite_and_groups）：IC加权合成 + N 分位分组收益计算
"""

import glob
import warnings
import os

import numpy as np
import pandas as pd

_RET_HORIZONS = {
    "ret100": "ret_fwd_100",
    "ret200": "ret_fwd_200",
    "ret300": "ret_fwd_300",
}


# ── 白名单读取 ─────────────────────────────────────────────────────────────────

def _load_whitelist(path: str) -> set[str]:
    """
    读取因子池白名单 txt 文件，返回 factor_col 集合。
    每行一个 factor_col，# 开头为注释行，空行忽略。
    """
    result = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                result.add(line)
    return result


# ── IC 权重加载 ────────────────────────────────────────────────────────────────

def load_ic_weights(
    ic_stats_root: str,
    threshold: float = 0.02,
    whitelist: set[str] | None = None,
) -> dict[str, dict]:
    """
    读取所有因子的 cs_ic_stats.csv，筛选 session=all 的行，
    按以下方式确定入选因子：
      - whitelist 为 None  → 保留 |ic_mean| >= threshold 的因子（默认行为）
      - whitelist 不为 None → 保留 factor_col 在白名单内的因子（忽略 threshold）

    Returns
    -------
    weights : dict  ret_horizon → {
        'weights':       {factor_col: normalized_weight},
        'factor_names':  {factor_col: factor_name},
        'ic_means':      {factor_col: raw_ic_mean},
    }
    """
    csv_files = glob.glob(os.path.join(ic_stats_root, "*", "cs_ic_stats.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 cs_ic_stats.csv：{ic_stats_root}")

    dfs = []
    for path in csv_files:
        factor_name = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        df["factor_name"] = factor_name
        dfs.append(df)

    all_stats = pd.concat(dfs, ignore_index=True)
    all_stats = all_stats[all_stats["session"] == "all"].copy()

    result = {}
    for ret_h in _RET_HORIZONS:
        base = all_stats[all_stats["ret_horizon"] == ret_h]

        if whitelist is not None:
            sub = base[base["factor_col"].isin(whitelist)].copy()
        else:
            sub = base[base["ic_mean"].abs() >= threshold].copy()

        if sub.empty:
            result[ret_h] = {"weights": {}, "factor_names": {}, "ic_means": {}}
            continue

        total = sub["ic_mean"].abs().sum()
        weights      = {}
        factor_names = {}
        ic_means     = {}
        for _, row in sub.iterrows():
            fc = row["factor_col"]
            weights[fc]      = row["ic_mean"] / total
            factor_names[fc] = row["factor_name"]
            ic_means[fc]     = row["ic_mean"]

        result[ret_h] = {
            "weights":      weights,
            "factor_names": factor_names,
            "ic_means":     ic_means,
        }

    return result


def _save_weights(out_dir: str, ic_info: dict) -> None:
    """将本次筛选的权重表写入 _weights.csv。"""
    rows = [
        {"factor_col": fc, "ic_mean": ic_info["ic_means"][fc], "weight": ic_info["weights"][fc]}
        for fc in ic_info["weights"]
    ]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "_weights.csv"), index=False)


# ── 截面评分函数 ───────────────────────────────────────────────────────────────

def _percentile_scores(f_mat: np.ndarray) -> np.ndarray:
    """
    f_mat : (T, N)  因子值矩阵（含 NaN）
    返回  : (T, N)  分位数得分矩阵，NaN 股票仍为 NaN，其余在 [0, 1)

    向量化实现：将 NaN 填为 -inf，对整个矩阵做两次 argsort，
    再扣除每行无效位置数得到有效股票内的 0-indexed 排名，除以 n_valid。
    """
    valid_mask  = np.isfinite(f_mat)
    n_invalid   = (~valid_mask).sum(axis=1, keepdims=True)
    n_valid     = valid_mask.sum(axis=1, keepdims=True)

    f_filled = np.where(valid_mask, f_mat, -np.inf)
    ranks    = np.argsort(np.argsort(f_filled, axis=1), axis=1)
    adjusted = ranks - n_invalid

    nv_safe = np.where(n_valid > 0, n_valid, 1)
    scores  = np.where(valid_mask, adjusted / nv_safe, np.nan)
    return scores.astype(np.float64)


def _zscore_scores(f_mat: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
    """
    f_mat    : (T, N)  因子值矩阵（含 NaN）
    clip_std : 截断阈值，默认 ±3σ
    返回     : (T, N)  Z-score 得分矩阵，NaN 股票仍为 NaN

    向量化实现：nanmean/nanstd 批量计算，
    std < 1e-9 时输出 0.0，有效股票数 < 2 时输出 NaN。
    """
    valid_mask = np.isfinite(f_mat)
    n_valid    = valid_mask.sum(axis=1, keepdims=True)

    f_nan  = np.where(valid_mask, f_mat, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu    = np.nanmean(f_nan, axis=1, keepdims=True)
        sigma = np.nanstd( f_nan, axis=1, keepdims=True)

    sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
    z = np.clip((f_nan - mu) / sigma_safe, -clip_std, clip_std)

    scores = np.where(n_valid < 2, np.nan,
             np.where(sigma < 1e-9, 0.0, z))
    return scores.astype(np.float64)


def _minmax_scores(f_mat: np.ndarray) -> np.ndarray:
    """
    f_mat : (T, N)  因子值矩阵（含 NaN）
    返回  : (T, N)  MinMax 归一化得分，NaN 股票仍为 NaN，其余在 [0, 1]

    (x - min) / (max - min)；极差 < 1e-9 时输出 0.0；有效股票数 < 2 时输出 NaN。
    """
    valid_mask = np.isfinite(f_mat)
    n_valid    = valid_mask.sum(axis=1, keepdims=True)
    f_nan      = np.where(valid_mask, f_mat, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f_min = np.nanmin(f_nan, axis=1, keepdims=True)
        f_max = np.nanmax(f_nan, axis=1, keepdims=True)

    rng      = f_max - f_min
    rng_safe = np.where(rng < 1e-9, 1.0, rng)
    scores   = np.where(rng < 1e-9, 0.0, (f_nan - f_min) / rng_safe)
    scores   = np.where(valid_mask, scores, np.nan)
    return np.where(n_valid < 2, np.nan, scores).astype(np.float64)


def _get_score_fn(score_method: str):
    """根据 score_method 字符串返回对应的截面评分函数。"""
    if score_method == "zscore":
        return _zscore_scores
    elif score_method == "minmax":
        return _minmax_scores
    else:
        return _percentile_scores


# ── 合成得分与分组收益 ─────────────────────────────────────────────────────────

def _composite_and_groups(
    wide_factors: dict[str, "pd.DataFrame"],
    weights: dict[str, float],
    r_wide: "pd.DataFrame",
    score_method: str = "rank",
    score_cache: dict[str, np.ndarray] | None = None,
    n_groups: int = 10,
) -> "pd.DataFrame":
    """
    wide_factors : {factor_col: wide_df (index=SampleTime, cols=SecurityID)}
    weights      : {factor_col: w_i}
    r_wide       : wide_df for one return horizon
    score_method : "rank" / "zscore" / "minmax"
    score_cache  : 预计算的因子得分 {fc: (T,N) ndarray}，有缓存时跳过重新评分
    n_groups     : 分组数，默认 10

    Returns
    -------
    DataFrame  index=SampleTime, cols=[composite_mean, composite_std, n_valid, g1..g{n_groups}]

    全程向量化，无 Python 逐行循环。
    """
    ref_index   = r_wide.index
    ref_columns = r_wide.columns
    T = len(ref_index)
    N = len(ref_columns)

    _score_fn = _get_score_fn(score_method)

    composite  = np.zeros((T, N))
    weight_sum = np.zeros((T, N))

    for fc, w in weights.items():
        if score_cache is not None and fc in score_cache:
            sc_mat = score_cache[fc]
        elif fc in wide_factors:
            f_mat  = wide_factors[fc].reindex(index=ref_index, columns=ref_columns).to_numpy(np.float64)
            sc_mat = _score_fn(f_mat)
        else:
            continue
        valid  = np.isfinite(sc_mat)
        composite[valid]  += w * sc_mat[valid]
        weight_sum[valid] += abs(w)

    can_use   = weight_sum > 1e-9
    composite = np.where(can_use, composite / np.where(can_use, weight_sum, 1.0), np.nan)

    r_mat = r_wide.to_numpy(np.float64)

    # ── 向量化分组 ────────────────────────────────────────────────────────────
    cv_mask   = np.isfinite(composite)
    n_valid   = cv_mask.sum(axis=1)
    n_invalid = (~cv_mask).sum(axis=1, keepdims=True)

    comp_filled = np.where(cv_mask, composite, -np.inf)
    ranks       = np.argsort(np.argsort(comp_filled, axis=1), axis=1)
    adjusted    = ranks - n_invalid

    nv_safe_2d  = np.where(n_valid[:, np.newaxis] > 0, n_valid[:, np.newaxis], 1)
    groups_mat  = (adjusted * n_groups // nv_safe_2d).clip(0, n_groups - 1)

    enough   = n_valid >= n_groups
    r_finite = np.isfinite(r_mat)

    g_arrays = []
    for g in range(n_groups):
        in_group  = cv_mask & (groups_mat == g) & enough[:, np.newaxis]
        has_ret   = in_group & r_finite
        count     = has_ret.sum(axis=1)
        total     = np.where(has_ret, r_mat, 0.0).sum(axis=1)
        mean_ret  = np.where(count > 0, total / np.where(count > 0, count, 1.0), np.nan)
        mean_ret  = np.where(enough, mean_ret, np.nan)
        g_arrays.append(mean_ret)

    nv_safe_1d = np.where(n_valid > 0, n_valid, 1).astype(np.float64)
    c_sum      = np.where(cv_mask, composite, 0.0).sum(axis=1)
    c_means    = np.where(n_valid > 0, c_sum / nv_safe_1d, np.nan)
    c_sum2     = np.where(cv_mask, composite ** 2, 0.0).sum(axis=1)
    variance   = np.where(n_valid > 0, c_sum2 / nv_safe_1d - c_means ** 2, np.nan)
    c_stds     = np.where(n_valid > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan)

    out = pd.DataFrame(
        {"composite_mean": c_means, "composite_std": c_stds, "n_valid": n_valid},
        index=ref_index,
    )
    for g in range(n_groups):
        out[f"g{g + 1}"] = g_arrays[g]

    out.index.name = "SampleTime"
    return out
