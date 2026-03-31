"""
单因子截面分层计算模块。

对每个时间点 (Date, SampleTime)，将所有股票按因子值分成 N_GROUPS=5 组
（g1 = 因子值最低，g5 = 因子值最高），计算各组的前向收益均值。

分组方式
--------
rank-based 均匀分组，全程向量化（(T, N) 矩阵操作），无逐时刻 Python 循环。

输出
----
result/eval/cs_quantile/{factor_name}/{ret_horizon}_{session}/{day}.parquet

列：Date, SampleTime, g1_{fc}, ..., g5_{fc}, n_valid_{fc}, ...
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ..._panel import build_wide_day
from .charts import run_post_compute

N_GROUPS = 5

_RET_HORIZONS = {
    "ret100": "ret_fwd_100",
    "ret200": "ret_fwd_200",
    "ret300": "ret_fwd_300",
}


def _compute_day(
    factor_root: str,
    base_root: str,
    factor_name: str,
    day: str,
) -> dict[str, pd.DataFrame]:
    """
    单日截面分层，返回各 (ret_horizon, session) 的 DataFrame。
    分组计算全程向量化：(T, N) 矩阵操作，无逐时刻 Python 循环。
    """
    wide, factor_cols = build_wide_day(factor_root, base_root, factor_name, day)
    if not wide or not factor_cols:
        return {}

    results: dict[str, pd.DataFrame] = {}

    for h_key, h_col in _RET_HORIZONS.items():
        r_wide = wide[h_col]
        r_mat  = r_wide.to_numpy(np.float64)

        col_data: dict[str, np.ndarray] = {}

        for fc in factor_cols:
            f_mat = (
                wide[fc]
                .reindex(index=r_wide.index, columns=r_wide.columns)
                .to_numpy(np.float64)
            )

            valid_mask = np.isfinite(f_mat)
            n_valid    = valid_mask.sum(axis=1)
            n_invalid  = (~valid_mask).sum(axis=1, keepdims=True)

            f_filled = np.where(valid_mask, f_mat, -np.inf)
            ranks    = np.argsort(np.argsort(f_filled, axis=1), axis=1)
            adjusted = ranks - n_invalid

            nv_safe    = np.where(n_valid[:, np.newaxis] > 0, n_valid[:, np.newaxis], 1)
            groups_mat = (adjusted * N_GROUPS // nv_safe).clip(0, N_GROUPS - 1)

            enough   = n_valid >= N_GROUPS
            r_finite = np.isfinite(r_mat)

            for g in range(N_GROUPS):
                in_group = valid_mask & (groups_mat == g) & enough[:, np.newaxis]
                has_ret  = in_group & r_finite
                count    = has_ret.sum(axis=1)
                total    = np.where(has_ret, r_mat, 0.0).sum(axis=1)
                mean_ret = np.where(count > 0, total / np.where(count > 0, count, 1.0), np.nan)
                col_data[f"g{g + 1}_{fc}"] = np.where(enough, mean_ret, np.nan)

            col_data[f"n_valid_{fc}"] = n_valid.astype(np.float64)

        base_df = pd.DataFrame(col_data, index=r_wide.index)
        base_df.index.name = "SampleTime"
        base_df = base_df.reset_index()
        base_df.insert(0, "Date", day)

        results[h_key] = base_df.reset_index(drop=True)

    return results


def _worker(args) -> str:
    factor_root, base_root, base_dir, factor_name, day = args
    day_results = _compute_day(factor_root, base_root, factor_name, day)
    for key, df in day_results.items():
        out_dir = os.path.join(base_dir, key)
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, f"{day}.parquet"), index=False)
    return day


def run_cs_quantile(
    factor_root: str,
    base_root: str,
    eval_root: str,
    factor_name: str,
    dates: list[str] | None = None,
    max_workers: int | None = None,
):
    """批量计算单因子截面分层，完成后自动生成汇总与图表。"""
    if dates is None:
        factor_day_root = os.path.join(factor_root, factor_name)
        dates = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(factor_day_root)
            if f.endswith(".parquet") and not f.startswith("_")
            and len(os.path.splitext(f)[0]) == 8
            and os.path.splitext(f)[0].isdigit()
        )

    base_dir = os.path.join(eval_root, "cs_quantile", factor_name)
    tasks = [(factor_root, base_root, base_dir, factor_name, day) for day in dates]

    if max_workers == 1:
        day_iter = tqdm(tasks, desc="CS-Quantile") if tqdm else tasks
        for t in day_iter:
            _worker(t)
    else:
        pool = ProcessPoolExecutor(max_workers=max_workers)
        try:
            futs  = [pool.submit(_worker, t) for t in tasks]
            inner = tqdm(as_completed(futs), total=len(futs), desc="CS-Quantile") \
                    if tqdm else as_completed(futs)
            for f in inner:
                f.result()
        finally:
            for p in pool._processes.values():
                p.terminate()
            pool.shutdown(wait=False)

    run_post_compute(base_dir, _RET_HORIZONS)
    print(f"单因子分层计算完成：{base_dir}")
