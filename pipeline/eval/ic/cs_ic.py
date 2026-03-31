"""
截面 IC 计算模块。

对每个时间点 (Date, SampleTime)，在所有股票截面上计算因子值与未来收益率的
Pearson（IC）和 Spearman（RankIC）相关系数。

输出
----
result/eval/cs_ic/{factor_name}/ret100/{day}.parquet
result/eval/cs_ic/{factor_name}/ret200/{day}.parquet
result/eval/cs_ic/{factor_name}/ret300/{day}.parquet

每个文件含完整全天时间序列（含 Date、SampleTime 列），
Session 过滤（all / am / pm）由 ic_report 统计阶段完成。
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .._panel import build_wide_day, compute_ic_pair

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
    计算单日所有 ret_horizon 的 CS-IC。

    Returns
    -------
    {"ret100": df, "ret200": df, "ret300": df}
    每个 df 含列：Date, SampleTime, ic_{fc}, rankic_{fc}, ...
    """
    wide, factor_cols = build_wide_day(factor_root, base_root, factor_name, day)
    if not wide or not factor_cols:
        return {}

    results: dict[str, pd.DataFrame] = {}

    for h_key, h_col in _RET_HORIZONS.items():
        r_wide = wide[h_col]

        ic_cols: dict[str, pd.Series] = {}
        for fc in factor_cols:
            ic, rankic = compute_ic_pair(wide[fc], r_wide, axis=1)
            ic_cols[f"ic_{fc}"]     = ic
            ic_cols[f"rankic_{fc}"] = rankic

        df = pd.DataFrame(ic_cols)
        df.index.name = "SampleTime"
        df = df.reset_index()
        df.insert(0, "Date", day)
        results[h_key] = df

    return results


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def _worker(args) -> str:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    factor_root, base_root, base_dir, factor_name, day = args
    day_results = _compute_day(factor_root, base_root, factor_name, day)
    for h_key, df in day_results.items():
        out_dir = os.path.join(base_dir, h_key)
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, f"{day}.parquet"), index=False)
    return day


def run_cs_ic(
    factor_root: str,
    base_root: str,
    eval_root: str,
    factor_name: str,
    dates: list[str] | None = None,
    max_workers: int | None = None,
):
    """
    批量计算截面 IC。

    Parameters
    ----------
    factor_root : 因子数据根目录
    base_root   : base 数据根目录
    eval_root   : 评估结果输出根目录
    factor_name : 因子名称，如 "bap"
    dates       : 指定日期列表；None 时自动扫描
    max_workers : 并行进程数
    """
    if dates is None:
        factor_day_root = os.path.join(factor_root, factor_name)
        dates = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(factor_day_root)
            if f.endswith(".parquet") and not f.startswith("_")
            and len(os.path.splitext(f)[0]) == 8
            and os.path.splitext(f)[0].isdigit()
        )

    base_dir = os.path.join(eval_root, "cs_ic", factor_name)
    tasks = [(factor_root, base_root, base_dir, factor_name, day) for day in dates]

    if max_workers == 1:
        day_iter = tqdm(tasks, desc="CS-IC") if tqdm else tasks
        for t in day_iter:
            _worker(t)
    else:
        pool = ProcessPoolExecutor(max_workers=max_workers)
        try:
            futs  = [pool.submit(_worker, t) for t in tasks]
            inner = tqdm(as_completed(futs), total=len(futs), desc="CS-IC") \
                    if tqdm else as_completed(futs)
            for f in inner:
                f.result()
        finally:
            for p in pool._processes.values():
                p.terminate()
            pool.shutdown(wait=False)

    print(f"CS-IC 计算完成：{base_dir}")
