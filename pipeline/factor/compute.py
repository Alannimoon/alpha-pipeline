"""
分钟线因子计算编排器。

注册的因子均为价格/成交量驱动型（不依赖盘口数据），
可在分钟 OHLCV 数据上运行。
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ._core import load_data
from . import mom, acc_mom, neg_skew, amp_slice, rigidity, pv_corr

_FACTOR_MAP = {
    "mom":       mom,
    "acc_mom":   acc_mom,
    "neg_skew":  neg_skew,
    "amp_slice": amp_slice,
    "rigidity":  rigidity,
    "pv_corr":   pv_corr,
}


def _worker(args) -> dict:
    day, secid, base_path, out_path, horizons, factor_name = args
    try:
        df   = load_data(base_path, horizons)

        ret_cols = [f"ret_fwd_{h}" for h in horizons]
        meta_cols = ["Date", "SampleTime", "SecurityID", "Market"] + ret_cols
        meta    = df[[c for c in meta_cols if c in df.columns]].copy()
        factors = _FACTOR_MAP[factor_name].compute(df)
        out     = pd.concat([meta, factors], axis=1)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)

        value_cols = [c for c in factors.columns if not c.endswith("_has_limit")]
        summary = {"Date": day, "SecurityID": secid, "Status": "OK", "Rows": len(out)}
        for c in value_cols:
            summary[f"nnz_{c}"] = int(out[c].notna().sum())
        return summary

    except Exception as e:
        return {"Date": day, "SecurityID": secid, "Status": f"FAIL: {e}"}


def run_factors(
    base_root: str,
    factor_root: str,
    factor_name: str,
    horizons: list[int],
    dates: list | None = None,
    max_workers: int | None = None,
):
    """
    批量计算分钟线因子。

    Parameters
    ----------
    base_root   : 分钟线 base 数据根目录
    factor_root : 输出根目录（写入 factor_root/{factor_name}/{date}/）
    factor_name : 因子名称，须在 _FACTOR_MAP 中注册
    horizons    : 前向收益率窗口（tick 数），如 [5, 10, 15]
    dates       : 指定日期列表；None 时自动扫描
    max_workers : 并行进程数
    """
    if factor_name not in _FACTOR_MAP:
        raise ValueError(f"未知因子 '{factor_name}'，可选：{list(_FACTOR_MAP)}")

    factor_out_root = os.path.join(factor_root, factor_name)
    os.makedirs(factor_out_root, exist_ok=True)

    if dates is None:
        dates = sorted(
            d for d in os.listdir(base_root)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(base_root, d))
        )

    all_tasks = []
    for day in dates:
        base_day_dir = os.path.join(base_root, day)
        out_day_dir  = os.path.join(factor_out_root, day)
        stock_files  = sorted(
            f for f in os.listdir(base_day_dir)
            if f.endswith(".csv") and not f.startswith("_")
        )
        for f in stock_files:
            all_tasks.append((
                day,
                os.path.splitext(f)[0],
                os.path.join(base_day_dir, f),
                os.path.join(out_day_dir, f),
                horizons,
                factor_name,
            ))

    tasks_by_day = {
        day: list(group)
        for day, group in groupby(all_tasks, key=lambda t: t[0])
    }

    def _write_day_summary(day_results, out_day_dir):
        os.makedirs(out_day_dir, exist_ok=True)
        pd.DataFrame(day_results).sort_values("SecurityID").reset_index(drop=True) \
          .to_csv(os.path.join(out_day_dir, "_summary.csv"), index=False)

    if max_workers == 1:
        iter_ = tqdm(all_tasks, desc="factors") if tqdm else all_tasks
        for day, group in groupby(iter_, key=lambda t: t[0]):
            day_results = [_worker(t) for t in group]
            _write_day_summary(day_results, os.path.join(factor_out_root, day))
    else:
        pbar = tqdm(total=len(all_tasks), desc="factors") if tqdm else None
        pool = ProcessPoolExecutor(max_workers=max_workers)
        try:
            for day, day_tasks in tasks_by_day.items():
                futs = [pool.submit(_worker, t) for t in day_tasks]
                day_results = []
                for f in as_completed(futs):
                    day_results.append(f.result())
                    if pbar:
                        pbar.update(1)
                _write_day_summary(day_results, os.path.join(factor_out_root, day))
        finally:
            for p in pool._processes.values():
                p.terminate()
            pool.shutdown(wait=False)
        if pbar:
            pbar.close()

    # 全局汇总
    summary_path = os.path.join(factor_out_root, "_summary.csv")
    day_dirs = sorted(
        d for d in os.listdir(factor_out_root)
        if len(d) == 8 and d.isdigit()
    )
    pd.concat(
        [pd.read_csv(os.path.join(factor_out_root, d, "_summary.csv"), dtype=str)
         for d in day_dirs],
        ignore_index=True,
    ).to_csv(summary_path, index=False)
    print(f"分钟线因子计算完成，汇总：{summary_path}")
