"""
因子计算编排器。

如何添加新因子
--------------
1. 在 pipeline/factor/ 下新建 <name>.py，实现：
       def compute(df: pd.DataFrame) -> pd.DataFrame:
           ...  # 输入完整 stock-day df，输出只含因子列的 df
2. 在本文件底部的 _FACTORS 列表里加一行 import 和引用：
       from . import <name>
       _FACTORS = [..., <name>]

编排器会自动将所有因子的输出拼接后写入文件，无需其他改动。
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ._core import load_data
from . import bap

# ── 注册因子 ──────────────────────────────────────────────────────────────────
# 新增因子：import 后加入此列表即可
_FACTORS = [bap]


# ── 单股票日计算 ──────────────────────────────────────────────────────────────

def _compute_one(df: pd.DataFrame) -> pd.DataFrame:
    """依次调用所有因子模块，拼接结果。"""
    parts = [f.compute(df) for f in _FACTORS]
    return pd.concat(parts, axis=1)


def _worker(args) -> dict:
    day, secid, base_path, out_path, horizons = args
    try:
        df  = load_data(base_path, horizons)

        meta = df[["Date", "SampleTime", "SecurityID", "Market",
                   "ret_fwd_100", "ret_fwd_200", "ret_fwd_300"]].copy()
        factors = _compute_one(df)
        out = pd.concat([meta, factors], axis=1)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)

        # summary：只统计因子值列（不含 _has_limit）
        value_cols = [c for c in factors.columns if not c.endswith("_has_limit")]
        summary = {"Date": day, "SecurityID": secid, "Status": "OK", "Rows": len(out)}
        for c in value_cols:
            summary[f"nnz_{c}"] = int(out[c].notna().sum())
        return summary

    except Exception as e:
        return {"Date": day, "SecurityID": secid, "Status": f"FAIL: {e}"}


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def run_factors(
    base_root: str,
    factor_root: str,
    horizons: list[int],
    dates: list | None = None,
    max_workers: int | None = None,
):
    """
    批量计算因子。收益率在读取 base 文件时内联计算，无需单独的 returns 目录。

    Parameters
    ----------
    base_root   : base 数据根目录
    factor_root : 输出根目录
    horizons    : 前向收益率窗口列表（tick 数），如 [100, 200, 300]
    dates       : 指定日期列表；None 时自动扫描
    max_workers : 并行进程数
    """
    os.makedirs(factor_root, exist_ok=True)

    if dates is None:
        dates = sorted(
            d for d in os.listdir(base_root)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(base_root, d))
        )

    all_tasks = []
    for day in dates:
        base_day_dir = os.path.join(base_root, day)
        out_day_dir  = os.path.join(factor_root, day)
        stock_files = sorted(
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
            ))

    if max_workers == 1:
        iter_ = tqdm(all_tasks, desc="factors") if tqdm else all_tasks
        results = [_worker(t) for t in iter_]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs  = [pool.submit(_worker, t) for t in all_tasks]
            iter_ = tqdm(as_completed(futs), total=len(futs), desc="factors") \
                    if tqdm else as_completed(futs)
            results = [f.result() for f in iter_]

    day_map = defaultdict(list)
    for r in results:
        day_map[r["Date"]].append(r)
    for day, day_results in sorted(day_map.items()):
        out_day_dir = os.path.join(factor_root, day)
        os.makedirs(out_day_dir, exist_ok=True)
        pd.DataFrame(day_results).sort_values("SecurityID").reset_index(drop=True) \
          .to_csv(os.path.join(out_day_dir, "_summary.csv"), index=False)

    summary_path = os.path.join(factor_root, "_summary.csv")
    pd.DataFrame(results).sort_values(["Date", "SecurityID"]).reset_index(drop=True) \
      .to_csv(summary_path, index=False)
    print(f"因子计算完成，汇总：{summary_path}")
