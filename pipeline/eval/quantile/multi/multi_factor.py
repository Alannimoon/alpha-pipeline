"""
多因子合成分层模块（编排层）。

运行逻辑
--------
对每个 date：
  1. 读取 factor/{name}/{date}.parquet（各因子长表，pivot 成宽表）
  2. 读取 base/{date}.parquet（列投影读 ret_fwd 列，pivot 成宽表）
  3. 预计算因子得分矩阵（复用于 3 个 ret_horizon）
  4. 对每个 ret_horizon：IC加权合成分 → N 分位分组收益 → 写 {date}.parquet

输出目录
--------
{eval_root}/multi_factor_quantile/g{n_groups}/{factor_pool}/{score_method}/ret{100|200|300}/
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .factor_score import (
    _RET_HORIZONS,
    _load_whitelist,
    _get_score_fn,
    _composite_and_groups,
    _save_weights,
    load_ic_weights,
)
from .charts import run_post_compute


# ── 单日数据读取 ───────────────────────────────────────────────────────────────

def _read_day_data(
    factor_root: str,
    base_root: str,
    all_fc_to_fn: dict[str, str],
    day: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None:
    """
    读取单日所有因子 parquet + base parquet，pivot 成宽表。

    Parameters
    ----------
    factor_root  : 因子根目录
    base_root    : base 根目录
    all_fc_to_fn : {factor_col: factor_name}
    day          : 日期字符串，如 "20250102"

    Returns
    -------
    (wide_factors, ret_wides) 或 None（数据缺失时）
    wide_factors : {factor_col: DataFrame(index=SampleTime, cols=SecurityID)}
    ret_wides    : {ret_col: DataFrame(index=SampleTime, cols=SecurityID)}
    """
    # 按 factor_name 分组读取
    name_to_cols: dict[str, list[str]] = {}
    for fc, fn in all_fc_to_fn.items():
        name_to_cols.setdefault(fn, []).append(fc)

    wide_factors: dict[str, pd.DataFrame] = {}
    for factor_name, cols in name_to_cols.items():
        factor_path = os.path.join(factor_root, factor_name, f"{day}.parquet")
        if not os.path.exists(factor_path):
            continue
        needed = ["SampleTime", "SecurityID"] + [c for c in cols]
        factor_long = pd.read_parquet(factor_path, columns=needed)
        for fc in cols:
            if fc in factor_long.columns:
                wide_factors[fc] = factor_long.pivot(
                    index="SampleTime", columns="SecurityID", values=fc
                )

    if not wide_factors:
        return None

    ret_cols  = list(_RET_HORIZONS.values())
    base_path = os.path.join(base_root, f"{day}.parquet")
    if not os.path.exists(base_path):
        return None

    base_long = pd.read_parquet(
        base_path, columns=["SampleTime", "SecurityID"] + ret_cols
    )
    ret_wides = {
        rc: base_long.pivot(index="SampleTime", columns="SecurityID", values=rc)
        for rc in ret_cols
    }

    return wide_factors, ret_wides


# ── 单日计算 ───────────────────────────────────────────────────────────────────

def _compute_day(
    factor_root: str,
    base_root: str,
    out_dirs: dict[str, str],
    ic_info_by_horizon: dict[str, dict],
    day: str,
    score_method: str = "rank",
    n_groups: int = 10,
) -> str:
    """
    单日计算入口。

    执行顺序
    --------
    1. 跳过检查：3个horizon的输出均已存在 → 直接返回
    2. 读取数据（factor parquets + base parquet）
    3. 预计算 score_cache：因子得分矩阵各算一次，供 3 个 ret_horizon 复用
    4. 对每个 ret_horizon：IC加权合成分 → 十分位分组 → 写 {date}.parquet
    """
    # ── 1. 跳过已完成的日期 ──────────────────────────────────────────────────
    horizons_with_weights = [
        ret_h for ret_h in _RET_HORIZONS
        if ic_info_by_horizon.get(ret_h, {}).get("weights")
    ]
    if horizons_with_weights and all(
        os.path.exists(os.path.join(out_dirs[ret_h], f"{day}.parquet"))
        for ret_h in horizons_with_weights
    ):
        return day

    # ── 2. 确定本次需要的因子列 ──────────────────────────────────────────────
    all_fc_to_fn: dict[str, str] = {}
    for info in ic_info_by_horizon.values():
        all_fc_to_fn.update(info["factor_names"])

    if not all_fc_to_fn:
        return day

    # ── 3. 读取数据 ───────────────────────────────────────────────────────────
    result = _read_day_data(factor_root, base_root, all_fc_to_fn, day)
    if result is None:
        return day
    wide_factors, ret_wides = result

    # ── 4. 预计算因子得分（复用于 3 个 ret_horizon）─────────────────────────
    ref_r_wide = next(
        (ret_wides[rc] for rc in _RET_HORIZONS.values()
         if rc in ret_wides and not ret_wides[rc].empty),
        None,
    )
    score_cache: dict[str, np.ndarray] | None = None
    if ref_r_wide is not None:
        _score_fn = _get_score_fn(score_method)
        ref_index   = ref_r_wide.index
        ref_columns = ref_r_wide.columns
        score_cache = {
            fc: _score_fn(
                wide_factors[fc].reindex(index=ref_index, columns=ref_columns)
                .to_numpy(np.float64)
            )
            for fc in all_fc_to_fn
            if fc in wide_factors
        }

    # ── 5. 逐 horizon 合成分 → 分组 → 写 parquet ────────────────────────────
    for ret_h, ret_col in _RET_HORIZONS.items():
        info    = ic_info_by_horizon.get(ret_h, {})
        weights = info.get("weights", {})
        if not weights or ret_col not in ret_wides:
            continue

        r_wide = ret_wides[ret_col]
        if r_wide.empty:
            continue

        df = _composite_and_groups(
            wide_factors, weights, r_wide,
            score_method=score_method,
            score_cache=score_cache,
            n_groups=n_groups,
        )
        df = df.reset_index()
        df.insert(0, "Date", day)

        out_dir = out_dirs[ret_h]
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, f"{day}.parquet"), index=False)

    return day


# ── Worker（供 ProcessPoolExecutor 调用）──────────────────────────────────────

def _worker(args) -> str:
    factor_root, base_root, out_dirs, ic_info_by_horizon, day, score_method, n_groups = args
    return _compute_day(
        factor_root, base_root, out_dirs, ic_info_by_horizon, day, score_method, n_groups
    )


# ── 批量入口 ───────────────────────────────────────────────────────────────────

def run_multi_factor_quantile(
    factor_root: str,
    base_root: str,
    eval_root: str,
    ic_stats_root: str,
    threshold: float = 0.02,
    dates: list[str] | None = None,
    max_workers: int | None = None,
    score_method: str = "rank",
    factor_pool: str = "threshold",
    whitelist_path: str | None = None,
    n_groups: int = 10,
):
    """
    Parameters
    ----------
    factor_root    : 因子数据根目录（如 result/factor）
    base_root      : base 数据根目录（如 result/base）
    eval_root      : 评估结果输出根目录（如 result/eval）
    ic_stats_root  : cs_ic_stats.csv 的根目录（如 result/ic_stats）
    threshold      : IC 筛选阈值，factor_pool="threshold" 时生效
    dates          : 指定日期列表；None 时自动扫描
    max_workers    : 并行进程数；建议 4~8，None 时用 CPU 核数
    score_method   : "rank"（分位数得分）/ "zscore"（Z-score ±3截断）/ "minmax"（MinMax 归一化）
    factor_pool    : "threshold" / "union" / "intersection"，决定因子集合和输出子目录
    whitelist_path : 白名单 txt 路径；factor_pool != "threshold" 时需提供
    n_groups       : 分组数，默认 10
    """
    whitelist  = _load_whitelist(whitelist_path) if whitelist_path else None
    ic_weights = load_ic_weights(ic_stats_root, threshold=threshold, whitelist=whitelist)

    print(f"[factor_pool={factor_pool}  score_method={score_method}]")
    for ret_h, info in ic_weights.items():
        cols = list(info["weights"].keys())
        print(f"[{ret_h}] 通过筛选因子（{len(cols)} 个）：{cols}")
        for fc, w in info["weights"].items():
            print(f"    {fc:30s}  ic_mean={info['ic_means'][fc]:+.4f}  weight={w:+.4f}")

    if dates is None:
        any_fn = next(
            (fn for info in ic_weights.values() for fn in info["factor_names"].values()),
            None,
        )
        if any_fn is None:
            print("没有通过筛选的因子，退出。")
            return
        scan_dir = os.path.join(factor_root, any_fn)
        dates = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(scan_dir)
            if f.endswith(".parquet") and not f.startswith("_")
            and len(os.path.splitext(f)[0]) == 8
            and os.path.splitext(f)[0].isdigit()
        )

    base_out = os.path.join(eval_root, "multi_factor_quantile", f"g{n_groups}", factor_pool, score_method)
    out_dirs: dict[str, str] = {}
    for ret_h in _RET_HORIZONS:
        d = os.path.join(base_out, ret_h)
        os.makedirs(d, exist_ok=True)
        out_dirs[ret_h] = d
        _save_weights(d, ic_weights[ret_h])

    tasks = [
        (factor_root, base_root, out_dirs, ic_weights, day, score_method, n_groups)
        for day in dates
    ]

    if max_workers == 1:
        day_iter = tqdm(tasks, desc="Multi-Factor Quantile") if tqdm else tasks
        for t in day_iter:
            _worker(t)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs  = [pool.submit(_worker, t) for t in tasks]
            inner = (
                tqdm(as_completed(futs), total=len(futs), desc="Multi-Factor Quantile")
                if tqdm else as_completed(futs)
            )
            for fut in inner:
                fut.result()

    run_post_compute(out_dirs)
    print(f"多因子分层计算完成：{base_out}")
