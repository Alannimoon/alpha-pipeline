"""
多因子合成分层模块（编排层）。

运行逻辑
--------
第一次运行（cold start）：
  对每个 date：
    1. 读取原始 CSV：51 factor目录 × ~500 只股票 ≈ 25,500 个文件 (ThreadPoolExecutor 并行)
    2. 将所有因子值 + 3条收益率列 合并成一张 (T×N, 54列) 长表写入 parquet 缓存
    3. 从缓存数据计算合成分、十分位分组收益，写 {date}.csv

后续运行（有缓存）：
  对每个 date：
    1. 若 3 个 ret_horizon 的 {date}.csv 均已存在 → 直接跳过
    2. 否则读 parquet（按需选列）→ reshape(T, N) 还原宽表
    3. 继续步骤 3（计算 + 写输出）

缓存目录
--------
{cache_root}/{date}.parquet
  列：SampleTime, SecurityID, <factor_col_1>, ..., <factor_col_N>,
      ret_fwd_100, ret_fwd_200, ret_fwd_300
  行序：SampleTime 外层（repeat T×N），SecurityID 内层（tile T×N），
        保证 reshape(T, N) 可直接还原宽表，无需 pivot/unstack。
  dtype：float32（节省约 50% 空间）

输出目录
--------
{eval_root}/multi_factor_quantile/{factor_pool}/{score_method}/ret{100|200|300}/
"""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .factor_score import (
    _RET_HORIZONS,
    _load_whitelist,
    _percentile_scores,
    _zscore_scores,
    _composite_and_groups,
    _save_weights,
    load_ic_weights,
)
from .wide_cache import save_wide_cache, load_wide_cache
from .multi_factor_charts import (
    _build_daily,
    _build_summary,
    _build_cum_tick,
    _build_cum_daily,
    _build_cum_tick_chart,
    _build_intraday_slot_charts,
)


# ── 原始 CSV 宽表读取 ──────────────────────────────────────────────────────────

def _read_one_factor_dir(
    day_dir: str,
    cols: list[str],
) -> dict[str, dict[str, pd.Series]]:
    """读取单个 factor_name 目录下所有股票文件，返回 {fc: {secid: Series}}。"""
    files = sorted(
        f for f in os.listdir(day_dir)
        if f.endswith(".csv") and not f.startswith("_")
    )
    needed = ["SampleTime", "SecurityID"] + cols
    result: dict[str, dict[str, pd.Series]] = {fc: {} for fc in cols}

    for fname in files:
        try:
            df = pd.read_csv(
                os.path.join(day_dir, fname),
                usecols=needed,
                dtype={"SampleTime": str, "SecurityID": str},
            )
        except ValueError:
            continue
        if df.empty:
            continue
        secid   = df["SecurityID"].iloc[0]
        indexed = df.set_index("SampleTime")
        for fc in cols:
            if fc in indexed.columns:
                result[fc][secid] = indexed[fc]

    return result


def _build_wide_multi(
    factor_root: str,
    factor_names_by_col: dict[str, str],
    day: str,
) -> dict[str, pd.DataFrame]:
    """
    按 factor_name 分组并行读取单日所有股票文件（ThreadPoolExecutor），
    返回 {factor_col: wide_df (index=SampleTime, columns=SecurityID)}。
    """
    name_to_cols: dict[str, list[str]] = {}
    for fc, fn in factor_names_by_col.items():
        name_to_cols.setdefault(fn, []).append(fc)

    series_by_col: dict[str, dict[str, pd.Series]] = {
        fc: {} for fc in factor_names_by_col
    }

    n_threads = min(len(name_to_cols), 16)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        fut_to_name = {}
        for factor_name, cols in name_to_cols.items():
            day_dir = os.path.join(factor_root, factor_name, day)
            if not os.path.isdir(day_dir):
                continue
            fut = pool.submit(_read_one_factor_dir, day_dir, cols)
            fut_to_name[fut] = factor_name

        for fut in as_completed(fut_to_name):
            partial = fut.result()
            for fc, sdict in partial.items():
                series_by_col[fc].update(sdict)

    return {fc: pd.DataFrame(sdict) for fc, sdict in series_by_col.items() if sdict}


def _build_ret_wide(
    factor_root: str,
    factor_name: str,
    day: str,
) -> dict[str, pd.DataFrame] | None:
    """读取某因子目录下单日所有股票的三条收益率列，返回 {ret_col: wide_df}。"""
    day_dir = os.path.join(factor_root, factor_name, day)
    if not os.path.isdir(day_dir):
        return None

    files = sorted(
        f for f in os.listdir(day_dir)
        if f.endswith(".csv") and not f.startswith("_")
    )
    ret_cols = list(_RET_HORIZONS.values())
    needed   = ["SampleTime", "SecurityID"] + ret_cols

    series: dict[str, dict[str, pd.Series]] = {rc: {} for rc in ret_cols}
    for fname in files:
        try:
            df = pd.read_csv(
                os.path.join(day_dir, fname),
                usecols=needed,
                dtype={"SampleTime": str, "SecurityID": str},
            )
        except ValueError:
            continue
        if df.empty:
            continue
        secid   = df["SecurityID"].iloc[0]
        indexed = df.set_index("SampleTime")
        for rc in ret_cols:
            if rc in indexed.columns:
                series[rc][secid] = indexed[rc]

    if not any(series.values()):
        return None

    return {rc: pd.DataFrame(d) for rc, d in series.items()}


# ── 单日计算 ───────────────────────────────────────────────────────────────────

def _compute_day(
    factor_root: str,
    out_dirs: dict[str, str],
    ic_info_by_horizon: dict[str, dict],
    day: str,
    score_method: str = "rank",
    cache_root: str | None = None,
) -> str:
    """
    单日计算入口。

    执行顺序
    --------
    1. 跳过检查：3个horizon的输出CSV均已存在 → 直接返回
    2. 数据获取（优先缓存）：
       - cache hit  → 读 parquet，reshape 还原宽表（无 pivot）
       - cache miss → 读原始 CSV，构建宽表，写 parquet 缓存
    3. 预计算 score_cache：51 个因子的得分矩阵各算一次，
       供 3 个 ret_horizon 复用（避免重复计算）
    4. 对每个 ret_horizon：IC加权合成分 → 十分位分组 → 写 {date}.csv
    """
    # ── 1. 跳过已完成的日期 ──────────────────────────────────────────────────
    horizons_with_weights = [
        ret_h for ret_h in _RET_HORIZONS
        if ic_info_by_horizon.get(ret_h, {}).get("weights")
    ]
    if horizons_with_weights and all(
        os.path.exists(os.path.join(out_dirs[ret_h], f"{day}.csv"))
        for ret_h in horizons_with_weights
    ):
        return day

    # ── 2. 确定本次需要的因子列 ──────────────────────────────────────────────
    all_fc_to_fn: dict[str, str] = {}
    for info in ic_info_by_horizon.values():
        all_fc_to_fn.update(info["factor_names"])

    if not all_fc_to_fn:
        return day

    ret_cols = list(_RET_HORIZONS.values())

    # ── 3. 数据获取：优先读缓存 ───────────────────────────────────────────────
    cache_result = None
    if cache_root:
        cache_result = load_wide_cache(
            cache_root, day, list(all_fc_to_fn.keys()), ret_cols
        )

    if cache_result is not None:
        wide_factors, ret_wides = cache_result
    else:
        # cache miss：从原始 CSV 构建
        wide_factors = _build_wide_multi(factor_root, all_fc_to_fn, day)

        ret_wides = None
        for fn in dict.fromkeys(all_fc_to_fn.values()):
            ret_wides = _build_ret_wide(factor_root, fn, day)
            if ret_wides is not None:
                break
        if ret_wides is None:
            return day

        # 写 parquet 缓存，供后续同日不同 factor_pool / score_method 复用
        if cache_root:
            save_wide_cache(cache_root, day, wide_factors, ret_wides)

    # ── 4. 预计算因子得分（复用于 3 个 ret_horizon）─────────────────────────
    ref_r_wide = next(
        (ret_wides[rc] for rc in _RET_HORIZONS.values()
         if rc in ret_wides and not ret_wides[rc].empty),
        None,
    )
    score_cache: dict[str, np.ndarray] | None = None
    if ref_r_wide is not None:
        _score_fn = _zscore_scores if score_method == "zscore" else _percentile_scores
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

    # ── 5. 逐 horizon 合成分 → 分组 → 写 CSV ────────────────────────────────
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
        )
        df = df.reset_index()
        df.insert(0, "Date", day)

        out_dir = out_dirs[ret_h]
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{day}.csv"), index=False)

    return day


# ── Worker（供 ProcessPoolExecutor 调用）──────────────────────────────────────

def _worker(args) -> str:
    factor_root, out_dirs, ic_info_by_horizon, day, score_method, cache_root = args
    return _compute_day(
        factor_root, out_dirs, ic_info_by_horizon, day, score_method, cache_root
    )


# ── 批量入口 ───────────────────────────────────────────────────────────────────

def run_multi_factor_quantile(
    factor_root: str,
    eval_root: str,
    ic_stats_root: str,
    threshold: float = 0.02,
    dates: list[str] | None = None,
    max_workers: int | None = None,
    score_method: str = "rank",
    factor_pool: str = "threshold",
    whitelist_path: str | None = None,
    cache_root: str | None = None,
):
    """
    Parameters
    ----------
    factor_root    : 因子数据根目录（如 result/factor）
    eval_root      : 评估结果输出根目录（如 result/eval）
    ic_stats_root  : cs_ic_stats.csv 的根目录（如 result/ic_stats）
    threshold      : IC 筛选阈值，factor_pool="threshold" 时生效
    dates          : 指定日期列表；None 时自动扫描
    max_workers    : 并行进程数；建议 4~8，None 时用 CPU 核数（容易 I/O 饱和）
    score_method   : "rank"（分位数得分）或 "zscore"（Z-score ±3截断）
    factor_pool    : "threshold" / "union" / "intersection"，决定因子集合和输出子目录
    whitelist_path : 白名单 txt 路径；factor_pool != "threshold" 时需提供
    cache_root     : 宽表 parquet 缓存目录；None 时不缓存（每次从原始 CSV 读取）
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
            d for d in os.listdir(scan_dir)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(scan_dir, d))
        )

    base_out = os.path.join(eval_root, "multi_factor_quantile", factor_pool, score_method)
    out_dirs: dict[str, str] = {}
    for ret_h in _RET_HORIZONS:
        d = os.path.join(base_out, ret_h)
        os.makedirs(d, exist_ok=True)
        out_dirs[ret_h] = d
        _save_weights(d, ic_weights[ret_h])

    tasks = [
        (factor_root, out_dirs, ic_weights, day, score_method, cache_root)
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

    for ret_h in _RET_HORIZONS:
        sub_dir = out_dirs[ret_h]
        _build_daily(sub_dir)
        _build_summary(sub_dir)
        _build_cum_tick(sub_dir)
        _build_cum_daily(sub_dir)
        _build_cum_tick_chart(sub_dir)
        _build_intraday_slot_charts(sub_dir)
        print(f"[{ret_h}] 汇总完成：{sub_dir}")

    print(f"多因子分层计算完成：{base_out}")
