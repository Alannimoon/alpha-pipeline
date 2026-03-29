"""
宽表 parquet 缓存模块。

职责
----
把原始"每股票一个 CSV"的碎片文件聚合成每日一个 parquet，
供后续所有分层、IC 计算直接读取，避免重复 I/O。

缓存格式（{cache_root}/{date}.parquet）
---------------------------------------
  行：T × N 行（T≈4740 时刻，N≈500 只股票）
  列：SampleTime, SecurityID,
      <所有 factor_col>,
      ret_fwd_100, ret_fwd_200, ret_fwd_300
  存储顺序：SampleTime 外层（repeat），SecurityID 内层（tile），
            保证 reshape(T, N) 可直接还原宽表，无需 pivot/unstack。
  dtype：float32（节省约 50% 空间）

对外接口
--------
  save_wide_cache(cache_root, day, wide_factors, ret_wides)
      写缓存（由 multi_factor._compute_day 在 cache miss 时调用）

  load_wide_cache(cache_root, day, needed_fc, ret_cols)
      读缓存（cache hit 时返回 wide_factors + ret_wides，否则返回 None）

  run_build_cache(factor_root, cache_root, dates, max_workers)
      独立命令入口：一次性扫描所有因子目录，预构建完整缓存
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 收益率列名（与 factor_score._RET_HORIZONS 保持一致）
_RET_COLS = ["ret_fwd_100", "ret_fwd_200", "ret_fwd_300"]

# 非因子列（读 CSV 头时需排除）
_NON_FACTOR_COLS = {
    "SampleTime", "SecurityID", "Date", "Market",
    "ret_fwd_100", "ret_fwd_200", "ret_fwd_300",
    "has_limit_up", "has_limit_dn", "has_limit_down",
}


# ── 底层读写 ───────────────────────────────────────────────────────────────────

def _cache_path(cache_root: str, day: str) -> str:
    return os.path.join(cache_root, f"{day}.parquet")


def save_wide_cache(
    cache_root: str,
    day: str,
    wide_factors: dict[str, pd.DataFrame],
    ret_wides: dict[str, pd.DataFrame],
) -> None:
    """
    将单日宽表写入 parquet 缓存。

    存储时按"SampleTime 外层 repeat、SecurityID 内层 tile"的行主序排列，
    读取时可直接 reshape(T, N) 还原，无需任何 pivot/unstack。
    """
    ref_df = next(iter(ret_wides.values()))
    T = len(ref_df.index)
    N = len(ref_df.columns)

    data: dict[str, np.ndarray] = {
        "SampleTime": np.repeat(ref_df.index.to_numpy(), N),
        "SecurityID":  np.tile(ref_df.columns.to_numpy(), T),
    }
    for fc, df in wide_factors.items():
        data[fc] = (
            df.reindex(index=ref_df.index, columns=ref_df.columns)
            .to_numpy(np.float32)
            .ravel()
        )
    for rc, df in ret_wides.items():
        data[rc] = (
            df.reindex(index=ref_df.index, columns=ref_df.columns)
            .to_numpy(np.float32)
            .ravel()
        )

    os.makedirs(cache_root, exist_ok=True)
    pd.DataFrame(data).to_parquet(
        _cache_path(cache_root, day), index=False, compression="snappy"
    )


def load_wide_cache(
    cache_root: str,
    day: str,
    needed_fc: list[str],
    ret_cols: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None:
    """
    从缓存读取宽表。

    利用写入时的行主序，直接 reshape(T, N) 还原 wide_df，
    比 pivot/unstack 快一个数量级。

    返回 (wide_factors, ret_wides)；文件不存在或列不完整时返回 None。
    """
    path = _cache_path(cache_root, day)
    if not os.path.exists(path):
        return None

    needed_cols = needed_fc + ret_cols
    try:
        df = pd.read_parquet(
            path, columns=["SampleTime", "SecurityID"] + needed_cols
        )
    except Exception:
        return None

    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        return None  # 缓存列不足（如缓存由更小的因子池构建）

    sample_times = df["SampleTime"].unique()   # 保序，长度 T
    security_ids = df["SecurityID"].unique()   # 保序，长度 N
    T, N = len(sample_times), len(security_ids)

    wide_factors = {
        fc: pd.DataFrame(
            df[fc].to_numpy(np.float64).reshape(T, N),
            index=sample_times,
            columns=security_ids,
        )
        for fc in needed_fc
    }
    ret_wides = {
        rc: pd.DataFrame(
            df[rc].to_numpy(np.float64).reshape(T, N),
            index=sample_times,
            columns=security_ids,
        )
        for rc in ret_cols
    }
    return wide_factors, ret_wides


# ── 因子目录扫描 ───────────────────────────────────────────────────────────────

def discover_factor_cols(factor_root: str) -> dict[str, str]:
    """
    扫描 factor_root 下所有 factor_name 目录，读取第一个可用 CSV 的列名，
    识别 factor_col（排除非因子列）。

    返回 {factor_col: factor_name}，覆盖 factor_root 下全部因子。
    """
    fc_to_fn: dict[str, str] = {}

    for factor_name in sorted(os.listdir(factor_root)):
        fn_dir = os.path.join(factor_root, factor_name)
        if not os.path.isdir(fn_dir):
            continue

        # 找第一个包含 CSV 的 day 目录
        sample_csv = None
        for day in sorted(os.listdir(fn_dir)):
            day_dir = os.path.join(fn_dir, day)
            if not os.path.isdir(day_dir):
                continue
            csvs = [
                f for f in os.listdir(day_dir)
                if f.endswith(".csv") and not f.startswith("_")
            ]
            if csvs:
                sample_csv = os.path.join(day_dir, csvs[0])
                break

        if sample_csv is None:
            continue

        try:
            cols = pd.read_csv(sample_csv, nrows=0).columns.tolist()
        except Exception:
            continue

        for col in cols:
            if col not in _NON_FACTOR_COLS and col not in fc_to_fn:
                fc_to_fn[col] = factor_name

    return fc_to_fn


# ── 单日缓存构建（供并行调用）────────────────────────────────────────────────

def _build_cache_one_day(args: tuple) -> str:
    """
    构建单日缓存的 worker 函数。
    若缓存已存在且列完整则跳过。
    """
    from .multi_factor import _build_wide_multi, _build_ret_wide

    factor_root, cache_root, fc_to_fn, day = args

    # 跳过已存在的完整缓存
    if load_wide_cache(cache_root, day, list(fc_to_fn.keys()), _RET_COLS) is not None:
        return day

    # 从原始 CSV 构建宽表
    wide_factors = _build_wide_multi(factor_root, fc_to_fn, day)

    ret_wides = None
    for fn in dict.fromkeys(fc_to_fn.values()):
        ret_wides = _build_ret_wide(factor_root, fn, day)
        if ret_wides is not None:
            break
    if ret_wides is None:
        return day

    save_wide_cache(cache_root, day, wide_factors, ret_wides)
    return day


# ── 批量入口 ───────────────────────────────────────────────────────────────────

def run_build_cache(
    factor_root: str,
    cache_root: str,
    dates: list[str] | None = None,
    max_workers: int | None = None,
) -> None:
    """
    预构建全量宽表 parquet 缓存。

    扫描 factor_root 下所有因子目录，逐日读取所有因子值 + 收益率，
    合并成一张宽表写入 {cache_root}/{date}.parquet。

    建议在首次运行任何分层/IC 计算前单独执行：
        python run.py build_cache --workers 4

    之后 multi_factor_quantile 的所有 factor_pool / score_method 组合
    均可直接读缓存，不再重复读原始 CSV。

    Parameters
    ----------
    factor_root  : 因子数据根目录（如 result/factor）
    cache_root   : 缓存输出目录（如 result/cache/wide）
    dates        : 指定日期列表；None 时自动扫描所有可用日期
    max_workers  : 并行进程数；建议 4~8
    """
    fc_to_fn = discover_factor_cols(factor_root)
    n_factors = len(fc_to_fn)
    n_dirs    = len(set(fc_to_fn.values()))
    print(f"发现 {n_factors} 个因子列，来自 {n_dirs} 个因子目录")

    if dates is None:
        first_fn = next(iter(set(fc_to_fn.values())), None)
        if first_fn is None:
            print("未发现任何因子，退出。")
            return
        fn_dir = os.path.join(factor_root, first_fn)
        dates = sorted(
            d for d in os.listdir(fn_dir)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(fn_dir, d))
        )

    print(f"共 {len(dates)} 个日期，输出到 {cache_root}")
    tasks = [(factor_root, cache_root, fc_to_fn, day) for day in dates]

    if max_workers == 1:
        day_iter = tqdm(tasks, desc="Build Cache") if tqdm else tasks
        for t in day_iter:
            _build_cache_one_day(t)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs  = [pool.submit(_build_cache_one_day, t) for t in tasks]
            inner = (
                tqdm(as_completed(futs), total=len(futs), desc="Build Cache")
                if tqdm else as_completed(futs)
            )
            for fut in inner:
                fut.result()

    print(f"缓存构建完成：{cache_root}")
