"""
XGBoost 截面分层 — 数据集构建模块。

split_dates              : 训练 / 验证日期划分（固定种子，可复现）
get_factor_cols_for_pool : 按因子池返回 {factor_col: factor_name}
load_day_data            : 读取单日特征 + 收益 + 标注（长格式 DataFrame）
build_dataset            : 批量读取多日，拼接为训练/验证 DataFrame
"""

import glob
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


_VAL_MONTHS         = {"11", "12"}   # 11、12 月全部进验证集
_RAND_VAL_PER_MONTH = 2              # 01-10 月每月随机抽 N 天进验证集
_RANDOM_SEED        = 42

_RET_HORIZONS = {
    "ret100": "ret_fwd_100",
    "ret200": "ret_fwd_200",
    "ret300": "ret_fwd_300",
}

_WARN_LIMIT = 20
_WARN_COUNTERS: dict[str, int] = defaultdict(int)


def _warn(key: str, message: str) -> None:
    """限制重复 warning 数量，避免刷屏。"""
    _WARN_COUNTERS[key] += 1
    count = _WARN_COUNTERS[key]
    if count <= _WARN_LIMIT:
        print(f"[WARN] {message}")
    elif count == _WARN_LIMIT + 1:
        print(f"[WARN] {key} 后续相同告警已省略 ...")


# ── 日期划分 ───────────────────────────────────────────────────────────────────

def split_dates(
    dates: list[str],
    seed: int = _RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """
    将日期列表划分为训练集和验证集。

    规则
    ----
    - 月份 11、12 → 全部进验证集
    - 月份 01-10 → 每月随机抽 _RAND_VAL_PER_MONTH 天进验证集，其余训练集

    Returns
    -------
    (train_dates, val_dates)，均已排序。
    """
    rng = random.Random(seed)

    by_month: dict[str, list[str]] = defaultdict(list)
    for d in sorted(dates):
        by_month[d[4:6]].append(d)

    train, val = [], []
    for month, month_dates in sorted(by_month.items()):
        if month in _VAL_MONTHS:
            val.extend(month_dates)
        else:
            k = min(_RAND_VAL_PER_MONTH, len(month_dates))
            picked = set(rng.sample(month_dates, k))
            val.extend(sorted(picked))
            train.extend([d for d in month_dates if d not in picked])

    return sorted(train), sorted(val)


# ── 因子列映射 ─────────────────────────────────────────────────────────────────

def _scan_all_factor_cols(factor_root: str) -> dict[str, str]:
    """扫描所有因子目录，返回 {factor_col: factor_name}。"""
    fc_to_fn: dict[str, str] = {}
    skip = {"Date", "SampleTime", "SecurityID", "Market"}
    for factor_name in sorted(os.listdir(factor_root)):
        fdir = os.path.join(factor_root, factor_name)
        if not os.path.isdir(fdir):
            continue
        files = [
            f for f in glob.glob(os.path.join(fdir, "*.parquet"))
            if not os.path.basename(f).startswith("_")
        ]
        if not files:
            continue
        try:
            cols = pd.read_parquet(files[0]).columns.tolist()
            for c in cols:
                if c not in skip:
                    fc_to_fn[c] = factor_name
        except Exception as e:
            _warn(
                "scan_factor_cols",
                f"扫描因子列失败: factor={factor_name}, file={files[0]} | "
                f"{type(e).__name__}: {e}",
            )
    return fc_to_fn


def _load_whitelist(path: str) -> set[str]:
    result: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                result.add(line)
    return result


def get_factor_cols_for_pool(
    factor_root: str,
    factor_pool: str,
    union_path: str | None = None,
    intersection_path: str | None = None,
) -> dict[str, str]:
    """
    返回 {factor_col: factor_name} 映射。

    factor_pool
    -----------
    "all"          → factor_root 下所有列（无白名单过滤）
    "union"        → factor_pool_union.txt 白名单
    "intersection" → factor_pool_intersection.txt 白名单
    """
    all_fc = _scan_all_factor_cols(factor_root)
    if factor_pool == "union" and union_path:
        wl = _load_whitelist(union_path)
        return {fc: fn for fc, fn in all_fc.items() if fc in wl}
    if factor_pool == "intersection" and intersection_path:
        wl = _load_whitelist(intersection_path)
        return {fc: fn for fc, fn in all_fc.items() if fc in wl}
    return all_fc


# ── 截面标注 ───────────────────────────────────────────────────────────────────

def _assign_labels(
    df: pd.DataFrame,
    ret_col: str,
    n_groups: int,
) -> pd.DataFrame:
    """
    对每个 SampleTime 截面按 ret_col 做等量 qcut，生成 label（0..n_groups-1）。

    处理策略
    --------
    - qcut 成功且 n_bins == n_groups  → 直接使用
    - qcut 成功但 n_bins < n_groups   → 线性重映射到 [0, n_groups-1]
    - n_bins < max(2, n_groups//2)    → 该截面丢弃（label=NaN）
    - qcut 失败                        → label=NaN

    说明
    ----
    显式遍历各个 SampleTime 分组，避免 pandas groupby.apply 在未来版本中的
    include_groups 行为变化导致 FutureWarning。
    """
    def _qcut_group(grp: pd.DataFrame) -> pd.DataFrame:
        try:
            raw = pd.qcut(grp[ret_col], q=n_groups, labels=False, duplicates="drop")
        except Exception:
            grp = grp.copy()
            grp["label"] = np.nan
            return grp

        n_bins = int(raw.nunique())
        if n_bins < max(2, n_groups // 2):
            grp = grp.copy()
            grp["label"] = np.nan
            return grp

        grp = grp.copy()
        if n_bins == n_groups:
            grp["label"] = raw.astype(float)
        else:
            # 线性重映射：将实际 bin 均匀拉伸到 [0, n_groups-1]
            uniq = sorted(raw.dropna().unique())
            denom = max(len(uniq) - 1, 1)
            remap = {
                v: int(round(i * (n_groups - 1) / denom))
                for i, v in enumerate(uniq)
            }
            grp["label"] = raw.map(remap).astype(float)
        return grp

    parts: list[pd.DataFrame] = []
    for _, grp in df.groupby("SampleTime", sort=False):
        parts.append(_qcut_group(grp))

    if not parts:
        out = df.copy()
        out["label"] = np.nan
        return out

    return pd.concat(parts, ignore_index=False)


# ── 单日加载 ───────────────────────────────────────────────────────────────────

def load_day_data(
    factor_root: str,
    base_root: str,
    fc_to_fn: dict[str, str],
    day: str,
    ret_col: str,
    n_groups: int = 10,
    stride: int | None = None,
) -> pd.DataFrame | None:
    """
    读取单日数据，返回长格式 DataFrame：
      [Date, SampleTime, SecurityID, *feature_cols, ret_col, label]

    Parameters
    ----------
    stride : 在 SampleTime 维度每隔 stride 个时刻采一个截面。
             None 表示保留全量截面（推理时使用）。

    Notes
    -----
    - 每行是 (SampleTime, SecurityID) 的一个样本
    - ret_col 列保留原始收益值，供推理时计算组均值收益
    - label 列是该截面内的等量分组标签（0..n_groups-1）
    - 因子列含 NaN → XGBoost 作为缺失值处理（无需填充）
    """
    # ── 1. 加载各因子文件 ────────────────────────────────────────────────────
    name_to_cols: dict[str, list[str]] = defaultdict(list)
    for fc, fn in fc_to_fn.items():
        name_to_cols[fn].append(fc)

    factor_dfs: list[pd.DataFrame] = []
    for fn, cols in name_to_cols.items():
        path = os.path.join(factor_root, fn, f"{day}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path, columns=["SampleTime", "SecurityID"] + cols)
            factor_dfs.append(df)
        except Exception as e:
            _warn(
                "read_factor_parquet",
                f"读取因子文件失败: day={day}, factor={fn}, file={path} | "
                f"{type(e).__name__}: {e}",
            )

    if not factor_dfs:
        _warn("empty_factor_dfs", f"day={day} 未读到任何因子文件")
        return None

    # ── 2. 合并各因子长表 ────────────────────────────────────────────────────
    merged = factor_dfs[0]
    for df in factor_dfs[1:]:
        merged = merged.merge(df, on=["SampleTime", "SecurityID"], how="outer")

    # ── 3. 时刻降采样（训练时减少数据量）────────────────────────────────────
    if stride is not None and stride > 1:
        all_times = sorted(merged["SampleTime"].unique())
        kept = set(all_times[::stride])
        merged = merged[merged["SampleTime"].isin(kept)]

    if merged.empty:
        _warn("empty_after_factor_merge", f"day={day} 因子合并后为空")
        return None

    # ── 4. 加载 base 收益率 ──────────────────────────────────────────────────
    base_path = os.path.join(base_root, f"{day}.parquet")
    if not os.path.exists(base_path):
        _warn("missing_base_file", f"day={day} base 文件不存在: {base_path}")
        return None
    try:
        base_df = pd.read_parquet(base_path, columns=["SampleTime", "SecurityID", ret_col])
    except Exception as e:
        _warn(
            "read_base_parquet",
            f"读取 base 文件失败: day={day}, file={base_path} | {type(e).__name__}: {e}",
        )
        return None

    merged = merged.merge(base_df, on=["SampleTime", "SecurityID"], how="inner")
    merged = merged.dropna(subset=[ret_col])
    if merged.empty:
        _warn("empty_after_ret_merge", f"day={day} 与 {ret_col} 合并后为空")
        return None

    # ── 5. 截面标注 ─────────────────────────────────────────────────────────
    merged = _assign_labels(merged, ret_col, n_groups)
    merged = merged.dropna(subset=["label"])
    if merged.empty:
        _warn("empty_after_label", f"day={day} 截面标注后全部被丢弃")
        return None
    merged["label"] = merged["label"].astype(np.int32)
    merged.insert(0, "Date", day)

    # ── 6. 丢弃特征全空的行 ──────────────────────────────────────────────────
    fc_present = [c for c in fc_to_fn if c in merged.columns]
    if not fc_present:
        _warn("no_feature_cols_present", f"day={day} 合并结果中不存在任何特征列")
        return None
    merged = merged[merged[fc_present].notna().any(axis=1)]
    if merged.empty:
        _warn("empty_after_drop_all_nan_features", f"day={day} 去掉全空特征行后为空")
        return None

    keep = ["Date", "SampleTime", "SecurityID"] + fc_present + [ret_col, "label"]
    return merged[keep].reset_index(drop=True)


# ── 并行辅助（顶层函数，可被子进程 pickle）─────────────────────────────────────

def _load_day_args(args: tuple) -> "pd.DataFrame | None":
    """ProcessPoolExecutor.map 的包装：将 tuple 解包后调用 load_day_data。"""
    return load_day_data(*args)


# ── 批量构建 ───────────────────────────────────────────────────────────────────

def build_dataset(
    factor_root: str,
    base_root: str,
    fc_to_fn: dict[str, str],
    dates: list[str],
    ret_col: str,
    n_groups: int = 10,
    stride: int | None = 100,
    verbose: bool = True,
    desc: str | None = None,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    批量加载多日数据，返回完整特征-标注 DataFrame。

    stride=100  → 每天约 47 个截面（非重叠 5 分钟窗口），约 5-6M 行
    stride=None → 全量 4740 截面（推理时使用）

    n_workers > 1 时使用 ProcessPoolExecutor 并行加载各天数据，
    I/O 密集场景下可显著提速（建议 4-16，视磁盘带宽而定）。
    """
    parts: list[pd.DataFrame] = []

    if n_workers > 1:
        args_list = [
            (factor_root, base_root, fc_to_fn, day, ret_col, n_groups, stride)
            for day in dates
        ]
        if verbose:
            print(f"  [{desc or ret_col}] 并行加载 {len(dates)} 天 (n_workers={n_workers})")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for df in executor.map(_load_day_args, args_list):
                if df is not None:
                    parts.append(df)
    else:
        iterable = dates
        if verbose and tqdm is not None:
            iterable = tqdm(dates, desc=desc or f"build[{ret_col}]", dynamic_ncols=True, leave=False)
        for day in iterable:
            df = load_day_data(
                factor_root, base_root, fc_to_fn, day, ret_col, n_groups, stride
            )
            if df is not None:
                parts.append(df)
                if verbose and tqdm is None:
                    print(f"  {day}: {len(df):,} 行")

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
