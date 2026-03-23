"""
截面分层模块。

对每个时间点 (Date, SampleTime)，将所有股票按因子值分成 N_GROUPS=5 组
（第1组 = 因子值最低，第5组 = 因子值最高），计算各组的前向收益均值。

分组方式
--------
rank-based 均匀分组：对有效股票做 argsort 排名（无并列），
再用 rank * Q // n 映射到 0~Q-1 组，保证每组股票数尽可能相等。

涨跌停处理
----------
当前实现不对涨跌停做任何处理——涨跌停股票照常参与分组，
其因子值和前向收益均原样使用。
已知简化：涨跌停时因子值可能失真（如 BAP 恒为 ±1），
前向收益也受约束，可能导致极端组收益偏差。
后续如需改进，可在分组前用 {fc}_has_limit 列过滤对应股票。

有效股票数
----------
某时刻有效股票数（因子值非 NaN）< N_GROUPS 时，
该时刻所有组记为 NaN，不参与后续统计。
每列同时输出 n_valid_{fc}，记录该时刻有效股票数，供质量检查使用。

输出
----
result/eval/cs_quantile/{factor_name}/{ret_horizon}_{session}/{day}.csv

列：Date, SampleTime,
    g1_{fc}, g2_{fc}, g3_{fc}, g4_{fc}, g5_{fc}, n_valid_{fc}, ...
每天约 4740 行（session 切片后更少）。
"""

import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ._panel import get_factor_cols

N_GROUPS = 5

_RET_HORIZONS = {
    "ret100": "ret_fwd_100",
    "ret200": "ret_fwd_200",
    "ret300": "ret_fwd_300",
}


# ── 宽表构建（与 cs_ic._build_wide_tables 相同逻辑）────────────────────────────

def _build_wide(
    factor_root: str, factor_name: str, day: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    day_dir = os.path.join(factor_root, factor_name, day)
    files = sorted(
        f for f in os.listdir(day_dir)
        if f.endswith(".csv") and not f.startswith("_")
    )
    if not files:
        return {}, []

    first_cols = pd.read_csv(os.path.join(day_dir, files[0]), nrows=0).columns.tolist()
    factor_cols = get_factor_cols(pd.DataFrame(columns=first_cols), factor_name)
    ret_cols    = list(_RET_HORIZONS.values())
    needed_cols = ["SampleTime", "SecurityID"] + factor_cols + ret_cols

    series_by_col: dict[str, dict[str, pd.Series]] = {
        col: {} for col in factor_cols + ret_cols
    }
    for fname in files:
        df = pd.read_csv(
            os.path.join(day_dir, fname),
            usecols=needed_cols,
            dtype={"SampleTime": str, "SecurityID": str},
        )
        if df.empty:
            continue
        secid   = df["SecurityID"].iloc[0]
        indexed = df.set_index("SampleTime")
        for col in factor_cols + ret_cols:
            series_by_col[col][secid] = indexed[col]

    if not series_by_col[factor_cols[0]]:
        return {}, factor_cols

    wide = {col: pd.DataFrame(d) for col, d in series_by_col.items()}
    return wide, factor_cols


# ── 单时刻分组计算 ──────────────────────────────────────────────────────────────

def _group_returns(
    f_arr: np.ndarray,
    r_arr: np.ndarray,
    Q: int = N_GROUPS,
) -> tuple[list[float], int]:
    """
    对一个时刻的因子向量和收益向量，做截面分组并返回各组收益均值。

    Parameters
    ----------
    f_arr : 因子值（含 NaN），长度 = 股票数
    r_arr : 前向收益（含 NaN），长度 = 股票数
    Q     : 分组数

    Returns
    -------
    group_means : 长度 Q 的列表，g[0] = 最低组，g[Q-1] = 最高组
    n_valid     : 参与分组的有效股票数（因子非 NaN）
    """
    valid = np.isfinite(f_arr)
    n_valid = int(valid.sum())

    if n_valid < Q:
        return [np.nan] * Q, n_valid

    f_valid = f_arr[valid]
    r_valid = r_arr[valid]

    # rank-based 均匀分组：argsort(argsort(x)) 给出 0-indexed 排名
    ranks  = np.argsort(np.argsort(f_valid))        # 0 ~ n_valid-1
    groups = (ranks * Q // n_valid).clip(0, Q - 1)  # 0 ~ Q-1

    group_means = []
    for g in range(Q):
        ret_g = r_valid[groups == g]
        valid_ret = ret_g[np.isfinite(ret_g)]
        group_means.append(float(valid_ret.mean()) if len(valid_ret) > 0 else np.nan)

    return group_means, n_valid


# ── 单日计算 ──────────────────────────────────────────────────────────────────

def _compute_day(
    factor_root: str, factor_name: str, day: str,
) -> dict[str, pd.DataFrame]:
    """
    对单日所有 (ret_horizon, session) 组合计算截面分层结果。
    """
    wide, factor_cols = _build_wide(factor_root, factor_name, day)
    if not wide or not factor_cols:
        return {}

    results: dict[str, pd.DataFrame] = {}

    for h_key, h_col in _RET_HORIZONS.items():
        r_wide = wide[h_col]                      # SampleTime × SecurityID
        times  = r_wide.index.tolist()

        # 对每个因子列，逐时刻计算分组收益
        col_data: dict[str, list] = {}
        for fc in factor_cols:
            f_wide = wide[fc]
            # 以收益率宽表的 index 为准对齐（两者来自同一批文件，应相同）
            f_mat = f_wide.reindex(index=r_wide.index, columns=r_wide.columns).to_numpy(np.float64)
            r_mat = r_wide.to_numpy(np.float64)

            g_lists   = [[] for _ in range(N_GROUPS)]
            n_valids  = []

            for t in range(len(times)):
                gm, nv = _group_returns(f_mat[t], r_mat[t])
                for g in range(N_GROUPS):
                    g_lists[g].append(gm[g])
                n_valids.append(nv)

            for g in range(N_GROUPS):
                col_data[f"g{g + 1}_{fc}"] = g_lists[g]
            col_data[f"n_valid_{fc}"] = n_valids

        base_df = pd.DataFrame(col_data, index=r_wide.index)
        base_df.index.name = "SampleTime"
        base_df = base_df.reset_index()
        base_df.insert(0, "Date", day)

        results[f"{h_key}_all"] = base_df.reset_index(drop=True)
        results[f"{h_key}_am"]  = base_df[
            base_df["SampleTime"] <= "11:29:57"
        ].reset_index(drop=True)
        results[f"{h_key}_pm"]  = base_df[
            base_df["SampleTime"] >= "13:00:00"
        ].reset_index(drop=True)

    return results


# ── 汇总生成 ──────────────────────────────────────────────────────────────────

def _build_summary(csv_dir: str) -> None:
    """
    读取目录下所有日期文件，计算各组跨所有日期和时刻的收益均值，
    写入 _summary.csv（列：factor_col, g1, g2, g3, g4, g5）。
    """
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return

    # 每个文件对所有时刻取均值 → 该日各列的均值
    daily_means = []
    for f in files:
        df = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [c for c in df.columns if re.match(r"g\d+_", c)]
        if g_cols:
            daily_means.append(df[g_cols].mean())

    if not daily_means:
        return

    # 跨日再取均值
    overall = pd.DataFrame(daily_means).mean()

    # 从列名 g{n}_{factor_col} 中拆出 factor_col 和 组号
    fc_data: dict[str, dict[str, float]] = {}
    for col, val in overall.items():
        m = re.match(r"g(\d+)_(.*)", col)
        if not m:
            continue
        g, fc = m.group(1), m.group(2)
        fc_data.setdefault(fc, {})[f"g{g}"] = val

    rows = [{"factor_col": fc, **vals} for fc, vals in fc_data.items()]
    summary = pd.DataFrame(rows)[["factor_col", "g1", "g2", "g3", "g4", "g5"]]
    summary.to_csv(os.path.join(csv_dir, "_summary.csv"), index=False)


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def _worker(args) -> str:
    factor_root, base_dir, factor_name, day = args
    day_results = _compute_day(factor_root, factor_name, day)
    for key, df in day_results.items():
        out_dir = os.path.join(base_dir, key)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{day}.csv"), index=False)
    return day


def run_cs_quantile(
    factor_root: str,
    eval_root: str,
    factor_name: str,
    dates: list[str] | None = None,
    max_workers: int | None = None,
):
    """
    批量计算截面分层。

    Parameters
    ----------
    factor_root : 因子数据根目录
    eval_root   : 评估结果输出根目录
    factor_name : 因子名称，如 "bap"
    dates       : 指定日期列表；None 时自动扫描
    max_workers : 并行进程数；None 表示使用 CPU 核数
    """
    if dates is None:
        factor_day_root = os.path.join(factor_root, factor_name)
        dates = sorted(
            d for d in os.listdir(factor_day_root)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(factor_day_root, d))
        )

    base_dir = os.path.join(eval_root, "cs_quantile", factor_name)
    tasks = [(factor_root, base_dir, factor_name, day) for day in dates]

    if max_workers == 1:
        day_iter = tqdm(tasks, desc="CS-Quantile") if tqdm else tasks
        for t in day_iter:
            _worker(t)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs  = [pool.submit(_worker, t) for t in tasks]
            inner = tqdm(as_completed(futs), total=len(futs), desc="CS-Quantile") \
                    if tqdm else as_completed(futs)
            for f in inner:
                f.result()

    # 自动生成每个子目录的汇总文件
    for h_key in _RET_HORIZONS:
        for sess in ("all", "am", "pm"):
            _build_summary(os.path.join(base_dir, f"{h_key}_{sess}"))

    print(f"截面分层计算完成：{base_dir}")
