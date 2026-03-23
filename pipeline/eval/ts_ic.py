"""
时序 IC 计算模块。

对每只股票 (Date, SecurityID)，在该日时间序列上计算因子值与未来收益率的
Pearson（IC）和 Spearman（RankIC）相关系数。

算法说明
--------
直接读取每只股票的因子文件（每文件约 4800 行），对列做相关系数计算，
无需将所有股票合并成宽表，避免了大量 unstack 操作。

Session 处理
------------
时序 IC 对时间段敏感：AM 和 PM 混合会引入午休跳空噪声。
因此分三组计算：
  - all : 全天（AM + PM）
  - am  : SampleTime <= "11:29:57"
  - pm  : SampleTime >= "13:00:00"

has_limit 处理
--------------
计算前将 {factor_col}_has_limit=True 的 tick 的因子值置 NaN，
使其不参与时序相关系数计算。

输出
----
result/eval/ts_ic/{factor_name}/{ret_horizon}_{session}/{day}.csv

例：ts_ic/bap/ret100_all/20250102.csv
    ts_ic/bap/ret100_am/20250102.csv
    ...（共 3 horizon × 3 session = 9 个子目录）

列：Date, SecurityID,
    ts_ic_{factor_col}, ts_rankic_{factor_col}, ...
    （每个因子窗口 2 列，共 5 窗口 × 2 = 10 列 + 2 个索引列）
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ._panel import get_factor_cols

_RET_HORIZONS = {
    "ret100": "ret_fwd_100",
    "ret200": "ret_fwd_200",
    "ret300": "ret_fwd_300",
}


# ── 1D 相关系数工具 ───────────────────────────────────────────────────────────

def _pearson_1d(f: np.ndarray, r: np.ndarray) -> float:
    """
    对两个 1D 数组计算 Pearson 相关系数，联合排除 NaN。
    有效样本不足 2 个时返回 NaN。
    """
    valid = np.isfinite(f) & np.isfinite(r)
    if valid.sum() < 2:
        return np.nan
    fv = f[valid] - f[valid].mean()
    rv = r[valid] - r[valid].mean()
    denom = np.sqrt((fv ** 2).sum() * (rv ** 2).sum())
    return float((fv * rv).sum() / denom) if denom > 1e-12 else np.nan


def _spearman_1d(f: np.ndarray, r: np.ndarray) -> float:
    """
    对两个 1D 数组计算 Spearman 相关系数（先联合排除 NaN，再各自排名，再 Pearson）。
    有效样本不足 2 个时返回 NaN。
    """
    valid = np.isfinite(f) & np.isfinite(r)
    if valid.sum() < 2:
        return np.nan
    fv, rv = f[valid], r[valid]
    # argsort(argsort(x)) 给出 0-indexed 排名，+1 变为 1-indexed
    f_ranked = (np.argsort(np.argsort(fv)) + 1).astype(np.float64)
    r_ranked = (np.argsort(np.argsort(rv)) + 1).astype(np.float64)
    fd = f_ranked - f_ranked.mean()
    rd = r_ranked - r_ranked.mean()
    denom = np.sqrt((fd ** 2).sum() * (rd ** 2).sum())
    return float((fd * rd).sum() / denom) if denom > 1e-12 else np.nan


# ── 单日计算 ──────────────────────────────────────────────────────────────────

def _compute_day(
    factor_root: str,
    factor_name: str,
    day: str,
) -> dict[str, pd.DataFrame]:
    """
    逐股票文件读取，直接对列计算 TS-IC，返回各 (horizon, session) 的 DataFrame。

    每只股票贡献 9 行（3 horizon × 3 session），每行一个字典追加到对应列表，
    最后一次性转成 DataFrame，避免行拼接开销。
    """
    day_dir = os.path.join(factor_root, factor_name, day)
    files = sorted(
        f for f in os.listdir(day_dir)
        if f.endswith(".csv") and not f.startswith("_")
    )

    # 每个 (horizon, session) 组合积累一个 row 列表
    rows: dict[str, list[dict]] = {
        f"{h}_{s}": [] for h in _RET_HORIZONS for s in ("all", "am", "pm")
    }

    for fname in files:
        df = pd.read_csv(
            os.path.join(day_dir, fname),
            dtype={"Date": str, "SecurityID": str, "SampleTime": str},
        )
        if df.empty:
            continue

        secid = df["SecurityID"].iloc[0]
        factor_cols = get_factor_cols(df, factor_name)
        if not factor_cols:
            continue

        # has_limit masking：将涨跌停 tick 的因子值置 NaN，一次性处理所有列
        for fc in factor_cols:
            limit_col = f"{fc}_has_limit"
            if limit_col in df.columns:
                df[fc] = df[fc].where(~df[limit_col].astype(bool))

        # 按 session 切行（字符串比较，HH:MM:SS 格式下字典序即时间序）
        times = df["SampleTime"]
        sessions = {
            "all": df,
            "am":  df[times <= "11:29:57"],
            "pm":  df[times >= "13:00:00"],
        }

        for h_key, h_col in _RET_HORIZONS.items():
            for sess_name, sub in sessions.items():
                if sub.empty:
                    continue
                r = sub[h_col].to_numpy(dtype=np.float64)
                row: dict = {"Date": day, "SecurityID": secid}
                for fc in factor_cols:
                    f = sub[fc].to_numpy(dtype=np.float64)
                    row[f"ts_ic_{fc}"]     = _pearson_1d(f, r)
                    row[f"ts_rankic_{fc}"] = _spearman_1d(f, r)
                rows[f"{h_key}_{sess_name}"].append(row)

    return {
        key: pd.DataFrame(row_list).sort_values("SecurityID").reset_index(drop=True)
        for key, row_list in rows.items()
        if row_list
    }


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def _worker(args) -> str:
    """ProcessPoolExecutor worker，处理单日 TS-IC 并写文件。"""
    factor_root, base_dir, factor_name, day = args
    day_results = _compute_day(factor_root, factor_name, day)
    for key, df in day_results.items():
        out_dir = os.path.join(base_dir, key)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{day}.csv"), index=False)
    return day


def run_ts_ic(
    factor_root: str,
    eval_root: str,
    factor_name: str,
    dates: list[str] | None = None,
    max_workers: int | None = None,
):
    """
    批量计算时序 IC。

    Parameters
    ----------
    factor_root : 因子数据根目录
    eval_root   : 评估结果输出根目录
    factor_name : 因子名称，如 "bap"
    dates       : 指定日期列表；None 时自动扫描
    max_workers : 并行进程数；None 表示使用 CPU 核数
    """
    if dates is None:
        dates = sorted(
            d for d in os.listdir(factor_root)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(factor_root, d))
        )

    base_dir = os.path.join(eval_root, "ts_ic", factor_name)
    tasks = [(factor_root, base_dir, factor_name, day) for day in dates]

    if max_workers == 1:
        day_iter = tqdm(tasks, desc="TS-IC") if tqdm else tasks
        for t in day_iter:
            _worker(t)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_worker, t) for t in tasks]
            inner = tqdm(as_completed(futs), total=len(futs), desc="TS-IC") \
                    if tqdm else as_completed(futs)
            for f in inner:
                f.result()

    print(f"TS-IC 计算完成：{base_dir}")
