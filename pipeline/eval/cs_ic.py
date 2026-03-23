"""
截面 IC 计算模块。

对每个时间点 (Date, SampleTime)，在所有股票截面上计算因子值与未来收益率的
Pearson（IC）和 Spearman（RankIC）相关系数。

算法说明
--------
逐股票读取因子文件，直接将每列组装成宽表（行=时间点，列=股票），
避免先 concat 成 240万行长表再 unstack 的开销。

具体步骤：
  1. 读每只股票的文件，取 SampleTime 为 index，只加载因子列和收益率列
  2. 以 {SecurityID: Series} 字典形式积累，pd.DataFrame(dict) 直接得到宽表
  3. 对每个 (factor_col, ret_horizon) 组合，在宽表上按行（axis=1）算相关
  4. 按 session 切行输出（IC 只算一遍，不重复计算）

输出
----
result/eval/cs_ic/{factor_name}/{ret_horizon}_{session}/{day}.csv

例：cs_ic/bap/ret100_all/20250102.csv
    cs_ic/bap/ret100_am/20250102.csv
    ...（共 3 horizon × 3 session = 9 个子目录）

列：Date, SampleTime,
    ic_{factor_col}, rankic_{factor_col}, ...
    （每个因子窗口 2 列，共 5 窗口 × 2 = 10 列 + 2 个索引列）

Session 说明
------------
all : 全天所有时间点
am  : SampleTime <= "11:29:57"
pm  : SampleTime >= "13:00:00"

NaN 处理
--------
某时间点若因子或收益率均为 NaN（股票数量不足），IC 自然为 NaN，
不做主动过滤，保持时序完整。
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ._panel import get_factor_cols, compute_ic_pair

_RET_HORIZONS = {
    "ret100": "ret_fwd_100",
    "ret200": "ret_fwd_200",
    "ret300": "ret_fwd_300",
}


def _build_wide_tables(
    factor_root: str,
    factor_name: str,
    day: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    逐文件读取，直接构建宽表，返回 (wide, factor_cols)。

    wide: {列名: DataFrame(行=SampleTime, 列=SecurityID)}
    跳过 _has_limit 列和其他元数据列，只加载计算 IC 所需的列。
    """
    day_dir = os.path.join(factor_root, factor_name, day)
    files = sorted(
        f for f in os.listdir(day_dir)
        if f.endswith(".csv") and not f.startswith("_")
    )
    if not files:
        return {}, []

    # 读第一个文件的列头，确定因子列名
    first_cols = pd.read_csv(
        os.path.join(day_dir, files[0]), nrows=0,
    ).columns.tolist()
    factor_cols = get_factor_cols(pd.DataFrame(columns=first_cols), factor_name)
    ret_cols    = list(_RET_HORIZONS.values())
    needed_cols = ["SampleTime", "SecurityID"] + factor_cols + ret_cols

    # 逐文件积累：{col: {secid: Series(SampleTime → value)}}
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

    # 一步组装宽表：pd.DataFrame(dict_of_series) 按 SampleTime 对齐
    wide = {col: pd.DataFrame(d) for col, d in series_by_col.items()}
    return wide, factor_cols


def _compute_day(
    factor_root: str,
    factor_name: str,
    day: str,
) -> dict[str, pd.DataFrame]:
    """
    对单日数据计算所有 (ret_horizon, session) 组合的 CS-IC。

    IC 只对全天宽表算一次，再按 SampleTime 切行得到 am / pm 子集。
    返回 {"{ret_horizon}_{session}": DataFrame}。
    """
    wide, factor_cols = _build_wide_tables(factor_root, factor_name, day)
    if not wide or not factor_cols:
        return {}

    results: dict[str, pd.DataFrame] = {}

    for h_key, h_col in _RET_HORIZONS.items():
        r_wide = wide[h_col]

        # 计算全天 IC（axis=1：每行跨所有股票做相关）
        ic_cols: dict[str, pd.Series] = {}
        for fc in factor_cols:
            ic, rankic          = compute_ic_pair(wide[fc], r_wide, axis=1)
            ic_cols[f"ic_{fc}"]     = ic
            ic_cols[f"rankic_{fc}"] = rankic

        # 组装结果 DataFrame，index 是 SampleTime
        base_df = pd.DataFrame(ic_cols)
        base_df.index.name = "SampleTime"
        base_df = base_df.reset_index()
        base_df.insert(0, "Date", day)

        # 按 session 切行（IC 值已算好，只是筛选行）
        results[f"{h_key}_all"] = base_df.reset_index(drop=True)
        results[f"{h_key}_am"]  = base_df[
            base_df["SampleTime"] <= "11:29:57"
        ].reset_index(drop=True)
        results[f"{h_key}_pm"]  = base_df[
            base_df["SampleTime"] >= "13:00:00"
        ].reset_index(drop=True)

    return results


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def _worker(args) -> str:
    """ProcessPoolExecutor worker，处理单日 CS-IC 并写文件。"""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    factor_root, base_dir, factor_name, day = args
    day_results = _compute_day(factor_root, factor_name, day)
    for key, df in day_results.items():
        out_dir = os.path.join(base_dir, key)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{day}.csv"), index=False)
    return day


def run_cs_ic(
    factor_root: str,
    eval_root: str,
    factor_name: str,
    dates: list[str] | None = None,
    max_workers: int | None = None,
):
    """
    批量计算截面 IC。

    Parameters
    ----------
    factor_root : 因子数据根目录（含各日期子目录）
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

    base_dir = os.path.join(eval_root, "cs_ic", factor_name)
    tasks = [(factor_root, base_dir, factor_name, day) for day in dates]

    if max_workers == 1:
        day_iter = tqdm(tasks, desc="CS-IC") if tqdm else tasks
        for t in day_iter:
            _worker(t)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_worker, t) for t in tasks]
            inner = tqdm(as_completed(futs), total=len(futs), desc="CS-IC") \
                    if tqdm else as_completed(futs)
            for f in inner:
                f.result()

    print(f"CS-IC 计算完成：{base_dir}")
