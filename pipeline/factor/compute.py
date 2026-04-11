"""
因子计算编排器。

如何添加新因子
--------------
1. 在 pipeline/factor/ 下新建 <name>.py，实现：
       def compute(df: pd.DataFrame) -> pd.DataFrame:
           ...  # 输入单只股票单日 DataFrame，输出只含因子列的 DataFrame
2. 在本文件的 _FACTOR_MAP 里注册：
       from . import <name>
       _FACTOR_MAP = {..., "<name>": <name>}

输入
----
每天读一次 base/{date}.parquet（long format，含 Price/masks/盘口/ret_fwd），
按 SecurityID 分组，将单股 DataFrame 分发给各因子的 compute() 函数。

输出
----
factor/{factor_name}/{date}.parquet（long format）
列：Date, SampleTime, SecurityID, Market, {factor_cols}...
ret_fwd 不写入因子文件，eval 阶段直接从 base parquet 读取。

并行策略
--------
任务粒度为「每天」而非「每股票」：
  worker 接收字符串参数，自行读 parquet、处理所有股票、写输出，
  跨进程仅传递路径字符串，序列化开销趋近于零。
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from . import bap, mom, acc_mom, neg_skew, amp_slice, rigidity, pv_corr, rsrs, oir, ofd
from . import market_state

# ── 注册因子 ──────────────────────────────────────────────────────────────────
_FACTOR_MAP = {
    "bap":          bap,
    "mom":          mom,
    "acc_mom":      acc_mom,
    "neg_skew":     neg_skew,
    "amp_slice":    amp_slice,
    "rigidity":     rigidity,
    "pv_corr":      pv_corr,
    "rsrs":         rsrs,
    "oir":          oir,
    "ofd":          ofd,
    "market_state": market_state,
}


# ── 单股票计算 ────────────────────────────────────────────────────────────────

def _compute_stock(
    day: str, secid, stock_df: pd.DataFrame, factor_name: str
) -> tuple[dict, pd.DataFrame | None]:
    """计算单只股票单日因子值，返回 (summary_dict, output_df)。"""
    try:
        factors = _FACTOR_MAP[factor_name].compute(stock_df)
        meta    = stock_df[["Date", "SampleTime", "SecurityID", "Market"]]
        out     = pd.concat([meta, factors], axis=1)
        summary = {"Date": day, "SecurityID": secid, "Status": "OK", "Rows": len(out)}
        for c in factors.columns:
            summary[f"nnz_{c}"] = int(out[c].notna().sum())
        return summary, out
    except Exception as e:
        return {"Date": day, "SecurityID": secid, "Status": f"FAIL: {e}"}, None


# ── 单日计算（Worker 执行体）─────────────────────────────────────────────────

def _process_day(
    day: str, factor_name: str, base_root: str, factor_out_root: str
) -> list[dict]:
    """
    读取单日 base parquet → 对所有股票计算因子 → 写出 factor parquet。
    返回该日所有股票的 summary 列表。
    跨进程仅接收字符串，无 DataFrame 序列化开销。
    """
    base_path = os.path.join(base_root, f"{day}.parquet")
    if not os.path.exists(base_path):
        return []

    base_df   = pd.read_parquet(base_path)
    summaries: list[dict]          = []
    dfs:       list[pd.DataFrame]  = []

    for secid, sdf in base_df.groupby("SecurityID", sort=True):
        summary, out = _compute_stock(day, secid, sdf.reset_index(drop=True), factor_name)
        summaries.append(summary)
        if out is not None:
            dfs.append(out)

    if dfs:
        (pd.concat(dfs, ignore_index=True)
         .sort_values(["SampleTime", "SecurityID"])
         .reset_index(drop=True)
         .to_parquet(os.path.join(factor_out_root, f"{day}.parquet"), index=False))

    return summaries


def _worker(args) -> list[dict]:
    day, factor_name, base_root, factor_out_root = args
    return _process_day(day, factor_name, base_root, factor_out_root)


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def run_factors(
    base_root: str,
    factor_root: str,
    factor_name: str,
    dates: list | None = None,
    max_workers: int | None = None,
):
    """
    批量计算单个因子。

    Parameters
    ----------
    base_root   : base 数据根目录（含 {date}.parquet 文件）
    factor_root : 输出根目录（因子文件写入 factor_root/{factor_name}/{date}.parquet）
    factor_name : 因子名称，须在 _FACTOR_MAP 中注册
    dates       : 指定日期列表；None 时自动扫描
    max_workers : 并行进程数
    """
    if factor_name not in _FACTOR_MAP:
        raise ValueError(f"未知因子 '{factor_name}'，可选：{list(_FACTOR_MAP)}")

    factor_out_root = os.path.join(factor_root, factor_name)
    os.makedirs(factor_out_root, exist_ok=True)

    if dates is None:
        dates = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(base_root)
            if f.endswith(".parquet") and not f.startswith("_")
            and len(os.path.splitext(f)[0]) == 8
            and os.path.splitext(f)[0].isdigit()
        )

    tasks        = [(day, factor_name, base_root, factor_out_root) for day in dates]
    all_results: list[dict] = []

    if max_workers == 1:
        day_iter = tqdm(tasks, desc=f"factors/{factor_name}") if tqdm else tasks
        for t in day_iter:
            all_results.extend(_worker(t))
    else:
        # 提交所有天，进程池按 max_workers 并发执行
        pool = ProcessPoolExecutor(max_workers=max_workers)
        try:
            futs  = [pool.submit(_worker, t) for t in tasks]
            inner = (
                tqdm(as_completed(futs), total=len(futs), desc=f"factors/{factor_name}")
                if tqdm else as_completed(futs)
            )
            for f in inner:
                all_results.extend(f.result())
        finally:
            for p in pool._processes.values():
                p.terminate()
            pool.shutdown(wait=False)

    summary_path = os.path.join(factor_out_root, "_summary.csv")
    pd.DataFrame(all_results).sort_values(["Date", "SecurityID"]).reset_index(drop=True) \
      .to_csv(summary_path, index=False)
    print(f"因子计算完成，汇总：{summary_path}")
