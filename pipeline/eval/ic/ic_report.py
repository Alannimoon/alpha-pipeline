"""
IC 统计与画图模块。

统计逻辑
--------
读取 cs_ic/{factor}/ret{100|200|300}/{day}.parquet（全天文件），
按 session 过滤 SampleTime 后：
  1. 每天：对该 session 内所有时刻的 IC 取均值 → 日度 IC
  2. 跨日：对日度 IC 取 mean / std(ddof=1) / ICIR

Session 定义
-----------
all : 全天（不过滤）
am  : SampleTime <= "11:29:57"
pm  : SampleTime >= "13:00:00"

画图
----
每个因子生成 3 张图（cs_ret100 / cs_ret200 / cs_ret300），
每张图上下两子图：IC 均值 / ICIR，各 6 条线（all/am/pm × IC/RankIC）。

输出
----
result/eval/ic_stats/{factor_name}/cs_ic_stats.csv
result/eval/ic_stats/{factor_name}/cs_ret100.png
result/eval/ic_stats/{factor_name}/cs_ret200.png
result/eval/ic_stats/{factor_name}/cs_ret300.png
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RET_HORIZONS = ["ret100", "ret200", "ret300"]
_SESSIONS = ["all", "am", "pm"]

_AM_END = "11:29:57"
_PM_START = "13:00:00"

# (session, ic_col, icir_col, color, linestyle, label)
_LINE_STYLES = [
    ("all", "ic_mean", "icir",        "#1f77b4", "-",  "IC  all"),
    ("all", "rankic_mean", "rankic_ir", "#1f77b4", "--", "RankIC  all"),
    ("am",  "ic_mean", "icir",        "#ff7f0e", "-",  "IC  am"),
    ("am",  "rankic_mean", "rankic_ir", "#ff7f0e", "--", "RankIC  am"),
    ("pm",  "ic_mean", "icir",        "#2ca02c", "-",  "IC  pm"),
    ("pm",  "rankic_mean", "rankic_ir", "#2ca02c", "--", "RankIC  pm"),
]


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _parse_window(fc: str, factor_name: str) -> int:
    """从因子列名提取窗口数值（用于排序）。"""
    suffix = fc[len(factor_name) + 1:]
    m = re.search(r'\d+', suffix)
    return int(m.group()) if m else 0


def _session_mask(df: pd.DataFrame, session: str) -> pd.Series:
    """返回对应 session 的行布尔掩码。"""
    t = df["SampleTime"]
    if session == "am":
        return t <= _AM_END
    if session == "pm":
        return t >= _PM_START
    return pd.Series(True, index=df.index)


# ── 统计计算 ───────────────────────────────────────────────────────────────────

def _cs_stats_one(
    parquet_dir: str,
    factor_cols: list[str],
    session: str,
) -> dict[str, dict]:
    """
    读取某个 ret_horizon 目录下所有日期文件，
    按 session 过滤后计算各因子列的统计量。
    """
    files = sorted(
        f for f in glob.glob(os.path.join(parquet_dir, "*.parquet"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return {}

    daily_ic     = {fc: [] for fc in factor_cols}
    daily_rankic = {fc: [] for fc in factor_cols}

    for f in files:
        df   = pd.read_parquet(f)
        mask = _session_mask(df, session)
        sub  = df[mask]
        if sub.empty:
            continue
        for fc in factor_cols:
            ic_col  = f"ic_{fc}"
            ric_col = f"rankic_{fc}"
            if ic_col in sub.columns:
                daily_ic[fc].append(sub[ic_col].mean())
            if ric_col in sub.columns:
                daily_rankic[fc].append(sub[ric_col].mean())

    rows: dict[str, dict] = {}
    for fc in factor_cols:
        ic_arr  = np.array(daily_ic[fc],     dtype=np.float64)
        ric_arr = np.array(daily_rankic[fc], dtype=np.float64)

        ic_mean     = np.nanmean(ic_arr)
        ic_std      = np.nanstd(ic_arr,  ddof=1) if len(ic_arr)  > 1 else np.nan
        ric_mean    = np.nanmean(ric_arr)
        ric_std     = np.nanstd(ric_arr, ddof=1) if len(ric_arr) > 1 else np.nan

        rows[fc] = {
            "ic_mean":    ic_mean,
            "rankic_mean": ric_mean,
            "ic_std":     ic_std,
            "rankic_std": ric_std,
            "icir":      ic_mean / ic_std if (not np.isnan(ic_std) and ic_std > 1e-12) else np.nan,
            "rankic_ir": ric_mean / ric_std if (not np.isnan(ric_std) and ric_std > 1e-12) else np.nan,
            "n_days":    len(ic_arr),
        }
    return rows


def compute_cs_stats(eval_root: str, factor_name: str) -> pd.DataFrame:
    """汇总所有 (ret_horizon, session) 组合的 IC 统计。"""
    base_dir = os.path.join(eval_root, "cs_ic", factor_name)

    # 从第一个可用文件推断因子列名
    first_file = next(
        (
            files[0]
            for h in _RET_HORIZONS
            for files in [
                sorted(
                    f for f in glob.glob(os.path.join(base_dir, h, "*.parquet"))
                    if not os.path.basename(f).startswith("_")
                )
            ]
            if files
        ),
        None,
    )
    if first_file is None:
        raise FileNotFoundError(f"cs_ic 结果目录为空：{base_dir}")

    sample_df   = pd.read_parquet(first_file)
    factor_cols = [c[3:] for c in sample_df.columns if c.startswith("ic_")]

    records = []
    for ret_h in _RET_HORIZONS:
        parquet_dir = os.path.join(base_dir, ret_h)
        for sess in _SESSIONS:
            stats = _cs_stats_one(parquet_dir, factor_cols, sess)
            for fc, s in stats.items():
                window = _parse_window(fc, factor_name)
                records.append({
                    "ret_horizon":  ret_h,
                    "session":      sess,
                    "factor_window": window,
                    "factor_col":   fc,
                    **s,
                })

    return (
        pd.DataFrame(records)
        .sort_values(["ret_horizon", "session", "factor_window"])
        .reset_index(drop=True)
    )


# ── 画图 ───────────────────────────────────────────────────────────────────────

def _plot_one(
    stats_df: pd.DataFrame,
    ret_h: str,
    out_path: str,
    factor_name: str,
) -> None:
    """画单张图（上：IC 均值；下：ICIR），固定 ret_horizon。"""
    sub     = stats_df[stats_df["ret_horizon"] == ret_h]
    windows = sorted(sub["factor_window"].unique())

    fig, (ax_ic, ax_ir) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for sess, ic_col, ir_col, color, ls, label in _LINE_STYLES:
        s = sub[sub["session"] == sess].sort_values("factor_window")
        ax_ic.plot(s["factor_window"], s[ic_col],
                   color=color, linestyle=ls, marker="o", label=label)
        if ir_col in s.columns:
            ax_ir.plot(s["factor_window"], s[ir_col],
                       color=color, linestyle=ls, marker="o", label=label)

    ax_ic.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_ic.set_ylabel("IC mean")
    ax_ic.set_title(f"{factor_name}  CS-IC  {ret_h}")
    ax_ic.legend(loc="best", fontsize=8)
    ax_ic.grid(True, alpha=0.3)

    ax_ir.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_ir.set_xticks(windows)
    ax_ir.set_xlabel("Factor window (min)")
    ax_ir.set_ylabel("ICIR")
    ax_ir.legend(loc="best", fontsize=8)
    ax_ir.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def run_ic_report(eval_root: str, factor_name: str):
    """
    计算 IC 统计并生成图表。

    统计结果写入 ic_stats/{factor_name}/cs_ic_stats.csv，
    图表写入 ic_stats/{factor_name}/cs_ret{100|200|300}.png。
    """
    out_dir = os.path.join(eval_root, "ic_stats", factor_name)
    os.makedirs(out_dir, exist_ok=True)

    stats_df  = compute_cs_stats(eval_root, factor_name)
    stats_path = os.path.join(out_dir, "cs_ic_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"CS-IC 统计完成：{stats_path}")

    for ret_h in _RET_HORIZONS:
        out = os.path.join(out_dir, f"cs_{ret_h}.png")
        _plot_one(stats_df, ret_h, out, factor_name)
        print(f"已生成：{out}")
