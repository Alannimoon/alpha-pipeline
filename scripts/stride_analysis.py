#!/usr/bin/env python
"""
stride 降采样分析脚本 —— 仅看测试集 1430 时段

对 baseline 和 market_state 两个设置下各 18 组配置，
在 14:30 时段内以 stride=10 的 10 种 offset（0~9）分别计算
多空 avg_pnl（bps），并与原始 avg_pnl_1430_1457 做一致性校验。

理论保证：10 个 offset 的均值 ≈ stride=1 的全量结果（误差仅来自
每天 tick 数不能被 10 整除时的边界效应，可忽略不计）。

用法：
    python scripts/stride_analysis.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

# ── 配置 ──────────────────────────────────────────────────────────────────────
FACTOR_POOLS  = ["union", "intersection", "all"]
N_GROUPS_LIST = [10, 20]
RET_HORIZONS  = ["ret100", "ret200", "ret300"]

# ret_horizon → 持仓 tick 数
HOLD_TICKS = {"ret100": 100, "ret200": 200, "ret300": 300}

# 两种实验设置
SETTINGS = {
    "baseline":     os.path.join(config.EVAL_ROOT, "xgb_quantile_test"),
    "market_state": os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_test"),
}

SLOT_START = "14:30:00"
STRIDE     = 10
N_OFFSETS  = 10   # offset 0 ~ 9


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def load_slot_ls(pred_dir: str, n_groups: int) -> pd.DataFrame:
    """
    读取该配置下所有 {date}.parquet，筛选 SampleTime >= SLOT_START，
    计算多空价差 ls = g{n_groups} - g1，丢弃 NaN（末尾无法计算收益的 tick）。

    返回 DataFrame：columns = [Date, SampleTime, ls]，按 (Date, SampleTime) 排序。
    """
    g_last = f"g{n_groups}"
    parts  = []

    for p in sorted(Path(pred_dir).glob("????????.parquet")):
        try:
            df = pd.read_parquet(p, columns=["Date", "SampleTime", "g1", g_last])
        except Exception as e:
            print(f"  [WARN] 读取 {p.name} 失败: {e}")
            continue

        df = df[df["SampleTime"] >= SLOT_START].copy()
        if df.empty:
            continue

        df["ls"] = df[g_last] - df["g1"]
        df = df.dropna(subset=["ls"])
        if df.empty:
            continue

        parts.append(df[["Date", "SampleTime", "ls"]])

    if not parts:
        return pd.DataFrame(columns=["Date", "SampleTime", "ls"])

    return (pd.concat(parts, ignore_index=True)
              .sort_values(["Date", "SampleTime"])
              .reset_index(drop=True))


def compute_stride_stats(df: pd.DataFrame) -> dict:
    """
    对已按 (Date, SampleTime) 排序的 DataFrame，
    在每天内对 1430 时段内的 tick 从 0 开始编号，
    计算 stride=10 的 N_OFFSETS 种 offset 各自的 avg_pnl（bps），
    以及它们的均值和全量均值（用于一致性校验）。

    返回 dict：
        offset_0 ~ offset_9  : 各 offset 的 avg_pnl (bps)
        stride_mean          : 10 个 offset 均值
        full_mean            : stride=1 全量均值（与 avg_pnl_1430_1457 对齐）
        n_ticks              : 全量有效 tick 数
    """
    # 每天内对时段 tick 编序号
    df = df.copy()
    df["tick_idx"] = df.groupby("Date").cumcount()
    df["offset"]   = df["tick_idx"] % N_OFFSETS

    result = {}
    offset_means = []
    for k in range(N_OFFSETS):
        sub  = df[df["offset"] == k]["ls"]
        mean = float(sub.mean()) * 1e4 if len(sub) > 0 else np.nan
        result[f"offset_{k}"] = round(mean, 4)
        offset_means.append(mean)

    result["stride_mean"] = round(float(np.nanmean(offset_means)), 4)
    result["full_mean"]   = round(float(df["ls"].mean()) * 1e4, 4)
    result["n_ticks"]     = len(df)
    return result


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    all_rows = []

    for setting, pred_root in SETTINGS.items():
        if not os.path.isdir(pred_root):
            print(f"[SKIP] 目录不存在：{pred_root}")
            continue

        print(f"\n{'='*60}")
        print(f"  设置：{setting}")
        print(f"{'='*60}")

        for pool in FACTOR_POOLS:
            for n_groups in N_GROUPS_LIST:
                for ret_h in RET_HORIZONS:
                    pred_dir = os.path.join(pred_root, pool, f"g{n_groups}", ret_h)
                    tag = f"{pool}/g{n_groups}/{ret_h}"

                    if not os.path.isdir(pred_dir):
                        print(f"  [SKIP] {tag}：目录不存在")
                        continue

                    df = load_slot_ls(pred_dir, n_groups)
                    if df.empty:
                        print(f"  [SKIP] {tag}：无有效数据")
                        continue

                    stats = compute_stride_stats(df)
                    print(
                        f"  {tag:<35}  "
                        f"full={stats['full_mean']:7.4f}  "
                        f"stride_mean={stats['stride_mean']:7.4f}  "
                        f"n={stats['n_ticks']}"
                    )

                    row = {
                        "setting":      setting,
                        "factor_pool":  pool,
                        "n_groups":     n_groups,
                        "ret_horizon":  ret_h,
                        **{k: v for k, v in stats.items()},
                    }
                    all_rows.append(row)

    if not all_rows:
        print("\n没有找到任何数据。")
        return

    result_df = pd.DataFrame(all_rows)

    # ── 排列列顺序 ───────────────────────────────────────────────────────────
    offset_cols = [f"offset_{k}" for k in range(N_OFFSETS)]
    col_order   = (
        ["setting", "factor_pool", "n_groups", "ret_horizon"]
        + offset_cols
        + ["stride_mean", "full_mean", "n_ticks"]
    )
    result_df = result_df[col_order]

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(ROOT, "stride_analysis.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\n已保存：{out_path}")

    # ── 打印汇总（pivot：行=配置，列=offset + 均值）─────────────────────────
    for setting in result_df["setting"].unique():
        sub = result_df[result_df["setting"] == setting].copy()
        sub["config"] = (
            sub["factor_pool"]
            + " | g" + sub["n_groups"].astype(str)
            + " | " + sub["ret_horizon"]
        )
        disp = sub.set_index("config")[offset_cols + ["stride_mean", "full_mean"]]
        print(f"\n{'='*60}")
        print(f"  {setting}  —  1430时段 stride=10 各offset avg_pnl (bps)")
        print(f"{'='*60}")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(disp.to_string())

    # ── 一致性校验摘要 ───────────────────────────────────────────────────────
    print("\n=== 一致性校验：stride_mean vs full_mean（两列应接近相等）===")
    check = result_df[["setting", "factor_pool", "n_groups", "ret_horizon",
                        "stride_mean", "full_mean"]].copy()
    check["diff"] = (check["stride_mean"] - check["full_mean"]).round(4)
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
