#!/usr/bin/env python
"""
stride 降采样分析脚本 —— 仅看测试集 1430 时段

对 baseline 和 market_state 两个设置下各 18 组配置，
在 14:30 时段内以 stride=10 的 10 种 offset（0~9）分别计算
多空 avg_pnl（bps），并与原始 avg_pnl_1430_1457 做一致性校验。

理论保证：10 个 offset 的均值 ≈ stride=1 的全量结果（误差仅来自
每天 tick 数不能被 10 整除时的边界效应，可忽略不计）。

输出（result/eval/pnl/inference/）：
  - stride_analysis.csv
  - stride_range.png

用法：
    python pipeline/eval/xgb_quantile/inference/stride.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

# ── 配置 ──────────────────────────────────────────────────────────────────────
FACTOR_POOLS  = ["union", "intersection", "all"]
N_GROUPS_LIST = [10, 20]
RET_HORIZONS  = ["ret100", "ret200", "ret300"]

SETTINGS = {
    "baseline":     os.path.join(config.EVAL_ROOT, "xgb_quantile_test"),
    "market_state": os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_test"),
}

SLOT_START = "14:30:00"
N_OFFSETS  = 10

OUT_DIR = os.path.join(config.EVAL_ROOT, "pnl", "inference")

# x 轴标签：pool/g{n} 的6种组合
CONFIG_LABELS = [f"{p}/g{n}" for p in FACTOR_POOLS for n in N_GROUPS_LIST]


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_slot_ls(pred_dir: str, n_groups: int) -> pd.DataFrame:
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


# ── 可视化 ────────────────────────────────────────────────────────────────────

def plot_stride_range(result_df: pd.DataFrame, out_path: str) -> None:
    """
    2×3 分面图（行=setting，列=ret_horizon）。
    每个子图：x 轴为 6 个 pool/g{n} 配置，
    竖线 = [offset_min, offset_max]，圆点 = full_mean。
    """
    settings     = ["baseline", "market_state"]
    ret_horizons = ["ret100", "ret200", "ret300"]
    offset_cols  = [f"offset_{k}" for k in range(N_OFFSETS)]

    fig, axes = plt.subplots(
        2, 3, figsize=(14, 8), sharey=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.3},
    )

    for r, setting in enumerate(settings):
        for c, ret_h in enumerate(ret_horizons):
            ax = axes[r][c]
            sub = result_df[
                (result_df["setting"] == setting) &
                (result_df["ret_horizon"] == ret_h)
            ].copy()

            if sub.empty:
                ax.set_visible(False)
                continue

            # 按 CONFIG_LABELS 顺序排列
            sub["cfg"] = sub["factor_pool"] + "/g" + sub["n_groups"].astype(str)
            sub = sub.set_index("cfg").reindex(CONFIG_LABELS).reset_index()

            xs         = np.arange(len(CONFIG_LABELS))
            full_means = sub["full_mean"].values
            mins       = sub[offset_cols].min(axis=1).values
            maxs       = sub[offset_cols].max(axis=1).values

            # 竖线：offset 范围
            for x, lo, hi in zip(xs, mins, maxs):
                ax.vlines(x, lo, hi, color="#4575b4", linewidth=2.5, alpha=0.7)

            # 端点小横线
            ax.hlines(mins, xs - 0.15, xs + 0.15, color="#4575b4", linewidth=1.2, alpha=0.7)
            ax.hlines(maxs, xs - 0.15, xs + 0.15, color="#4575b4", linewidth=1.2, alpha=0.7)

            # full_mean 圆点
            ax.scatter(xs, full_means, color="#d73027", s=40, zorder=5, label="full mean")

            ax.set_xticks(xs)
            ax.set_xticklabels(CONFIG_LABELS, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("avg_pnl (bps)", fontsize=8)
            ax.set_title(f"{setting} | {ret_h}", fontsize=9)
            ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)

            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("stride=10 sampling: offset range vs full mean per config (slot 1430, test set)",
                 fontsize=11, y=1.01)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图表：{out_path}")


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
                    all_rows.append({
                        "setting":     setting,
                        "factor_pool": pool,
                        "n_groups":    n_groups,
                        "ret_horizon": ret_h,
                        **stats,
                    })

    if not all_rows:
        print("\n没有找到任何数据。")
        return

    offset_cols = [f"offset_{k}" for k in range(N_OFFSETS)]
    result_df   = pd.DataFrame(all_rows)
    col_order   = (
        ["setting", "factor_pool", "n_groups", "ret_horizon"]
        + offset_cols
        + ["stride_mean", "full_mean", "n_ticks"]
    )
    result_df = result_df[col_order]

    # ── 保存 CSV ─────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "stride_analysis.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"\n已保存：{csv_path}")

    # ── 图表 ─────────────────────────────────────────────────────────────────
    plot_stride_range(result_df, os.path.join(OUT_DIR, "stride_range.png"))

    # ── 打印汇总 ─────────────────────────────────────────────────────────────
    for s in result_df["setting"].unique():
        sub = result_df[result_df["setting"] == s].copy()
        sub["config"] = (
            sub["factor_pool"] + " | g" + sub["n_groups"].astype(str)
            + " | " + sub["ret_horizon"]
        )
        disp = sub.set_index("config")[offset_cols + ["stride_mean", "full_mean"]]
        print(f"\n{'='*60}")
        print(f"  {s}  —  1430时段 stride=10 各offset avg_pnl (bps)")
        print(f"{'='*60}")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(disp.to_string())

    print("\n=== 一致性校验：stride_mean vs full_mean ===")
    check = result_df[["setting", "factor_pool", "n_groups", "ret_horizon",
                        "stride_mean", "full_mean"]].copy()
    check["diff"] = (check["stride_mean"] - check["full_mean"]).round(4)
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
