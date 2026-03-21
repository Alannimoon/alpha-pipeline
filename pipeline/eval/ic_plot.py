"""
IC 画图模块。

读取 ic_stats 输出的汇总统计，为每个因子生成 6 张图：
  - CS-IC × 3 个收益率窗口（ret100 / ret200 / ret300）
  - TS-IC × 3 个收益率窗口

每张图 6 条线：
  IC_all / RankIC_all / IC_am / RankIC_am / IC_pm / RankIC_pm

X 轴：因子计算窗口（分钟）
Y 轴：IC 均值

输出
----
data/eval/ic_stats/{factor_name}/cs_ret100.png
data/eval/ic_stats/{factor_name}/cs_ret200.png
data/eval/ic_stats/{factor_name}/cs_ret300.png
data/eval/ic_stats/{factor_name}/ts_ret100.png
data/eval/ic_stats/{factor_name}/ts_ret200.png
data/eval/ic_stats/{factor_name}/ts_ret300.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_RET_HORIZONS = ["ret100", "ret200", "ret300"]
_SESSIONS     = ["all", "am", "pm"]

# 每条线的样式：(session, metric, color, linestyle, label)
_LINE_STYLES = [
    ("all", "ic_mean",     "#1f77b4", "-",  "IC  all"),
    ("all", "rankic_mean", "#1f77b4", "--", "RankIC  all"),
    ("am",  "ic_mean",     "#ff7f0e", "-",  "IC  am"),
    ("am",  "rankic_mean", "#ff7f0e", "--", "RankIC  am"),
    ("pm",  "ic_mean",     "#2ca02c", "-",  "IC  pm"),
    ("pm",  "rankic_mean", "#2ca02c", "--", "RankIC  pm"),
]


def _plot_one(stats_df: pd.DataFrame, ret_h: str, ic_type: str, out_path: str, factor_name: str):
    """
    画单张图：固定 ret_horizon，展示 6 条线。
    ic_type: "cs" 或 "ts"
    """
    sub = stats_df[stats_df["ret_horizon"] == ret_h]
    windows = sorted(sub["factor_window"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for sess, metric, color, ls, label in _LINE_STYLES:
        s = sub[sub["session"] == sess].sort_values("factor_window")
        ax.plot(s["factor_window"], s[metric],
                color=color, linestyle=ls, marker="o", label=label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(windows)
    ax.set_xlabel("Factor window (min)")
    ax.set_ylabel("IC mean")
    ax.set_title(f"{factor_name}  {ic_type.upper()}-IC  {ret_h}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_ic_plot(eval_root: str, factor_name: str):
    stats_dir = os.path.join(eval_root, "ic_stats", factor_name)

    cs_path = os.path.join(stats_dir, "cs_ic_stats.csv")
    ts_path = os.path.join(stats_dir, "ts_ic_stats.csv")

    if not os.path.exists(cs_path) or not os.path.exists(ts_path):
        raise FileNotFoundError(
            f"统计文件不存在，请先运行 ic_stats。\n  {cs_path}\n  {ts_path}"
        )

    cs_df = pd.read_csv(cs_path)
    ts_df = pd.read_csv(ts_path)

    for ret_h in _RET_HORIZONS:
        # CS
        out = os.path.join(stats_dir, f"cs_{ret_h}.png")
        _plot_one(cs_df, ret_h, "cs", out, factor_name)
        print(f"已生成：{out}")

        # TS
        out = os.path.join(stats_dir, f"ts_{ret_h}.png")
        _plot_one(ts_df, ret_h, "ts", out, factor_name)
        print(f"已生成：{out}")
