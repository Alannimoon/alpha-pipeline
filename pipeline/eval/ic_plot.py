"""
IC 画图模块。

读取 ic_stats 输出的汇总统计，为每个因子生成 6 张图：
  - CS-IC × 3 个收益率窗口（ret100 / ret200 / ret300）
  - TS-IC × 3 个收益率窗口

每张图上下两个子图：
  上：IC 均值（ic_mean / rankic_mean）
  下：ICIR   （icir   / rankic_ir）
每个子图各 6 条线：all / am / pm × IC / RankIC

X 轴：因子计算窗口（分钟）

输出
----
result/eval/ic_stats/{factor_name}/cs_ret100.png
result/eval/ic_stats/{factor_name}/cs_ret200.png
result/eval/ic_stats/{factor_name}/cs_ret300.png
result/eval/ic_stats/{factor_name}/ts_ret100.png
result/eval/ic_stats/{factor_name}/ts_ret200.png
result/eval/ic_stats/{factor_name}/ts_ret300.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_RET_HORIZONS = ["ret100", "ret200", "ret300"]

# (session, ic_col, icir_col, color, linestyle, label_prefix)
_LINE_STYLES = [
    ("all", "ic_mean", "icir",       "#1f77b4", "-",  "IC  all"),
    ("all", "rankic_mean", "rankic_ir", "#1f77b4", "--", "RankIC  all"),
    ("am",  "ic_mean", "icir",       "#ff7f0e", "-",  "IC  am"),
    ("am",  "rankic_mean", "rankic_ir", "#ff7f0e", "--", "RankIC  am"),
    ("pm",  "ic_mean", "icir",       "#2ca02c", "-",  "IC  pm"),
    ("pm",  "rankic_mean", "rankic_ir", "#2ca02c", "--", "RankIC  pm"),
]


def _plot_one(stats_df: pd.DataFrame, ret_h: str, ic_type: str, out_path: str, factor_name: str):
    """
    画单张图（上下两子图）：固定 ret_horizon。
    上图：IC 均值；下图：ICIR。
    ic_type: "cs" 或 "ts"
    """
    sub = stats_df[stats_df["ret_horizon"] == ret_h]
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
    ax_ic.set_title(f"{factor_name}  {ic_type.upper()}-IC  {ret_h}")
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
