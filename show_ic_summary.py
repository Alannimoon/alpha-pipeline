"""
CS IC 汇总表（筛选候选表）

用法
----
python show_ic_summary.py                  # 默认阈值 0.02，输出到终端 + CSV
python show_ic_summary.py --threshold 0.015
python show_ic_summary.py --no-csv         # 只打印，不保存文件
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── 参数 ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.02,
                    help="IC 均值绝对值筛选阈值（默认 0.02）")
parser.add_argument("--no-csv", action="store_true",
                    help="不保存 CSV 文件")
args = parser.parse_args()

THRESHOLD = args.threshold

# ── 读取所有因子的 cs_ic_stats ────────────────────────────────────────────────

csv_files = glob.glob(os.path.join(config.IC_STATS_ROOT, "*", "cs_ic_stats.csv"))
if not csv_files:
    print(f"[错误] 未找到 cs_ic_stats.csv：{config.IC_STATS_ROOT}")
    sys.exit(1)

dfs = []
for path in sorted(csv_files):
    factor_name = os.path.basename(os.path.dirname(path))
    df = pd.read_csv(path)
    df.insert(0, "factor_name", factor_name)
    dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)
data = raw[raw["session"] == "all"][
    ["factor_name", "factor_col", "ret_horizon", "ic_mean", "rankic_mean"]
].copy()

# 排序：先按 ret_horizon，再按 |ic_mean| 降序
data["_abs_ic"] = data["ic_mean"].abs()
data = (
    data
    .sort_values(["ret_horizon", "_abs_ic"], ascending=[True, False])
    .drop(columns="_abs_ic")
    .reset_index(drop=True)
)

# ── 计算单调性得分 ─────────────────────────────────────────────────────────────
# 从 cs_quantile 的 _cum_daily.csv 最后一行读取 g1~g5 全年累计收益，
# 单调性得分 = (g1 - g5) / (g2 - g4)
# 正值表示分组单调（无论因子方向），绝对值越大越单调。

def _monotonicity(factor_name: str, factor_col: str, ret_horizon: str) -> float:
    path = os.path.join(
        config.EVAL_ROOT, "cs_quantile", factor_name,
        f"{ret_horizon}_all", "_cum_daily.csv",
    )
    if not os.path.exists(path):
        return float("nan")
    df = pd.read_csv(path, dtype={"Date": str})
    sub = df[df["factor_col"] == factor_col] if "factor_col" in df.columns else df
    if sub.empty:
        return float("nan")
    last = sub.iloc[-1]
    g1, g2, g4, g5 = last.get("g1"), last.get("g2"), last.get("g4"), last.get("g5")
    if any(pd.isna(v) for v in (g1, g2, g4, g5)):
        return float("nan")
    denom = float(g2) - float(g4)
    if abs(denom) < 1e-12:
        return float("nan")
    return (float(g1) - float(g5)) / denom

print("计算单调性得分...", end=" ", flush=True)
data["monotonicity"] = [
    _monotonicity(r.factor_name, r.factor_col, r.ret_horizon)
    for r in data.itertuples()
]
print("完成")

# ── 格式化输出 ─────────────────────────────────────────────────────────────────

PASS_MARK = "✓"

display = data.copy()
display["ic_mean"]     = display["ic_mean"].map("{:+.4f}".format)
display["rankic_mean"] = display["rankic_mean"].map("{:+.4f}".format)
display["monotonicity"] = display["monotonicity"].map(
    lambda v: f"{v:+.3f}" if pd.notna(v) else ""
)
display["pass"] = (data["ic_mean"].abs() >= THRESHOLD).map(
    lambda x: PASS_MARK if x else ""
)
display = display.rename(columns={
    "factor_name":  "因子类",
    "factor_col":   "因子列",
    "ret_horizon":  "收益率窗口",
    "ic_mean":      "IC均值",
    "rankic_mean":  "RankIC均值",
    "monotonicity": "单调性",
    "pass":         f"|IC|≥{THRESHOLD}",
})

for ret_h, grp in display.groupby("收益率窗口", sort=True):
    pass_n  = (grp[f"|IC|≥{THRESHOLD}"] == PASS_MARK).sum()
    total_n = len(grp)
    print(f"\n{'═'*80}")
    print(f"  {ret_h}   （{pass_n}/{total_n} 个因子通过 |IC均值| ≥ {THRESHOLD}）")
    print(f"{'═'*80}")
    print(grp.drop(columns="收益率窗口").to_string(index=False))

print()

# ── 保存 CSV ─────────────────────────────────────────────────────────────────

if not args.no_csv:
    out = data.copy()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"ic_summary_thr{THRESHOLD}.csv",
    )
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已保存：{out_path}")
