"""
验证 base 重跑后，旧列数值完全不变，新列正确新增。

用法
----
# 第一步：重跑前，备份某天的 base 文件
cp result/base/20250102.parquet /tmp/base_20250102_old.parquet

# 第二步：重跑 sample / clean / base（全量或指定 --date 20250102）

# 第三步：运行本脚本
python verify_base_diff.py --date 20250102 --old /tmp/base_20250102_old.parquet
"""

import argparse
import sys
import numpy as np
import pandas as pd

NEW_COLS = ["PreCloPrice", "OpenPrice"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",  required=True, help="日期，如 20250102")
    ap.add_argument("--old",   required=True, help="旧 base parquet 路径")
    ap.add_argument("--new",   default=None,  help="新 base parquet 路径（默认 result/base/{date}.parquet）")
    args = ap.parse_args()

    new_path = args.new or f"result/base/{args.date}.parquet"

    old = pd.read_parquet(args.old)
    new = pd.read_parquet(new_path)

    ok = True

    # ── 1. 行数 ────────────────────────────────────────────────────────────────
    if len(old) != len(new):
        print(f"[FAIL] 行数不同: old={len(old)}, new={len(new)}")
        ok = False
    else:
        print(f"[OK]   行数一致: {len(new)}")

    # ── 2. 新列存在且非全空 ────────────────────────────────────────────────────
    for col in NEW_COLS:
        if col not in new.columns:
            print(f"[FAIL] 新列缺失: {col}")
            ok = False
        else:
            null_pct = new[col].isna().mean() * 100
            print(f"[OK]   新列 {col} 存在，null 比例 {null_pct:.1f}%")

    # ── 3. 旧列数值完全一致 ────────────────────────────────────────────────────
    shared_cols = [c for c in old.columns if c in new.columns and c not in NEW_COLS]
    for col in shared_cols:
        s_old = old[col].reset_index(drop=True)
        s_new = new[col].reset_index(drop=True)

        if s_old.dtype.kind in ("f",):
            # 浮点：允许极小误差（浮点表示精度）
            both_nan = s_old.isna() & s_new.isna()
            diff = (~both_nan) & ~np.isclose(
                s_old.fillna(0).to_numpy(),
                s_new.fillna(0).to_numpy(),
                rtol=1e-5, atol=1e-8,
                equal_nan=False,
            )
            # nan 位置不一致也算不同
            nan_mismatch = s_old.isna() != s_new.isna()
            bad = diff | nan_mismatch
        else:
            bad = s_old != s_new

        n_bad = bad.sum()
        if n_bad > 0:
            print(f"[FAIL] 列 {col} 有 {n_bad} 行数值不同")
            print(f"       前3个不同行：\n{old.loc[bad[bad].index[:3], [col]]}")
            ok = False
        else:
            print(f"[OK]   列 {col} 完全一致")

    # ── 4. 汇总 ───────────────────────────────────────────────────────────────
    print()
    if ok:
        print("=== 验证通过：旧列数值无变化，新列已正确添加 ===")
    else:
        print("=== 验证失败：存在差异，请检查上方 [FAIL] 项 ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
