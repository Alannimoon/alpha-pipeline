"""
多空平均 PnL 汇总脚本。

用法
----
python pipeline/eval/xgb_quantile/pnl_summary.py

输出（result/eval/pnl/）
------------------------
pnl_single.csv      ：单因子各实验设置的多空 avg PnL（bps/tick）
pnl_multi.csv       ：多因子各实验设置的多空 avg PnL（bps/tick）
pnl_xgb.csv         ：XGBoost 各实验设置的多空 avg PnL（bps/tick）
pnl_xgb_val.csv     ：XGBoost 验证集
pnl_xgb_test.csv    ：XGBoost 测试集
pnl_xgb_ms_val.csv  ：XGBoost market_state 验证集
pnl_xgb_ms_test.csv ：XGBoost market_state 测试集
pnl_xgb_slot.csv    ：XGBoost 分时段全量推理
pnl_xgb_slot_val.csv：XGBoost 分时段验证集推理
同时在终端打印三张汇总表。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import config

# ── 工具 ───────────────────────────────────────────────────────────────────────

def _avg_pnl(cum_daily: pd.DataFrame,
             ls_col: str = "long_short",
             n_col:  str = "n_ticks") -> float:
    """
    从 _cum_daily.csv 的最后一行读取累计多空收益和 tick 数，
    返回每 tick 平均 PnL（单位：bps）。

    ls_col / n_col 可指定时段列，如 "ls_0930_1000" / "n_ticks_0930_1000"。
    """
    if cum_daily.empty or ls_col not in cum_daily.columns or n_col not in cum_daily.columns:
        return float("nan")
    last = cum_daily.iloc[-1]
    n    = last[n_col]
    ls   = last[ls_col]
    if pd.isna(n) or pd.isna(ls) or n == 0:
        return float("nan")
    return ls / n * 1e4


# ── 单因子 ─────────────────────────────────────────────────────────────────────

def collect_single(cs_quantile_root: str) -> pd.DataFrame:
    """
    遍历 cs_quantile/{factor_name}/ret{h}/_cum_daily.csv，
    每个 factor_col 取最后一行计算 avg_pnl_bps。

    列：factor_name, factor_col, ret_horizon, avg_pnl_bps
    """
    rows = []
    if not os.path.isdir(cs_quantile_root):
        return pd.DataFrame(rows)

    for factor_name in sorted(os.listdir(cs_quantile_root)):
        fn_dir = os.path.join(cs_quantile_root, factor_name)
        if not os.path.isdir(fn_dir):
            continue
        for ret_h in sorted(os.listdir(fn_dir)):
            rh_dir = os.path.join(fn_dir, ret_h)
            csv    = os.path.join(rh_dir, "_cum_daily.csv")
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv, dtype={"Date": str})
            if df.empty or "factor_col" not in df.columns:
                continue
            for fc, sub in df.groupby("factor_col"):
                sub = sub.sort_values("Date").reset_index(drop=True)
                rows.append({
                    "factor_name":  factor_name,
                    "factor_col":   fc,
                    "ret_horizon":  ret_h,
                    "avg_pnl_bps":  _avg_pnl(sub),
                })

    return (pd.DataFrame(rows)
            .sort_values(["factor_name", "factor_col", "ret_horizon"])
            .reset_index(drop=True))


# ── 多因子 ─────────────────────────────────────────────────────────────────────

def collect_multi(mfq_root: str) -> pd.DataFrame:
    """
    遍历 multi_factor_quantile/g{n}/{factor_pool}/{score_method}/ret{h}/_cum_daily.csv，
    取最后一行计算 avg_pnl_bps。

    列：n_groups, factor_pool, score_method, ret_horizon, avg_pnl_bps
    """
    rows = []
    if not os.path.isdir(mfq_root):
        return pd.DataFrame(rows)

    for g_dir in sorted(os.listdir(mfq_root)):            # g10 / g20 / ...
        if not g_dir.startswith("g"):
            continue
        try:
            n_groups = int(g_dir[1:])
        except ValueError:
            continue
        g_path = os.path.join(mfq_root, g_dir)
        if not os.path.isdir(g_path):
            continue

        for pool in sorted(os.listdir(g_path)):            # threshold / union / intersection
            pool_path = os.path.join(g_path, pool)
            if not os.path.isdir(pool_path):
                continue

            for method in sorted(os.listdir(pool_path)):   # rank / zscore / minmax
                method_path = os.path.join(pool_path, method)
                if not os.path.isdir(method_path):
                    continue

                for ret_h in sorted(os.listdir(method_path)):  # ret100 / ret200 / ret300
                    rh_dir = os.path.join(method_path, ret_h)
                    csv    = os.path.join(rh_dir, "_cum_daily.csv")
                    if not os.path.exists(csv):
                        continue
                    df = pd.read_csv(csv, dtype={"Date": str})
                    df = df.sort_values("Date").reset_index(drop=True)
                    row = {
                        "n_groups":     n_groups,
                        "factor_pool":  pool,
                        "score_method": method,
                        "ret_horizon":  ret_h,
                        "avg_pnl_bps":  _avg_pnl(df),
                    }
                    for lsc in [c for c in df.columns if c.startswith("ls_")]:
                        safe = lsc[3:]
                        row[f"avg_pnl_{safe}"] = _avg_pnl(df, ls_col=lsc, n_col=f"n_ticks_{safe}")
                    rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.sort_values(["n_groups", "factor_pool", "score_method", "ret_horizon"])
              .reset_index(drop=True))


# ── XGBoost 分层 ───────────────────────────────────────────────────────────────

def collect_xgb(xgb_root: str) -> pd.DataFrame:
    """
    遍历 xgb_quantile/{factor_pool}/g{n_groups}/ret{h}/_cum_daily.csv，
    取最后一行计算 avg_pnl_bps。

    列：factor_pool, n_groups, ret_horizon, avg_pnl_bps[, avg_pnl_{slot}...]
    """
    rows = []
    if not os.path.isdir(xgb_root):
        return pd.DataFrame(rows)

    for pool in sorted(os.listdir(xgb_root)):         # all / union / intersection
        pool_path = os.path.join(xgb_root, pool)
        if not os.path.isdir(pool_path):
            continue

        for g_dir in sorted(os.listdir(pool_path)):   # g10 / g20
            if not g_dir.startswith("g"):
                continue
            try:
                n_groups = int(g_dir[1:])
            except ValueError:
                continue
            g_path = os.path.join(pool_path, g_dir)
            if not os.path.isdir(g_path):
                continue

            for ret_h in sorted(os.listdir(g_path)):  # ret100 / ret200 / ret300
                rh_dir = os.path.join(g_path, ret_h)
                csv    = os.path.join(rh_dir, "_cum_daily.csv")
                if not os.path.exists(csv):
                    continue
                df = pd.read_csv(csv, dtype={"Date": str})
                df = df.sort_values("Date").reset_index(drop=True)
                row = {
                    "factor_pool": pool,
                    "n_groups":    n_groups,
                    "ret_horizon": ret_h,
                    "avg_pnl_bps": _avg_pnl(df),
                }
                for lsc in [c for c in df.columns if c.startswith("ls_")]:
                    safe = lsc[3:]
                    row[f"avg_pnl_{safe}"] = _avg_pnl(df, ls_col=lsc, n_col=f"n_ticks_{safe}")
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.sort_values(["factor_pool", "n_groups", "ret_horizon"])
              .reset_index(drop=True))


# ── XGBoost 分时段 ─────────────────────────────────────────────────────────────

def collect_xgb_slot(xgb_slot_root: str) -> pd.DataFrame:
    """
    遍历 xgb_quantile_slot/{slot}/{factor_pool}/g{n_groups}/ret{h}/_cum_daily.csv，
    复用 collect_xgb 逐时段扫描后加 slot 列拼接。

    列：slot, factor_pool, n_groups, ret_horizon, avg_pnl_bps[, avg_pnl_{slot_period}...]
    """
    parts = []
    if not os.path.isdir(xgb_slot_root):
        return pd.DataFrame()

    for slot in sorted(os.listdir(xgb_slot_root)):
        slot_path = os.path.join(xgb_slot_root, slot)
        if not os.path.isdir(slot_path):
            continue
        df = collect_xgb(slot_path)
        if not df.empty:
            df.insert(0, "slot", slot)
            parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main():
    cs_root           = os.path.join(config.EVAL_ROOT, "cs_quantile")
    mfq_root          = os.path.join(config.EVAL_ROOT, "multi_factor_quantile")
    xgb_root          = os.path.join(config.EVAL_ROOT, "xgb_quantile")
    xgb_val_root      = os.path.join(config.EVAL_ROOT, "xgb_quantile_val")
    xgb_test_root     = os.path.join(config.EVAL_ROOT, "xgb_quantile_test")
    xgb_ms_val_root   = os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_val")
    xgb_ms_test_root  = os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_test")
    xgb_msvt_val_root  = os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_vol_turnover_val")
    xgb_msvt_test_root = os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_vol_turnover_test")
    xgb_ms100_test_root  = os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_vol100_test")
    xgb_msvt100_test_root = os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_vol_turnover_vol100_test")
    xgb_ms_test2_root    = os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_test2")
    xgb_msvt_test2_root  = os.path.join(config.EVAL_ROOT, "xgb_quantile_market_state_vol_turnover_test2")
    xgb_slot_root     = os.path.join(config.EVAL_ROOT, "xgb_quantile_slot")
    xgb_slot_val_root = os.path.join(config.EVAL_ROOT, "xgb_quantile_slot_val")

    print("正在扫描单因子结果...")
    single           = collect_single(cs_root)
    print("正在扫描多因子结果...")
    multi            = collect_multi(mfq_root)
    print("正在扫描 XGBoost 全量推理结果...")
    xgb              = collect_xgb(xgb_root)
    print("正在扫描 XGBoost 验证集推理结果...")
    xgb_val          = collect_xgb(xgb_val_root)
    print("正在扫描 XGBoost 测试集推理结果...")
    xgb_test         = collect_xgb(xgb_test_root)
    print("正在扫描 XGBoost market_state 验证集推理结果...")
    xgb_ms_val       = collect_xgb(xgb_ms_val_root)
    print("正在扫描 XGBoost market_state 测试集推理结果...")
    xgb_ms_test      = collect_xgb(xgb_ms_test_root)
    print("正在扫描 XGBoost market_state_vol_turnover 验证集推理结果...")
    xgb_msvt_val     = collect_xgb(xgb_msvt_val_root)
    print("正在扫描 XGBoost market_state_vol_turnover 测试集推理结果...")
    xgb_msvt_test    = collect_xgb(xgb_msvt_test_root)
    print("正在扫描 XGBoost market_state vol100 测试集推理结果...")
    xgb_ms100_test   = collect_xgb(xgb_ms100_test_root)
    print("正在扫描 XGBoost market_state_vol_turnover vol100 测试集推理结果...")
    xgb_msvt100_test = collect_xgb(xgb_msvt100_test_root)
    print("正在扫描 XGBoost market_state test2 推理结果...")
    xgb_ms_test2     = collect_xgb(xgb_ms_test2_root)
    print("正在扫描 XGBoost market_state_vol_turnover test2 推理结果...")
    xgb_msvt_test2   = collect_xgb(xgb_msvt_test2_root)
    print("正在扫描 XGBoost 分时段全量推理结果...")
    xgb_slot         = collect_xgb_slot(xgb_slot_root)
    print("正在扫描 XGBoost 分时段验证集推理结果...")
    xgb_slot_val     = collect_xgb_slot(xgb_slot_val_root)

    # ── 保存 CSV（原有结构，每个目录一个文件）────────────────────────────────
    out_dir = os.path.join(config.EVAL_ROOT, "pnl")
    os.makedirs(out_dir, exist_ok=True)

    single.to_csv(          os.path.join(out_dir, "pnl_single.csv"),            index=False)
    multi.to_csv(           os.path.join(out_dir, "pnl_multi.csv"),             index=False)
    xgb.to_csv(             os.path.join(out_dir, "pnl_xgb.csv"),               index=False)
    xgb_val.to_csv(         os.path.join(out_dir, "pnl_xgb_val.csv"),           index=False)
    xgb_test.to_csv(        os.path.join(out_dir, "pnl_xgb_test.csv"),          index=False)
    xgb_ms_val.to_csv(      os.path.join(out_dir, "pnl_xgb_ms_val.csv"),        index=False)
    xgb_ms_test.to_csv(     os.path.join(out_dir, "pnl_xgb_ms_test.csv"),       index=False)
    xgb_msvt_val.to_csv(    os.path.join(out_dir, "pnl_xgb_msvt_val.csv"),      index=False)
    xgb_msvt_test.to_csv(   os.path.join(out_dir, "pnl_xgb_msvt_test.csv"),     index=False)
    xgb_ms100_test.to_csv(  os.path.join(out_dir, "pnl_xgb_ms100_test.csv"),    index=False)
    xgb_msvt100_test.to_csv(os.path.join(out_dir, "pnl_xgb_msvt100_test.csv"),  index=False)
    xgb_ms_test2.to_csv(    os.path.join(out_dir, "pnl_xgb_ms_test2.csv"),      index=False)
    xgb_msvt_test2.to_csv(  os.path.join(out_dir, "pnl_xgb_msvt_test2.csv"),    index=False)
    xgb_slot.to_csv(        os.path.join(out_dir, "pnl_xgb_slot.csv"),          index=False)
    xgb_slot_val.to_csv(    os.path.join(out_dir, "pnl_xgb_slot_val.csv"),      index=False)

    # ── 18次推理汇总到一个 CSV（带 tag / split 列）───────────────────────────
    new_results = [
        ("market_state",                "val",  xgb_ms_val),
        ("market_state",                "test", xgb_ms_test),
        ("market_state_vol_turnover",   "val",  xgb_msvt_val),
        ("market_state_vol_turnover",   "test", xgb_msvt_test),
        ("market_state_vol100",         "test",  xgb_ms100_test),
        ("market_state_vol_turnover_vol100", "test", xgb_msvt100_test),
        ("market_state",                "test2", xgb_ms_test2),
        ("market_state_vol_turnover",   "test2", xgb_msvt_test2),
    ]
    parts = []
    for tag, split, df in new_results:
        if not df.empty:
            d = df.copy()
            d.insert(0, "split", split)
            d.insert(0, "tag",   tag)
            parts.append(d)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            os.path.join(out_dir, "pnl_xgb_new_features.csv"), index=False
        )

    print(f"\n已保存至 {out_dir}/")

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)

    # ── 打印单因子汇总 ────────────────────────────────────────────────────────
    if not single.empty:
        print("\n=== 单因子多空平均 PnL（bps/tick）===")
        pivot = single.pivot_table(
            index=["factor_name", "factor_col"], columns="ret_horizon",
            values="avg_pnl_bps", aggfunc="first",
        ).round(4)
        col_order = [c for c in ["ret100", "ret200", "ret300"] if c in pivot.columns]
        print(pivot[col_order].to_string())
    else:
        print("\n未找到单因子结果（cs_quantile 目录为空或尚未运行）。")

    # ── 打印多因子汇总 ────────────────────────────────────────────────────────
    if not multi.empty:
        print("\n=== 多因子多空平均 PnL（bps/tick）===")
        multi["experiment"] = ("g" + multi["n_groups"].astype(str)
                               + " | " + multi["factor_pool"]
                               + " | " + multi["score_method"])
        pivot_m = multi.pivot_table(
            index="experiment", columns="ret_horizon",
            values="avg_pnl_bps", aggfunc="first",
        ).round(4)
        col_order = [c for c in ["ret100", "ret200", "ret300"] if c in pivot_m.columns]
        print(pivot_m[col_order].to_string())

    # ── 打印新特征汇总 ────────────────────────────────────────────────────────
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        print("\n=== 新特征实验 多空平均 PnL（bps/tick）===")
        combined["experiment"] = (combined["tag"] + " | " + combined["split"]
                                  + " | " + combined["factor_pool"]
                                  + " | g" + combined["n_groups"].astype(str))
        pivot_new = combined.pivot_table(
            index="experiment", columns="ret_horizon",
            values="avg_pnl_bps", aggfunc="first",
        ).round(4)
        col_order = [c for c in ["ret100", "ret200", "ret300"] if c in pivot_new.columns]
        print(pivot_new[col_order].to_string())

    # ── 分时段汇总 ────────────────────────────────────────────────────────────
    for label, df_slot in [("XGBoost 分时段全量推理", xgb_slot),
                            ("XGBoost 分时段验证集推理", xgb_slot_val)]:
        if not df_slot.empty:
            print(f"\n=== {label} 多空平均 PnL（bps/tick）===")
            df_slot["experiment"] = (df_slot["slot"] + " | " + df_slot["factor_pool"]
                                     + " | g" + df_slot["n_groups"].astype(str))
            pivot_s = df_slot.pivot_table(
                index="experiment", columns="ret_horizon",
                values="avg_pnl_bps", aggfunc="first",
            ).round(4)
            col_order = [c for c in ["ret100", "ret200", "ret300"] if c in pivot_s.columns]
            print(pivot_s[col_order].to_string())


if __name__ == "__main__":
    main()
