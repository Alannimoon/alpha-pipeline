"""
测试集数据提取脚本。

股票池：config/vol_top100.csv（100只高波动股票）
日期范围：自动扫描 /home/fund/data/ 中 202512xx 和 202601xx 的交易日
输出：
  data/test/{date}/{secid}.parquet     ← extract
  result/test/sampled/                 ← sample
  result/test/cleaned/                 ← clean
  result/test/base/                    ← base
  result/test/factor/                  ← factors

用法：
    python extract_test.py
    python extract_test.py --workers 32
"""

import argparse
import os
import sys

import pandas as pd

import config
from pipeline.extract import (
    list_days,
    load_code_set,
    split_sh_sz,
    run_one_day,
    append_extract_day_summary,
    is_day_done,
)
from pipeline.ingest.sample import run_sample
from pipeline.ingest.clean import run_clean
from pipeline.ingest.base import run_base
from pipeline.factor.compute import run_factors

# ── 路径配置（与主 pipeline 完全隔离）───────────────────────────────────────
RAW_DATA_ROOT   = "/home/fund/data"          # 原始行情 zip 文件
TEST_DATA_DIR   = os.path.join(config.ROOT, "data", "test")   # extract 输出
TEST_SAMPLED    = os.path.join(config.ROOT, "result", "test", "sampled")
TEST_CLEANED    = os.path.join(config.ROOT, "result", "test", "cleaned")
TEST_BASE       = os.path.join(config.ROOT, "result", "test", "base")
TEST_FACTOR     = os.path.join(config.ROOT, "result", "test", "factor")

UNIVERSE_CSV    = os.path.join(config.ROOT, "config", "vol_top100.csv")

ALL_FACTORS = [
    "bap", "mom", "acc_mom", "neg_skew", "amp_slice",
    "rigidity", "pv_corr", "rsrs", "oir", "ofd", "market_state",
]


def main():
    parser = argparse.ArgumentParser(description="测试集数据提取（100只股票，12月+1月）")
    parser.add_argument("--workers", type=int, default=32, help="并行进程数（默认32）")
    parser.add_argument("--force", action="store_true", help="强制重新提取（忽略已完成检查）")
    args = parser.parse_args()

    # ── 读取股票池 ───────────────────────────────────────────────────────────
    universe = load_code_set(UNIVERSE_CSV)
    sh_codes, sz_codes = split_sh_sz(universe)
    print(f"股票池：{len(universe)} 只（SH={len(sh_codes)}, SZ={len(sz_codes)}）")

    # ── 扫描测试日期（202512xx 和 202601xx）──────────────────────────────────
    dates = [
        d for d in list_days(RAW_DATA_ROOT)
        if d.startswith("202512") or d.startswith("202601")
    ]
    if not dates:
        print(f"未在 {RAW_DATA_ROOT} 中找到 202512xx 或 202601xx 的日期目录，请确认数据已就位。")
        sys.exit(1)
    print(f"测试日期：{len(dates)} 个交易日（{dates[0]} ~ {dates[-1]}）")

    # ── 1. Extract ───────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(">>> [1/5] Extract")
    print("="*50)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    done_threshold = int(len(universe) * 0.95)  # 95% 股票提取完即视为已完成
    todo_dates = [
        d for d in dates
        if args.force or not is_day_done(TEST_DATA_DIR, d, "parquet", threshold=done_threshold)
    ]
    print(f"  待提取：{len(todo_dates)} 天，已完成：{len(dates) - len(todo_dates)} 天")

    import argparse as _ap
    extract_args = _ap.Namespace(
        format="parquet",
        encoding="gb18030",
        chunksize=200_000,
        no_progress=False,
        data_root=RAW_DATA_ROOT,
        outdir=TEST_DATA_DIR,
    )
    for i, day in enumerate(todo_dates, 1):
        print(f"\n  [{i}/{len(todo_dates)}] {day}")
        summary = run_one_day(day, extract_args, sh_codes, sz_codes, universe)
        append_extract_day_summary(TEST_DATA_DIR, summary)
        print(f"  {summary['Status']}  extracted={summary['ExtractedStocks']}  missing={summary['MissingCount']}")

    # ── 2. Sample ────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(">>> [2/5] Sample")
    print("="*50)
    run_sample(
        raw_root=TEST_DATA_DIR,
        sampled_root=TEST_SAMPLED,
        dates=dates,
        freq=config.SAMPLE_FREQ,
        am_start=config.AM_START,
        am_end=config.AM_END,
        pm_start=config.PM_START,
        pm_end=config.PM_END,
        max_workers=args.workers,
    )

    # ── 3. Clean ─────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(">>> [3/5] Clean")
    print("="*50)
    run_clean(
        sampled_root=TEST_SAMPLED,
        cleaned_root=TEST_CLEANED,
        override_csv=config.DROP_OVERRIDES_CSV,
        gap_threshold=config.GAP_REVIEW_THRESHOLD,
        dates=dates,
        max_workers=args.workers,
    )

    # ── 4. Base ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(">>> [4/5] Base")
    print("="*50)
    run_base(
        cleaned_root=TEST_CLEANED,
        base_root=TEST_BASE,
        dates=dates,
        max_workers=args.workers,
    )

    # ── 5. Factors ───────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f">>> [5/5] Factors（{len(ALL_FACTORS)} 个）")
    print("="*50)
    for i, factor_name in enumerate(ALL_FACTORS, 1):
        print(f"\n  [{i}/{len(ALL_FACTORS)}] {factor_name}")
        run_factors(
            base_root=TEST_BASE,
            factor_root=TEST_FACTOR,
            factor_name=factor_name,
            dates=dates,
            max_workers=args.workers,
        )

    print("\n" + "="*50)
    print("测试集数据提取完成。")
    print(f"  extract → {TEST_DATA_DIR}")
    print(f"  factor  → {TEST_FACTOR}")
    print("下一步：运行 xgb_predict --test 进行推理")
    print("="*50)


if __name__ == "__main__":
    main()
