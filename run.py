"""
统一入口。

用法
----
python run.py sample   --date 20250102              # 重采样
python run.py clean    --date 20250102              # 清洗
python run.py base     --date 20250102              # 生成 base 数据（价格、掩码）
python run.py factors  --date 20250102              # 计算因子（含内联收益率）
python run.py cs_ic    --date 20250102 --factor bap # 截面 IC
python run.py ts_ic    --date 20250102 --factor bap # 时序 IC
python run.py ic_stats --factor bap                 # IC 汇总统计
python run.py ic_plot  --factor bap                 # IC 画图
"""

import argparse
import sys

import config
from pipeline.ingest.sample  import run_sample
from pipeline.ingest.clean   import run_clean
from pipeline.ingest.base    import run_base
from pipeline.factor.compute import run_factors
from pipeline.eval.cs_ic     import run_cs_ic
from pipeline.eval.ts_ic     import run_ts_ic
from pipeline.eval.ic_stats  import run_ic_stats
from pipeline.eval.ic_plot   import run_ic_plot


def main():
    parser = argparse.ArgumentParser(prog="run.py", description="因子评测流水线")
    sub = parser.add_subparsers(dest="stage", required=True)

    # ── 公共参数 ───────────────────────────────────────────────────────────
    def add_common(p):
        p.add_argument("--date",    default=None, help="只处理指定日期，如 20250102")
        p.add_argument("--workers", type=int, default=None, help="并行进程数（默认 CPU 核数）")

    def add_eval(p):
        p.add_argument("--date",    default=None, help="只处理指定日期，如 20250102")
        p.add_argument("--factor",  default="bap", help="因子名称，如 bap")
        p.add_argument("--workers", type=int, default=None, help="并行进程数（默认 CPU 核数）")

    def add_factor_only(p):
        p.add_argument("--factor", default="bap", help="因子名称，如 bap")

    # ── sample ─────────────────────────────────────────────────────────────
    add_common(sub.add_parser("sample",  help="重采样：原始快照 → 固定时间网格"))

    # ── clean ──────────────────────────────────────────────────────────────
    add_common(sub.add_parser("clean",   help="清洗：删停牌日，标记大间隔待复核"))

    # ── base ───────────────────────────────────────────────────────────────
    add_common(sub.add_parser("base",    help="Base：价格定义、涨跌停标注、盘口掩码"))

    # ── factors ────────────────────────────────────────────────────────────
    add_common(sub.add_parser("factors", help="因子：计算所有因子（含内联收益率）"))

    # ── cs_ic ──────────────────────────────────────────────────────────────
    add_eval(sub.add_parser("cs_ic",   help="截面 IC：按 (Date, SampleTime) 分组"))

    # ── ts_ic ──────────────────────────────────────────────────────────────
    add_eval(sub.add_parser("ts_ic",   help="时序 IC：按 (Date, SecurityID) 分组"))

    # ── ic_stats ───────────────────────────────────────────────────────────
    add_factor_only(sub.add_parser("ic_stats", help="IC 汇总统计：均值、标准差、ICIR"))

    # ── ic_plot ────────────────────────────────────────────────────────────
    add_factor_only(sub.add_parser("ic_plot",  help="IC 画图：6 张图（CS/TS × 3 ret horizon）"))

    args = parser.parse_args()
    dates = [args.date] if getattr(args, "date", None) else None

    if args.stage == "sample":
        run_sample(
            raw_root=config.RAW_ROOT, sampled_root=config.SAMPLED_ROOT,
            dates=dates, freq=config.SAMPLE_FREQ,
            am_start=config.AM_START, am_end=config.AM_END,
            pm_start=config.PM_START, pm_end=config.PM_END,
            max_workers=args.workers,
        )
    elif args.stage == "clean":
        run_clean(
            sampled_root=config.SAMPLED_ROOT, cleaned_root=config.CLEANED_ROOT,
            override_csv=config.DROP_OVERRIDES_CSV,
            gap_threshold=config.GAP_REVIEW_THRESHOLD,
            dates=dates, max_workers=args.workers,
        )
    elif args.stage == "base":
        run_base(
            cleaned_root=config.CLEANED_ROOT, base_root=config.BASE_ROOT,
            dates=dates, max_workers=args.workers,
        )
    elif args.stage == "factors":
        run_factors(
            base_root=config.BASE_ROOT,
            factor_root=config.FACTOR_ROOT,
            horizons=config.RETURN_HORIZONS,
            dates=dates, max_workers=args.workers,
        )
    elif args.stage == "cs_ic":
        run_cs_ic(
            factor_root=config.FACTOR_ROOT, eval_root=config.EVAL_ROOT,
            factor_name=args.factor, dates=dates,
            max_workers=getattr(args, "workers", None),
        )
    elif args.stage == "ts_ic":
        run_ts_ic(
            factor_root=config.FACTOR_ROOT, eval_root=config.EVAL_ROOT,
            factor_name=args.factor, dates=dates,
            max_workers=getattr(args, "workers", None),
        )
    elif args.stage == "ic_stats":
        run_ic_stats(
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
        )
    elif args.stage == "ic_plot":
        run_ic_plot(
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
