"""
统一入口。

用法
----
python run.py sample   --date 20250102              # 重采样
python run.py clean    --date 20250102              # 清洗
python run.py base     --date 20250102              # 生成 base 数据（价格、掩码、收益率）
python run.py factors  --date 20250102              # 计算因子
python run.py cs_ic     --date 20250102 --factor bap # 截面 IC
python run.py ic_report --factor bap                 # IC 统计 + 画图
python run.py cs_quantile --date 20250102 --factor bap  # 截面分层
python run.py cs_quantile_chart --factor bap        # 重新生成截面分层跨日 tick 图
python run.py multi_factor_quantile                 # 多因子合成分层（十分位）
"""

import argparse
import sys

import config
from pipeline.ingest.sample  import run_sample
from pipeline.ingest.clean   import run_clean
from pipeline.ingest.base    import run_base
from pipeline.factor.compute import run_factors
from pipeline.eval.ic.cs_ic      import run_cs_ic
from pipeline.eval.ic.ic_report  import run_ic_report
from pipeline.eval.quantile.single import run_cs_quantile, run_cs_quantile_chart
from pipeline.eval.quantile.multi  import run_multi_factor_quantile


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
    add_common(sub.add_parser("base",    help="Base：价格定义、涨跌停标注、盘口掩码、ret_fwd"))

    # ── factors ────────────────────────────────────────────────────────────
    p_factors = sub.add_parser("factors", help="因子：计算指定因子")
    add_common(p_factors)
    p_factors.add_argument("--factor", default="bap", help="因子名称，如 bap / mom")

    # ── cs_ic ──────────────────────────────────────────────────────────────
    add_eval(sub.add_parser("cs_ic",   help="截面 IC：按 (Date, SampleTime) 分组"))

    # ── ic_report ───────────────────────────────────────────────────────────────────────
    add_factor_only(sub.add_parser("ic_report", help="IC 统计 + 画图：均值/ICIR 汇总 & 3 张图"))

    # ── cs_quantile ────────────────────────────────────────────────────────────
    add_eval(sub.add_parser("cs_quantile", help="截面分层：五分位组收益均值"))

    # ── cs_quantile_chart ──────────────────────────────────────────────────────
    add_factor_only(sub.add_parser("cs_quantile_chart", help="重新生成截面分层跨日 tick 图（不重跑分层计算）"))

    # ── multi_factor_quantile ──────────────────────────────────────────────────
    p_mfq = sub.add_parser("multi_factor_quantile", help="多因子合成分层：IC 加权十分位")
    p_mfq.add_argument("--date",         default=None,   help="只处理指定日期，如 20250102")
    p_mfq.add_argument("--workers",      type=int, default=None, help="并行进程数（默认 CPU 核数）")
    p_mfq.add_argument("--threshold",    type=float, default=0.02, help="IC 筛选阈值（默认 0.02）")
    p_mfq.add_argument("--score-method", default="rank", choices=["rank", "zscore", "minmax"],
                       help="因子截面标准化方式：rank（分位数得分，默认）/ zscore（Z-score ±3截断）/ minmax（MinMax 归一化）")
    p_mfq.add_argument("--factor-pool", default="threshold",
                       choices=["threshold", "union", "intersection"],
                       help="因子池：threshold（IC阈值筛选，默认）/ union（并集51个）/ intersection（交集25个）")
    p_mfq.add_argument("--n-groups", type=int, default=10,
                       help="分层组数，默认 10")

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
            factor_name=args.factor,
            dates=dates, max_workers=args.workers,
        )
    elif args.stage == "cs_ic":
        run_cs_ic(
            factor_root=config.FACTOR_ROOT,
            base_root=config.BASE_ROOT,
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor, dates=dates,
            max_workers=getattr(args, "workers", None),
        )
    elif args.stage == "ic_report":
        run_ic_report(
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
        )
    elif args.stage == "cs_quantile":
        run_cs_quantile(
            factor_root=config.FACTOR_ROOT,
            base_root=config.BASE_ROOT,
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
            dates=dates,
            max_workers=getattr(args, "workers", None),
        )
    elif args.stage == "cs_quantile_chart":
        run_cs_quantile_chart(
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
        )
    elif args.stage == "multi_factor_quantile":
        _pool = args.factor_pool
        _whitelist_path = None
        if _pool == "union":
            _whitelist_path = config.FACTOR_POOL_UNION_TXT
        elif _pool == "intersection":
            _whitelist_path = config.FACTOR_POOL_INTERSECTION_TXT
        run_multi_factor_quantile(
            factor_root=config.FACTOR_ROOT,
            base_root=config.BASE_ROOT,
            eval_root=config.EVAL_ROOT,
            ic_stats_root=config.IC_STATS_ROOT,
            threshold=args.threshold,
            dates=dates,
            max_workers=args.workers,
            score_method=args.score_method,
            factor_pool=_pool,
            whitelist_path=_whitelist_path,
            n_groups=args.n_groups,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
