"""
因子评测流水线统一入口。

用法
----
python run.py base                      # 加载分钟线数据，生成 base 文件
python run.py factors --factor mom      # 计算因子（含内联前向收益率）
python run.py cs_ic   --factor mom      # 截面 IC
python run.py ts_ic   --factor mom      # 时序 IC
python run.py ic_stats --factor mom     # IC 汇总统计
python run.py ic_plot  --factor mom     # IC 画图

可用因子：mom, acc_mom, neg_skew, amp_slice, rigidity, pv_corr

数据路径（见 config/__init__.py）：
  输入：DATA_ROOT/*.csv
  输出：result/base/  result/factor/  result/eval/
"""

import argparse
import sys

import config
from pipeline.ingest.load   import run_load
from pipeline.factor.compute import run_factors
from pipeline.eval.cs_ic    import run_cs_ic
from pipeline.eval.ts_ic    import run_ts_ic
from pipeline.eval.ic_stats import run_ic_stats
from pipeline.eval.ic_plot  import run_ic_plot

_FACTORS = ["mom", "acc_mom", "neg_skew", "amp_slice", "rigidity", "pv_corr"]


def main():
    parser = argparse.ArgumentParser(prog="run.py", description="因子评测流水线")
    sub = parser.add_subparsers(dest="stage", required=True)

    def add_common(p):
        p.add_argument("--date",    default=None, help="只处理指定日期，如 20220104")
        p.add_argument("--workers", type=int, default=None, help="并行进程数（默认 CPU 核数）")

    def add_eval(p):
        p.add_argument("--date",    default=None)
        p.add_argument("--factor",  default="mom", help=f"可选：{_FACTORS}")
        p.add_argument("--workers", type=int, default=None)

    def add_factor_only(p):
        p.add_argument("--factor", default="mom", help=f"可选：{_FACTORS}")

    p_base = sub.add_parser("base", help="加载分钟线数据，生成 base 文件")
    p_base.add_argument("--data",    default=None, help="覆盖 DATA_ROOT，如 /data3/aiquanta/data/1min_data/2022")
    p_base.add_argument("--workers", type=int, default=None)

    p_factors = sub.add_parser("factors", help="计算因子（含内联前向收益率）")
    add_common(p_factors)
    p_factors.add_argument("--factor", default="mom", help=f"可选：{_FACTORS}")

    add_eval(sub.add_parser("cs_ic",   help="截面 IC"))
    add_eval(sub.add_parser("ts_ic",   help="时序 IC"))
    add_factor_only(sub.add_parser("ic_stats", help="IC 汇总统计"))
    add_factor_only(sub.add_parser("ic_plot",  help="IC 画图"))

    args  = parser.parse_args()
    dates = [args.date] if getattr(args, "date", None) else None

    if args.stage == "base":
        run_load(
            data_root=getattr(args, "data", None) or config.DATA_ROOT,
            base_root=config.BASE_ROOT,
            max_workers=args.workers,
        )
    elif args.stage == "factors":
        run_factors(
            base_root=config.BASE_ROOT,
            factor_root=config.FACTOR_ROOT,
            factor_name=args.factor,
            horizons=config.RETURN_HORIZONS,
            dates=dates,
            max_workers=args.workers,
        )
    elif args.stage == "cs_ic":
        run_cs_ic(
            factor_root=config.FACTOR_ROOT,
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
            dates=dates,
            max_workers=getattr(args, "workers", None),
            ret_horizons=config.RET_HORIZONS_MAP,
        )
    elif args.stage == "ts_ic":
        run_ts_ic(
            factor_root=config.FACTOR_ROOT,
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
            dates=dates,
            max_workers=getattr(args, "workers", None),
            ret_horizons=config.RET_HORIZONS_MAP,
        )
    elif args.stage == "ic_stats":
        run_ic_stats(
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
            ret_horizons=list(config.RET_HORIZONS_MAP.keys()),
        )
    elif args.stage == "ic_plot":
        run_ic_plot(
            eval_root=config.EVAL_ROOT,
            factor_name=args.factor,
            ret_horizons=list(config.RET_HORIZONS_MAP.keys()),
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
