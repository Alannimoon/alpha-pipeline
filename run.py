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
python run.py xgb_train                             # XGBoost 截面分层训练
python run.py xgb_predict --factor-pool union       # XGBoost 截面分层推理 + 汇总
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
from pipeline.eval.xgb_quantile   import run_xgb_train, run_xgb_predict


def main():
    parser = argparse.ArgumentParser(prog="run.py", description="因子评测流水线")
    sub = parser.add_subparsers(dest="stage", required=True)

    # ── 公共参数 ───────────────────────────────────────────────────────────
    def add_common(p):
        p.add_argument("--date",    default=None, help="只处理指定日期，如 20250102")
        p.add_argument("--workers", type=int, default=None, help="并行进程数（默认 CPU 核数）")
        p.add_argument("--pool",    default="a500", choices=["a500", "vol100", "test"],
                       help="数据池：a500（默认，A500全年）/ vol100（vol100全年）/ test（vol100测试集43天）")

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

    # ── xgb_train ──────────────────────────────────────────────────────────────
    p_xgbt = sub.add_parser("xgb_train", help="XGBoost 截面分层训练（代价敏感目标函数）")
    p_xgbt.add_argument("--factor-pool", nargs="+", default=None,
                        choices=["all", "union", "intersection"],
                        help="因子池（可多选，默认全部三种）")
    p_xgbt.add_argument("--n-groups", nargs="+", type=int, default=None,
                        help="分组数（可多选，如 --n-groups 10 20，默认 10 20）")
    p_xgbt.add_argument("--ret-horizon", nargs="+", default=None,
                        choices=["ret100", "ret200", "ret300"],
                        help="收益率窗口（可多选，默认全部三种）")
    p_xgbt.add_argument("--stride", type=int, default=None,
                        help="训练集截面降采样间隔（默认 100，约每 5 分钟一截面）")
    p_xgbt.add_argument("--num-rounds", type=int, default=500,
                        help="最大 boosting 轮次（默认 500）")
    p_xgbt.add_argument("--early-stop", type=int, default=30,
                        help="早停轮次（默认 30）")
    p_xgbt.add_argument("--within-scale", type=float, default=0.5,
                        help="同侧错分惩罚系数（默认 0.5）")
    p_xgbt.add_argument("--cross-base", type=float, default=5.0,
                        help="跨边界基础惩罚（默认 5.0）")
    p_xgbt.add_argument("--neighbor-zero", type=int, default=2,
                        help="最差前 k 组互错零惩罚（默认 2，即 1、2 类不惩罚）")
    p_xgbt.add_argument("--force", action="store_true",
                        help="强制重新训练（跳过已存在检查）")
    p_xgbt.add_argument("--verbose-eval", type=int, default=20,
                        help="每多少轮打印一次训练日志（默认 20）")
    p_xgbt.add_argument("--slot", default=None,
        choices=["open", "morning", "afternoon", "close"],
        help="分时段训练：只用该时段的样本（open=09:30-10:00 / morning=10:00-11:30 / afternoon=13:00-14:00 / close=14:00-14:57）")
    p_xgbt.add_argument("--data-workers", type=int, default=8,
                        help="数据加载并行进程数（默认 8，仅影响训练集/验证集构建速度）")
    p_xgbt.add_argument("--extra-features", default=None, metavar="NAME",
                        help="额外特征集名称，对应 config/extra_features/{NAME}.txt（如 market_state）；"
                             "追加到所有因子池，输出目录加后缀 _{NAME}")
    p_xgbt.add_argument("--pool", default="a500", choices=["a500", "vol100"],
                        help="训练数据池：a500（默认）/ vol100（vol100全年，输出目录加后缀 _vol100）")

    # ── xgb_predict ────────────────────────────────────────────────────────────
    p_xgbp = sub.add_parser("xgb_predict", help="XGBoost 截面分层推理 + 生成汇总图表")
    p_xgbp.add_argument("--factor-pool", nargs="+", default=None,
                        choices=["all", "union", "intersection"],
                        help="因子池（可多选，默认全部三种）")
    p_xgbp.add_argument("--n-groups", nargs="+", type=int, default=None,
                        help="分组数（可多选，如 --n-groups 10 20，默认 10 20）")
    p_xgbp.add_argument("--ret-horizon", nargs="+", default=None,
                        choices=["ret100", "ret200", "ret300"],
                        help="收益率窗口（可多选，默认全部三种）")
    p_xgbp.add_argument("--date", default=None, help="只推理指定日期，如 20250102")
    p_xgbp.add_argument("--slot", default=None,
        choices=["open", "morning", "afternoon", "close"],
        help="分时段推理：加载时段模型，只推理该时段截面（需先用 xgb_train --slot 训练）")
    p_xgbp.add_argument("--pool", default="a500", choices=["a500", "vol100", "test"],
                        help="推理数据池：a500（默认）/ vol100（vol100全年，用于验证集推理）/ test（43天测试集）")
    p_xgbp.add_argument("--val-only", action="store_true", default=False,
                        help="只推理验证集日期（读取 date_split.json），结果写入 *_val/")
    p_xgbp.add_argument("--workers", type=int, default=None, help="并行进程数")
    p_xgbp.add_argument("--extra-features", default=None, metavar="NAME",
                        help="额外特征集名称，对应 config/extra_features/{NAME}.txt（如 market_state）；"
                             "决定追加的特征列，需与训练时保持一致")
    p_xgbp.add_argument("--model-tag", default=None, metavar="TAG",
                        help="模型目录路由标签（默认与 --extra-features 相同）；"
                             "vol100 推理时设为 {extra-features}_vol100 以加载对应模型")

    args  = parser.parse_args()
    dates = [args.date] if getattr(args, "date", None) else None
    _pool = getattr(args, "pool", "a500")

    # ── 路径路由（按 pool 选择三套路径之一）──────────────────────────────────────
    _path = {
        "a500":  (config.RAW_ROOT,        config.SAMPLED_ROOT,        config.CLEANED_ROOT,        config.BASE_ROOT,        config.FACTOR_ROOT),
        "vol100":(config.VOL100_RAW_ROOT,  config.VOL100_SAMPLED_ROOT, config.VOL100_CLEANED_ROOT, config.VOL100_BASE_ROOT, config.VOL100_FACTOR_ROOT),
        "test":  (config.TEST_RAW_ROOT,    config.TEST_SAMPLED_ROOT,   config.TEST_CLEANED_ROOT,   config.TEST_BASE_ROOT,   config.TEST_FACTOR_ROOT),
    }[_pool]
    _raw_root, _sampled_root, _cleaned_root, _base_root, _factor_root = _path

    if args.stage == "sample":
        run_sample(
            raw_root=_raw_root, sampled_root=_sampled_root,
            dates=dates, freq=config.SAMPLE_FREQ,
            am_start=config.AM_START, am_end=config.AM_END,
            pm_start=config.PM_START, pm_end=config.PM_END,
            max_workers=args.workers,
        )
    elif args.stage == "clean":
        run_clean(
            sampled_root=_sampled_root, cleaned_root=_cleaned_root,
            override_csv=config.DROP_OVERRIDES_CSV,
            gap_threshold=config.GAP_REVIEW_THRESHOLD,
            dates=dates, max_workers=args.workers,
        )
    elif args.stage == "base":
        run_base(
            cleaned_root=_cleaned_root, base_root=_base_root,
            dates=dates, max_workers=args.workers,
        )
    elif args.stage == "factors":
        run_factors(
            base_root=_base_root, factor_root=_factor_root,
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
        from pipeline.eval.xgb_quantile.dataset import load_pools_file
        _pool = args.factor_pool
        _whitelist: set[str] | None = None
        if _pool in ("union", "intersection"):
            _pools = load_pools_file(config.FACTOR_POOLS_TXT)
            _whitelist = _pools.get(_pool)
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
            whitelist=_whitelist,
            n_groups=args.n_groups,
        )
    elif args.stage == "xgb_train":
        _ef_tag = getattr(args, "extra_features", None)
        _ef_set: set[str] | None = None
        if _ef_tag:
            import os as _os
            _ef_path = _os.path.join(config.EXTRA_FEATURES_DIR, f"{_ef_tag}.txt")
            if not _os.path.exists(_ef_path):
                raise FileNotFoundError(f"额外特征文件不存在：{_ef_path}")
            with open(_ef_path, encoding="utf-8") as _f:
                _ef_set = {l.strip() for l in _f if l.strip() and not l.startswith("#")}
            print(f"[xgb_train] 已加载额外特征集 {_ef_tag}（{len(_ef_set)} 个特征）")
        _train_pool  = getattr(args, "pool", "a500")
        _is_vol100   = (_train_pool == "vol100")
        _factor_root = config.VOL100_FACTOR_ROOT if _is_vol100 else config.FACTOR_ROOT
        _base_root   = config.VOL100_BASE_ROOT   if _is_vol100 else config.BASE_ROOT
        _ef_tag_out  = (f"{_ef_tag}_vol100" if _ef_tag else "vol100") if _is_vol100 else _ef_tag
        if _is_vol100:
            print(f"[xgb_train] vol100 全年训练模式：factor_root={_factor_root}")

        run_xgb_train(
            factor_root=_factor_root,
            base_root=_base_root,
            eval_root=config.EVAL_ROOT,
            factor_pools=args.factor_pool,
            n_groups_list=args.n_groups,
            ret_horizons=args.ret_horizon,
            stride=args.stride,
            pools_path=config.FACTOR_POOLS_TXT,
            extra_features=_ef_set,
            extra_features_tag=_ef_tag_out,
            num_boost_round=args.num_rounds,
            early_stopping_rounds=args.early_stop,
            penalty_kwargs={
                "within_scale":  args.within_scale,
                "cross_base":    args.cross_base,
                "neighbor_zero": args.neighbor_zero,
            },
            verbose_eval=args.verbose_eval,
            force=args.force,
            data_workers=args.data_workers,
            slot=args.slot,
        )
    elif args.stage == "xgb_predict":
        _dates = [args.date] if args.date else None
        _ef_tag = getattr(args, "extra_features", None)
        _ef_set = None
        if _ef_tag:
            import os as _os
            _ef_path = _os.path.join(config.EXTRA_FEATURES_DIR, f"{_ef_tag}.txt")
            if not _os.path.exists(_ef_path):
                raise FileNotFoundError(f"额外特征文件不存在：{_ef_path}")
            with open(_ef_path, encoding="utf-8") as _f:
                _ef_set = {l.strip() for l in _f if l.strip() and not l.startswith("#")}
            print(f"[xgb_predict] 已加载额外特征集 {_ef_tag}（{len(_ef_set)} 个特征）")
        _pred_pool = getattr(args, "pool", "a500")
        _pred_factor_root, _pred_base_root, _pred_is_test = {
            "a500":   (config.FACTOR_ROOT,        config.BASE_ROOT,        False),
            "vol100": (config.VOL100_FACTOR_ROOT,  config.VOL100_BASE_ROOT, False),
            "test":   (config.TEST_FACTOR_ROOT,    config.TEST_BASE_ROOT,   True),
        }[_pred_pool]
        run_xgb_predict(
            factor_root=_pred_factor_root,
            base_root=_pred_base_root,
            eval_root=config.EVAL_ROOT,
            factor_pools=args.factor_pool,
            n_groups_list=args.n_groups,
            ret_horizons=args.ret_horizon,
            dates=_dates,
            max_workers=args.workers,
            pools_path=config.FACTOR_POOLS_TXT,
            extra_features=_ef_set,
            extra_features_tag=_ef_tag,
            model_tag=getattr(args, "model_tag", None),
            val_only=args.val_only,
            slot=args.slot,
            test=_pred_is_test,
            test_factor_root=config.TEST_FACTOR_ROOT,
            test_base_root=config.TEST_BASE_ROOT,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
