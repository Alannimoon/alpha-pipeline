"""
XGBoost 截面分层 — 推理模块。

推理流程（每日每 ret_horizon）
------------------------------
1. 加载全量因子数据（无 stride，覆盖全天 4740 个截面）
2. 对每个样本（SampleTime, SecurityID）构造特征向量
3. XGBoost 输出各类别 logits → softmax → 期望类别值作为连续排序得分
4. 逐截面按得分排名，等量分为 n_groups 组
5. 计算各组平均前向收益 → 输出 parquet（格式与 multi_factor_quantile 完全一致）
6. 调用 run_post_compute 生成汇总 CSV 和图表

输出目录
--------
{eval_root}/xgb_quantile/{factor_pool}/g{n_groups}/{ret_h}/
  {date}.parquet    columns: Date, SampleTime, g1 .. g{n_groups}
  _daily.csv        每日各组均值
  _summary.csv      跨日整体均值
  _cum_daily.csv    跨日累计末值（供 pnl_summary.py 使用）
  _chart_tick.png   跨日 tick 连续曲线
  _chart_slot_*.png 8 个 30 分钟时段图

注意：推理时使用 model.predict(output_margin=True) 获取 logits，
然后自行应用 softmax，保证与训练时 cost_obj 的数值一致。
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .dataset import _RET_HORIZONS, get_factor_cols_for_pool

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("请先安装 xgboost：pip install xgboost") from e

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ── 单日推理 ───────────────────────────────────────────────────────────────────

def _predict_day(
    factor_root: str,
    base_root: str,
    fc_to_fn: dict[str, str],
    feature_cols: list[str],
    model_path: str,
    day: str,
    n_groups: int,
    ret_col: str,
    out_dir: str,
) -> str:
    """
    对单日全量截面做推理并写出 {day}.parquet。

    推理得分
    --------
    score_i = Σ_c c × p_c(f_i)   （期望类别值，连续排序信号）

    分组逻辑
    --------
    逐截面对 score 做双重 argsort 得到排名，
    然后与 multi_factor_quantile 完全相同地等量分组并计算组均值收益。
    """
    out_path = os.path.join(out_dir, f"{day}.parquet")
    if os.path.exists(out_path):
        return day

    # ── 1. 加载因子数据 ──────────────────────────────────────────────────────
    name_to_cols: dict[str, list[str]] = defaultdict(list)
    for fc, fn in fc_to_fn.items():
        if fc in feature_cols:
            name_to_cols[fn].append(fc)

    factor_dfs: list[pd.DataFrame] = []
    for fn, cols in name_to_cols.items():
        path = os.path.join(factor_root, fn, f"{day}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path, columns=["SampleTime", "SecurityID"] + cols)
            factor_dfs.append(df)
        except Exception:
            pass

    if not factor_dfs:
        return day

    merged = factor_dfs[0]
    for df in factor_dfs[1:]:
        merged = merged.merge(df, on=["SampleTime", "SecurityID"], how="outer")

    # ── 2. 加载收益率 ────────────────────────────────────────────────────────
    base_path = os.path.join(base_root, f"{day}.parquet")
    if not os.path.exists(base_path):
        return day
    try:
        base_df = pd.read_parquet(base_path, columns=["SampleTime", "SecurityID", ret_col])
    except Exception:
        return day

    merged = merged.merge(base_df, on=["SampleTime", "SecurityID"], how="inner")
    if merged.empty:
        return day

    # ── 3. 构造特征矩阵（对齐 feature_cols，缺失列填 NaN）──────────────────
    X = merged.reindex(columns=feature_cols).astype(np.float32).values
    dmat = xgb.DMatrix(X, feature_names=feature_cols, missing=np.nan)

    # ── 4. 加载模型并推理 ────────────────────────────────────────────────────
    model = xgb.Booster()
    model.load_model(model_path)

    raw_pred = model.predict(dmat, output_margin=True)     # logits
    raw_pred = raw_pred.reshape(len(merged), n_groups)
    raw_pred -= raw_pred.max(axis=1, keepdims=True)        # 数值稳定
    exp_p  = np.exp(raw_pred)
    probs  = exp_p / exp_p.sum(axis=1, keepdims=True)      # (n_samples, n_groups)

    # 期望类别值 = Σ_c c × p_c，作为连续排序得分
    class_vals = np.arange(n_groups, dtype=np.float64)
    scores = (probs * class_vals).sum(axis=1)               # (n_samples,)

    ret_vals  = merged[ret_col].values.astype(np.float64)
    times     = merged["SampleTime"].values

    # ── 5. 逐截面分组 → 计算组均值收益 ─────────────────────────────────────
    # 保持 SampleTime 原始顺序
    seen: set = set()
    unique_times = []
    for t in times:
        if t not in seen:
            unique_times.append(t)
            seen.add(t)

    rows = []
    for t in unique_times:
        mask     = times == t
        sc       = scores[mask]
        ret      = ret_vals[mask]
        n_stocks = mask.sum()

        if n_stocks < n_groups:
            continue

        # 等量分组：double argsort → rank → floor division
        rank  = np.argsort(np.argsort(sc))
        group = (rank * n_groups // n_stocks).clip(0, n_groups - 1)

        ret_finite = np.isfinite(ret)
        row = {"Date": day, "SampleTime": t}
        for g in range(n_groups):
            in_g = (group == g) & ret_finite
            row[f"g{g + 1}"] = float(ret[in_g].mean()) if in_g.any() else np.nan
        rows.append(row)

    if not rows:
        return day

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return day


# ── Worker（ProcessPoolExecutor 调用）────────────────────────────────────────

def _predict_worker(args) -> str:
    return _predict_day(*args)


# ── 批量推理入口 ───────────────────────────────────────────────────────────────

def run_xgb_predict(
    factor_root: str,
    base_root: str,
    eval_root: str,
    factor_pools: list[str] | None = None,
    n_groups_list: list[int] | None = None,
    ret_horizons: list[str] | None = None,
    dates: list[str] | None = None,
    max_workers: int | None = None,
    union_path: str | None = None,
    intersection_path: str | None = None,
) -> None:
    """
    批量推理入口，自动扫描已训练模型并生成汇总 CSV 和图表。

    支持一次推理多个 (factor_pool, n_groups, ret_horizon) 组合。
    推理结果格式与 multi_factor_quantile 完全相同，
    可直接被 pnl_summary.py 的 collect_xgb() 读取比较。

    Parameters
    ----------
    factor_root   : 因子数据根目录
    base_root     : base 数据根目录
    eval_root     : 评估结果根目录
    factor_pools  : 因子池列表，默认 ["all", "union", "intersection"]
    n_groups_list : 分组数列表，默认 [10, 20]
    ret_horizons  : 收益率窗口列表，默认全部三种
    dates         : 日期列表（None 时自动扫描）
    max_workers   : 并行进程数（按天并行）
    """
    from pipeline.eval.quantile.multi.charts import run_post_compute

    factor_pools  = factor_pools  or ["all", "union", "intersection"]
    n_groups_list = n_groups_list or [10, 20]
    ret_horizons  = ret_horizons  or list(_RET_HORIZONS.keys())
    xgb_root      = os.path.join(eval_root, "xgb_quantile")

    for factor_pool in factor_pools:
        fc_to_fn = get_factor_cols_for_pool(
            factor_root, factor_pool,
            union_path=union_path,
            intersection_path=intersection_path,
        )
        if not fc_to_fn:
            print(f"[xgb_predict] 没有因子列 (pool={factor_pool})，跳过")
            continue

        # 自动扫描日期（每个 pool 共用同一份日期列表）
        _dates = dates
        if _dates is None:
            any_fn = next(iter(fc_to_fn.values()), None)
            if any_fn is None:
                continue
            scan_dir = os.path.join(factor_root, any_fn)
            _dates = sorted(
                os.path.splitext(fname)[0]
                for fname in os.listdir(scan_dir)
                if fname.endswith(".parquet")
                   and not fname.startswith("_")
                   and os.path.splitext(fname)[0].isdigit()
                   and len(os.path.splitext(fname)[0]) == 8
            )

        for n_groups in n_groups_list:
            pool_dir = os.path.join(xgb_root, factor_pool, f"g{n_groups}")
            out_dirs: dict[str, str] = {}
            tasks: list[tuple] = []

            for ret_h in ret_horizons:
                ret_col    = _RET_HORIZONS[ret_h]
                ret_dir    = os.path.join(pool_dir, ret_h)
                model_path = os.path.join(ret_dir, "model.ubj")

                if not os.path.exists(model_path):
                    print(f"[xgb_predict] 模型不存在，跳过：{model_path}")
                    continue

                feat_file = os.path.join(ret_dir, "feature_names.txt")
                if not os.path.exists(feat_file):
                    print(f"[xgb_predict] feature_names.txt 不存在：{ret_dir}")
                    continue
                with open(feat_file) as f:
                    feature_cols = [line.strip() for line in f if line.strip()]

                out_dirs[ret_h] = ret_dir
                for day in _dates:
                    tasks.append((
                        factor_root, base_root, fc_to_fn, feature_cols,
                        model_path, day, n_groups, ret_col, ret_dir,
                    ))

            if not tasks:
                continue

            print(f"[xgb_predict] pool={factor_pool}  g={n_groups}  "
                  f"共 {len(tasks)} 个推理任务（{len(_dates)} 天 × {len(out_dirs)} ret）")

            if max_workers == 1:
                it = tqdm(tasks, desc="xgb_predict") if tqdm else tasks
                for t in it:
                    _predict_worker(t)
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futs  = [executor.submit(_predict_worker, t) for t in tasks]
                    inner = (
                        tqdm(as_completed(futs), total=len(futs), desc="xgb_predict")
                        if tqdm else as_completed(futs)
                    )
                    for fut in inner:
                        fut.result()

            run_post_compute(out_dirs)
            print(f"[xgb_predict] 完成：{pool_dir}")
