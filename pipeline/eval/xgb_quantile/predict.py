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

性能说明
--------
- 每个 ret_horizon 使用独立的 ProcessPoolExecutor，通过 initializer 在
  worker 进程启动时加载一次模型，避免每天重复从磁盘加载（243 天 → 1 次/worker）。
- 注意：推理时使用 model.predict(output_margin=True) 获取 logits，
  然后自行应用 softmax，保证与训练时 cost_obj 的数值一致。
"""

import json
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


# ── Worker 进程全局模型（每个 worker 只加载一次）─────────────────────────────────

_worker_model: "xgb.Booster | None" = None


def _init_predict_worker(model_path: str) -> None:
    """ProcessPoolExecutor initializer：worker 启动时加载模型到进程内存。"""
    global _worker_model
    _worker_model = xgb.Booster()
    _worker_model.load_model(model_path)


# ── 单日推理 ───────────────────────────────────────────────────────────────────

def _predict_day(
    factor_root: str,
    base_root: str,
    fc_to_fn: dict[str, str],
    feature_cols: list[str],
    day: str,
    n_groups: int,
    ret_col: str,
    out_dir: str,
) -> str:
    out_path = os.path.join(out_dir, f"{day}.parquet")
    if os.path.exists(out_path):
        return day

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
        except Exception as e:
            print(f"[WARN][predict][{day}] 读取因子文件失败: {path} | {type(e).__name__}: {e}")

    if not factor_dfs:
        print(f"[WARN][predict][{day}] 未读到任何因子文件")
        return day

    merged = factor_dfs[0]
    for df in factor_dfs[1:]:
        merged = merged.merge(df, on=["SampleTime", "SecurityID"], how="outer")

    base_path = os.path.join(base_root, f"{day}.parquet")
    if not os.path.exists(base_path):
        print(f"[WARN][predict][{day}] base 文件不存在: {base_path}")
        return day
    try:
        base_df = pd.read_parquet(base_path, columns=["SampleTime", "SecurityID", ret_col])
    except Exception as e:
        print(f"[WARN][predict][{day}] 读取 base 文件失败: {base_path} | {type(e).__name__}: {e}")
        return day

    merged = merged.merge(base_df, on=["SampleTime", "SecurityID"], how="inner")
    if merged.empty:
        print(f"[WARN][predict][{day}] 合并收益后为空")
        return day

    missing_feats = [c for c in feature_cols if c not in merged.columns]
    if missing_feats:
        print(
            f"[ERROR][predict][{day}] 特征列缺失 {len(missing_feats)} 个: "
            f"{missing_feats[:5]}{'...' if len(missing_feats) > 5 else ''}，跳过本日"
        )
        return day

    X = merged.reindex(columns=feature_cols).astype(np.float32).values
    dmat = xgb.DMatrix(X, feature_names=feature_cols, missing=np.nan)

    # 使用 worker 进程内已加载的模型（主进程单天调用时需先调用 _init_predict_worker）
    model = _worker_model
    if model is None:
        raise RuntimeError(
            "_worker_model 未初始化；请通过 ProcessPoolExecutor(initializer=_init_predict_worker) "
            "或在单线程模式下先调用 _init_predict_worker(model_path)"
        )

    raw_pred = model.predict(dmat, output_margin=True)
    raw_pred = raw_pred.reshape(len(merged), n_groups)
    raw_pred -= raw_pred.max(axis=1, keepdims=True)
    exp_p = np.exp(raw_pred)
    probs = exp_p / exp_p.sum(axis=1, keepdims=True)

    class_vals = np.arange(n_groups, dtype=np.float64)
    scores = (probs * class_vals).sum(axis=1)

    ret_vals = merged[ret_col].values.astype(np.float64)
    times = merged["SampleTime"].values

    seen: set = set()
    unique_times = []
    for t in times:
        if t not in seen:
            unique_times.append(t)
            seen.add(t)

    rows = []
    for t in unique_times:
        mask = times == t
        sc = scores[mask]
        ret = ret_vals[mask]
        n_stocks = mask.sum()

        if n_stocks < n_groups:
            continue

        rank = np.argsort(np.argsort(sc))
        group = (rank * n_groups // n_stocks).clip(0, n_groups - 1)

        ret_finite = np.isfinite(ret)
        row = {"Date": day, "SampleTime": t}
        for g in range(n_groups):
            in_g = (group == g) & ret_finite
            row[f"g{g + 1}"] = float(ret[in_g].mean()) if in_g.any() else np.nan
        rows.append(row)

    if not rows:
        print(f"[WARN][predict][{day}] 所有截面都不足以形成有效分组")
        return day

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return day


def _predict_worker(args) -> str:
    return _predict_day(*args)


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
    val_only: bool = False,
) -> None:
    """
    val_only=True 时只推理验证集日期（从 xgb_quantile/date_split.json 读取），
    结果写入 eval_root/xgb_quantile_val/ 而非 xgb_quantile/，两者互不干扰。
    模型始终从 xgb_quantile/ 加载。
    """
    from pipeline.eval.quantile.multi.charts import run_post_compute

    factor_pools = factor_pools or ["all", "union", "intersection"]
    n_groups_list = n_groups_list or [10, 20]
    ret_horizons = ret_horizons or list(_RET_HORIZONS.keys())

    # 模型固定在 xgb_quantile/，输出目录根据 val_only 切换
    model_root = os.path.join(eval_root, "xgb_quantile")
    out_root = os.path.join(eval_root, "xgb_quantile_val" if val_only else "xgb_quantile")

    # val_only 时读取训练时保存的验证集日期列表
    val_date_set: set[str] | None = None
    if val_only:
        split_path = os.path.join(model_root, "date_split.json")
        if not os.path.exists(split_path):
            print(f"[xgb_predict] val_only=True 但未找到 date_split.json: {split_path}")
            return
        with open(split_path) as f:
            split_info = json.load(f)
        val_date_set = set(split_info.get("val", []))
        print(f"[xgb_predict] val_only=True，共 {len(val_date_set)} 个验证日期，输出至 {out_root}")

    for factor_pool in factor_pools:
        fc_to_fn = get_factor_cols_for_pool(
            factor_root, factor_pool,
            union_path=union_path,
            intersection_path=intersection_path,
        )
        if not fc_to_fn:
            print(f"[xgb_predict] 没有因子列 (pool={factor_pool})，跳过")
            continue

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

        # val_only 时过滤到验证集日期
        if val_date_set is not None:
            _dates = [d for d in _dates if d in val_date_set]

        for n_groups in n_groups_list:
            out_dirs: dict[str, str] = {}

            for ret_h in ret_horizons:
                ret_col = _RET_HORIZONS[ret_h]
                model_dir = os.path.join(model_root, factor_pool, f"g{n_groups}", ret_h)
                out_dir = os.path.join(out_root, factor_pool, f"g{n_groups}", ret_h)
                model_path = os.path.join(model_dir, "model.ubj")

                if not os.path.exists(model_path):
                    print(f"[xgb_predict] 模型不存在，跳过：{model_path}")
                    continue

                feat_file = os.path.join(model_dir, "feature_names.txt")
                if not os.path.exists(feat_file):
                    print(f"[xgb_predict] feature_names.txt 不存在：{model_dir}")
                    continue
                with open(feat_file) as f:
                    feature_cols = [line.strip() for line in f if line.strip()]

                out_dirs[ret_h] = out_dir

                # 过滤掉已完成的天
                day_tasks = [
                    (factor_root, base_root, fc_to_fn, feature_cols,
                     day, n_groups, ret_col, out_dir)
                    for day in _dates
                    if not os.path.exists(os.path.join(out_dir, f"{day}.parquet"))
                ]
                already_done = len(_dates) - len(day_tasks)
                print(
                    f"[xgb_predict] pool={factor_pool} g={n_groups} {ret_h}: "
                    f"{len(day_tasks)} 天待推理"
                    + (f"（已跳过 {already_done} 天）" if already_done else "")
                )

                if not day_tasks:
                    continue

                # ── 每个 ret_h 独立的进程池，worker 启动时加载一次模型 ──────────
                if max_workers == 1:
                    # 单进程模式：在主进程中初始化一次模型
                    _init_predict_worker(model_path)
                    inner = (
                        tqdm(day_tasks, desc=f"predict[{ret_h}]", dynamic_ncols=True, leave=False)
                        if tqdm else day_tasks
                    )
                    for t in inner:
                        _predict_worker(t)
                else:
                    with ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=_init_predict_worker,
                        initargs=(model_path,),
                    ) as executor:
                        futs = [executor.submit(_predict_worker, t) for t in day_tasks]
                        inner = (
                            tqdm(
                                as_completed(futs),
                                total=len(futs),
                                desc=f"predict[{factor_pool}/g{n_groups}/{ret_h}]",
                                dynamic_ncols=True,
                                leave=False,
                            )
                            if tqdm else as_completed(futs)
                        )
                        for fut in inner:
                            fut.result()

            if out_dirs:
                print(f"[xgb_predict] pool={factor_pool} g={n_groups}: stage=post_compute")
                run_post_compute(out_dirs)
                print(f"[xgb_predict] 完成：pool={factor_pool} g={n_groups}")
