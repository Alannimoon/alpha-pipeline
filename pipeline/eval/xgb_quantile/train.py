"""
XGBoost 截面分层 — 训练编排模块。

对每种 (factor_pool, n_groups, ret_horizon) 组合：
  1. 自动扫描全部日期并划分训练集 / 验证集
  2. 构建训练集（stride 降采样）和验证集（stride 降采样）
  3. 训练 XGBoost（代价敏感目标函数 + 早停）
  4. 保存模型及元数据

输出目录
--------
{eval_root}/xgb_quantile/{factor_pool}/g{n_groups}/{ret_h}/
  model.ubj            XGBoost Booster（二进制）
  feature_names.txt    每行一个特征列名
  penalty_matrix.npy   惩罚矩阵（npy）
  split_info.json      训练/验证日期及训练元数据
"""

import json
import os

import numpy as np
import pandas as pd

from .dataset import (
    _RET_HORIZONS,
    build_dataset,
    get_factor_cols_for_pool,
    split_dates,
)
from .loss import build_penalty_matrix, make_cost_eval, make_cost_obj

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("请先安装 xgboost：pip install xgboost") from e


# ── 默认超参数 ─────────────────────────────────────────────────────────────────

_DEFAULT_XGB_PARAMS: dict = {
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "tree_method": "hist",
    "nthread": -1,          # 使用全部 CPU 核心
    "seed": 42,
    "disable_default_eval_metric": 1,
}

_DEFAULT_NUM_BOOST_ROUND  = 500
_DEFAULT_EARLY_STOPPING   = 30


# ── 单次训练 ───────────────────────────────────────────────────────────────────

def _train_one(
    factor_root: str,
    base_root: str,
    fc_to_fn: dict[str, str],
    train_dates: list[str],
    val_dates: list[str],
    out_dir: str,
    ret_col: str,
    n_groups: int,
    stride: int,
    xgb_params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    penalty_kwargs: dict,
    verbose_eval: int = 20,
) -> None:
    """
    训练单个模型（一种 factor_pool × n_groups × ret_horizon 组合）。

    训练过程
    --------
    1. 构建 (n_groups, n_groups) 惩罚矩阵
    2. 加载训练/验证集（stride 降采样）
    3. 构造 XGBoost DMatrix（float32，NaN 作为 missing 值）
    4. 以代价敏感目标函数训练，验证集期望代价为早停指标
    5. 保存模型 + 特征名 + 元数据
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 惩罚矩阵 ─────────────────────────────────────────────────────────
    P = build_penalty_matrix(n_groups=n_groups, **penalty_kwargs)
    np.save(os.path.join(out_dir, "penalty_matrix.npy"), P)

    cost_obj  = make_cost_obj(P)
    cost_eval = make_cost_eval(P)

    # ── 2. 加载数据 ─────────────────────────────────────────────────────────
    print(f"\n  [训练集] 加载 {len(train_dates)} 天，stride={stride} ...")
    train_df = build_dataset(
        factor_root, base_root, fc_to_fn, train_dates,
        ret_col, n_groups=n_groups, stride=stride, verbose=False,
    )
    print(f"  训练集：{len(train_df):,} 样本")

    print(f"  [验证集] 加载 {len(val_dates)} 天，stride={stride} ...")
    val_df = build_dataset(
        factor_root, base_root, fc_to_fn, val_dates,
        ret_col, n_groups=n_groups, stride=stride, verbose=False,
    )
    print(f"  验证集：{len(val_df):,} 样本")

    if train_df.empty or val_df.empty:
        print(f"  [跳过] 数据不足")
        return

    # ── 3. 确定特征列（排除元数据和收益率列）───────────────────────────────
    meta_cols = {"Date", "SampleTime", "SecurityID", "label", ret_col}
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    print(f"  特征数：{len(feature_cols)}")

    with open(os.path.join(out_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols) + "\n")

    X_train = train_df[feature_cols].astype(np.float32).values
    y_train = train_df["label"].values.astype(np.int32)
    X_val   = val_df[feature_cols].astype(np.float32).values
    y_val   = val_df["label"].values.astype(np.int32)

    # ── 4. 构造 DMatrix ─────────────────────────────────────────────────────
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, missing=np.nan)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_cols, missing=np.nan)

    # 释放原始 DataFrame 内存
    del X_train, X_val, train_df, val_df

    # ── 5. 训练 ─────────────────────────────────────────────────────────────
    params = {**xgb_params, "num_class": n_groups}

    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name="cost",
            save_best=True,
            maximize=False,
        )
    ]

    print(f"  [XGBoost] num_class={n_groups}, features={len(feature_cols)}, "
          f"max_rounds={num_boost_round}, early_stop={early_stopping_rounds}")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        obj=cost_obj,
        custom_metric=cost_eval,
        callbacks=callbacks,
        verbose_eval=verbose_eval,
    )

    # ── 6. 保存模型及元数据 ──────────────────────────────────────────────────
    model_path = os.path.join(out_dir, "model.ubj")
    model.save_model(model_path)
    print(f"  模型已保存：{model_path}  (best_iteration={model.best_iteration})")

    split_info = {
        "ret_col":        ret_col,
        "n_groups":       n_groups,
        "stride":         stride,
        "n_train_dates":  len(train_dates),
        "n_val_dates":    len(val_dates),
        "n_features":     len(feature_cols),
        "best_iteration": model.best_iteration,
        "penalty_kwargs": penalty_kwargs,
    }
    with open(os.path.join(out_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)


# ── 批量训练入口 ───────────────────────────────────────────────────────────────

def run_xgb_train(
    factor_root: str,
    base_root: str,
    eval_root: str,
    factor_pools: list[str] | None = None,
    n_groups_list: list[int] | None = None,
    ret_horizons: list[str] | None = None,
    stride: int = 100,
    union_path: str | None = None,
    intersection_path: str | None = None,
    xgb_params: dict | None = None,
    num_boost_round: int = _DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING,
    penalty_kwargs: dict | None = None,
    dates: list[str] | None = None,
    verbose_eval: int = 20,
    force: bool = False,
) -> None:
    """
    批量训练入口。顺序遍历所有 (factor_pool, n_groups, ret_horizon) 组合。

    Parameters
    ----------
    factor_root         : 因子数据根目录
    base_root           : base 数据根目录
    eval_root           : 评估结果根目录
    factor_pools        : 因子池列表，默认 ["all", "union", "intersection"]
    n_groups_list       : 分组数列表，默认 [10, 20]
    ret_horizons        : 收益率窗口列表，默认全部三种
    stride              : 训练集截面降采样间隔（默认 100，约每 5 分钟一个截面）
    union_path          : factor_pool_union.txt 路径
    intersection_path   : factor_pool_intersection.txt 路径
    xgb_params          : 自定义 XGBoost 超参数（覆盖默认值）
    num_boost_round     : 最大迭代轮次（默认 500）
    early_stopping_rounds : 早停轮次（默认 30）
    penalty_kwargs      : 传给 build_penalty_matrix 的额外参数
                          如 {"within_scale": 0.5, "cross_base": 5.0, "neighbor_zero": 2}
    dates               : 指定日期列表（None 时自动从 factor_root 扫描）
    verbose_eval        : 每多少轮打印一次训练日志
    force               : True 时跳过已存在检查，强制重新训练
    """
    factor_pools   = factor_pools   or ["all", "union", "intersection"]
    n_groups_list  = n_groups_list  or [10, 20]
    ret_horizons   = ret_horizons   or list(_RET_HORIZONS.keys())
    penalty_kwargs = penalty_kwargs or {}
    params         = {**_DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    # ── 自动扫描日期 ────────────────────────────────────────────────────────
    if dates is None:
        any_fn = next(
            (d for d in os.listdir(factor_root) if os.path.isdir(os.path.join(factor_root, d))),
            None,
        )
        if any_fn is None:
            raise RuntimeError(f"factor_root 为空：{factor_root}")
        scan_dir = os.path.join(factor_root, any_fn)
        dates = sorted(
            os.path.splitext(fname)[0]
            for fname in os.listdir(scan_dir)
            if fname.endswith(".parquet")
               and not fname.startswith("_")
               and os.path.splitext(fname)[0].isdigit()
               and len(os.path.splitext(fname)[0]) == 8
        )

    train_dates, val_dates = split_dates(dates)
    print(f"日期划分：训练集 {len(train_dates)} 天，验证集 {len(val_dates)} 天")

    # ── 保存日期划分 ─────────────────────────────────────────────────────────
    xgb_root = os.path.join(eval_root, "xgb_quantile")
    os.makedirs(xgb_root, exist_ok=True)
    with open(os.path.join(xgb_root, "date_split.json"), "w") as f:
        json.dump({"train": train_dates, "val": val_dates}, f, indent=2)

    # ── 主循环 ───────────────────────────────────────────────────────────────
    for factor_pool in factor_pools:
        fc_to_fn = get_factor_cols_for_pool(
            factor_root, factor_pool,
            union_path=union_path,
            intersection_path=intersection_path,
        )
        print(f"\n=== factor_pool={factor_pool}  特征数={len(fc_to_fn)} ===")
        if not fc_to_fn:
            print("  [跳过] 没有找到因子列")
            continue

        for n_groups in n_groups_list:
            for ret_h in ret_horizons:
                ret_col = _RET_HORIZONS[ret_h]
                out_dir = os.path.join(xgb_root, factor_pool, f"g{n_groups}", ret_h)
                model_path = os.path.join(out_dir, "model.ubj")

                if not force and os.path.exists(model_path):
                    print(f"  [已存在，跳过] {out_dir}")
                    continue

                print(f"\n--- pool={factor_pool}  g={n_groups}  {ret_h} ---")
                _train_one(
                    factor_root, base_root, fc_to_fn,
                    train_dates, val_dates,
                    out_dir, ret_col, n_groups, stride,
                    params, num_boost_round, early_stopping_rounds,
                    penalty_kwargs, verbose_eval,
                )

    print("\n所有模型训练完成。")
