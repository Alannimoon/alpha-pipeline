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

from .dataset import (
    _RET_HORIZONS,
    SLOT_RANGES,
    build_dataset,
    get_factor_cols_for_pool,
    load_pools_file,
    split_dates,
)
from .loss import build_penalty_matrix, make_cost_eval, make_cost_obj

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("请先安装 xgboost：pip install xgboost") from e

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


_DEFAULT_XGB_PARAMS: dict = {
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "tree_method": "hist",
    "max_bin": 128,
    "nthread": 128,
    "seed": 42,
    "disable_default_eval_metric": 1,
}

_DEFAULT_NUM_BOOST_ROUND = 500
_DEFAULT_EARLY_STOPPING = 30


class _TqdmTrainingCallback(xgb.callback.TrainingCallback):
    """用 tqdm 展示 boosting round 进度。"""

    def __init__(self, total_rounds: int, desc: str):
        self.total_rounds = total_rounds
        self.desc = desc
        self.pbar = None

    def before_training(self, model):
        if tqdm is not None:
            self.pbar = tqdm(
                total=self.total_rounds,
                desc=self.desc,
                dynamic_ncols=True,
                leave=False,
            )
        return model

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if self.pbar is None:
            return False
        self.pbar.n = epoch + 1
        val_cost = evals_log.get("val", {}).get("cost", None)
        if val_cost:
            self.pbar.set_postfix({"val.cost": f"{val_cost[-1]:.6f}"})
        self.pbar.refresh()
        return False

    def after_training(self, model):
        if self.pbar is not None:
            best_it = getattr(model, "best_iteration", None)
            if best_it is not None:
                self.pbar.n = min(best_it + 1, self.total_rounds)
                self.pbar.refresh()
            self.pbar.close()
        return model


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
    task_desc: str = "",
    data_workers: int = 1,
    time_range: tuple[str, str] | None = None,
) -> None:
    """训练单个模型（一种 factor_pool × n_groups × ret_horizon 组合）。"""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[{task_desc}] stage=build_penalty")

    P = build_penalty_matrix(n_groups=n_groups, **penalty_kwargs)
    np.save(os.path.join(out_dir, "penalty_matrix.npy"), P)

    cost_obj = make_cost_obj(P)
    cost_eval = make_cost_eval(P)

    print(f"[{task_desc}] stage=build_train  days={len(train_dates)} stride={stride} workers={data_workers}")
    train_df = build_dataset(
        factor_root,
        base_root,
        fc_to_fn,
        train_dates,
        ret_col,
        n_groups=n_groups,
        stride=stride,
        verbose=True,
        desc=f"train_data[{task_desc}]",
        n_workers=data_workers,
        time_range=time_range,
    )
    print(f"[{task_desc}] 训练集：{len(train_df):,} 样本")

    print(f"[{task_desc}] stage=build_val    days={len(val_dates)} stride={stride} workers={data_workers}")
    val_df = build_dataset(
        factor_root,
        base_root,
        fc_to_fn,
        val_dates,
        ret_col,
        n_groups=n_groups,
        stride=stride,
        verbose=True,
        desc=f"val_data[{task_desc}]",
        n_workers=data_workers,
        time_range=time_range,
    )
    print(f"[{task_desc}] 验证集：{len(val_df):,} 样本")

    if train_df.empty or val_df.empty:
        print(f"[{task_desc}] [跳过] 数据不足")
        return

    print(f"[{task_desc}] stage=prepare_features")
    meta_cols = {"Date", "SampleTime", "SecurityID", "label", ret_col}
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    print(f"[{task_desc}] 特征数：{len(feature_cols)}")

    with open(os.path.join(out_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols) + "\n")

    X_train = train_df[feature_cols].astype(np.float32).values
    y_train = train_df["label"].values.astype(np.int32)
    X_val = val_df[feature_cols].astype(np.float32).values
    y_val = val_df["label"].values.astype(np.int32)

    print(f"[{task_desc}] stage=make_dmatrix")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, missing=np.nan)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, missing=np.nan)

    del X_train, X_val, train_df, val_df

    params = {**xgb_params, "num_class": n_groups}
    print(
        f"[{task_desc}] stage=train_xgboost num_class={n_groups}, features={len(feature_cols)}, "
        f"max_rounds={num_boost_round}, early_stop={early_stopping_rounds}, "
        f"nthread={params.get('nthread')}"
    )

    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name="cost",
            save_best=True,
            maximize=False,
        )
    ]
    if tqdm is not None:
        callbacks.append(_TqdmTrainingCallback(num_boost_round, desc=f"boost[{task_desc}]"))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        obj=cost_obj,
        custom_metric=cost_eval,
        callbacks=callbacks,
        verbose_eval=verbose_eval if tqdm is None else False,
    )

    print(f"[{task_desc}] stage=save_model")
    model_path = os.path.join(out_dir, "model.ubj")
    model.save_model(model_path)
    print(f"[{task_desc}] 模型已保存：{model_path}  (best_iteration={model.best_iteration})")

    split_info = {
        "ret_col": ret_col,
        "n_groups": n_groups,
        "stride": stride,
        "n_train_dates": len(train_dates),
        "n_val_dates": len(val_dates),
        "n_features": len(feature_cols),
        "best_iteration": model.best_iteration,
        "penalty_kwargs": penalty_kwargs,
        "nthread": params.get("nthread"),
    }
    with open(os.path.join(out_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)


def run_xgb_train(
    factor_root: str,
    base_root: str,
    eval_root: str,
    factor_pools: list[str] | None = None,
    n_groups_list: list[int] | None = None,
    ret_horizons: list[str] | None = None,
    stride: int | None = None,
    pools_path: str | None = None,
    extra_features: set[str] | None = None,
    extra_features_tag: str | None = None,
    xgb_params: dict | None = None,
    num_boost_round: int = _DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING,
    penalty_kwargs: dict | None = None,
    dates: list[str] | None = None,
    verbose_eval: int = 20,
    force: bool = False,
    data_workers: int = 1,
    slot: str | None = None,
) -> None:
    factor_pools = factor_pools or ["all", "union", "intersection"]
    n_groups_list = n_groups_list or [10, 20]
    ret_horizons = ret_horizons or list(_RET_HORIZONS.keys())
    penalty_kwargs = penalty_kwargs or {}
    params = {**_DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    # ── 输出目录前缀（额外特征时加后缀）────────────────────────────────────────
    _xgb_base = f"xgb_quantile_{extra_features_tag}" if extra_features_tag else "xgb_quantile"

    # ── 时段路由 ─────────────────────────────────────────────────────────────────
    if slot is not None:
        if slot not in SLOT_RANGES:
            raise ValueError(f"未知时段 slot={slot!r}，可选：{sorted(SLOT_RANGES)}")
        t_start, t_end, default_stride = SLOT_RANGES[slot]
        time_range = (t_start, t_end)
        # 用户未显式指定 stride 时，使用时段推荐值（约每天 20 个截面）
        stride = stride if stride is not None else default_stride
        xgb_root = os.path.join(eval_root, f"{_xgb_base}_slot", slot)
        print(f"[xgb_train] 分时段模式 slot={slot}，时间范围 {t_start}–{t_end}，stride={stride}")
    else:
        time_range = None
        stride = stride if stride is not None else 100  # 全天默认 stride
        xgb_root = os.path.join(eval_root, _xgb_base)

    print(f"[xgb_train] 输出目录：{xgb_root}")
    if extra_features_tag:
        print(f"[xgb_train] 额外特征集={extra_features_tag}，追加特征数={len(extra_features or set())}")

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

    os.makedirs(xgb_root, exist_ok=True)
    with open(os.path.join(xgb_root, "date_split.json"), "w") as f:
        json.dump({"train": train_dates, "val": val_dates}, f, indent=2)

    tasks: list[tuple[str, int, str, str, dict[str, str]]] = []
    for factor_pool in factor_pools:
        fc_to_fn = get_factor_cols_for_pool(
            factor_root,
            factor_pool,
            pools_path=pools_path,
            extra_features=extra_features,
        )
        print(f"\n=== factor_pool={factor_pool}  特征数={len(fc_to_fn)} ===")
        if not fc_to_fn:
            print("  [跳过] 没有找到因子列")
            continue

        for n_groups in n_groups_list:
            for ret_h in ret_horizons:
                out_dir = os.path.join(xgb_root, factor_pool, f"g{n_groups}", ret_h)
                model_path = os.path.join(out_dir, "model.ubj")
                if not force and os.path.exists(model_path):
                    print(f"  [已存在，跳过] {out_dir}")
                    continue
                tasks.append((factor_pool, n_groups, ret_h, out_dir, fc_to_fn))

    if not tasks:
        print("\n没有需要训练的模型。")
        return

    outer = tqdm(tasks, desc="xgb_train", dynamic_ncols=True) if tqdm is not None else tasks
    total = len(tasks)
    for idx, (factor_pool, n_groups, ret_h, out_dir, fc_to_fn) in enumerate(outer, start=1):
        ret_col = _RET_HORIZONS[ret_h]
        task_desc = f"{idx}/{total} pool={factor_pool} g={n_groups} {ret_h}"
        if tqdm is not None:
            outer.set_postfix_str(f"pool={factor_pool} g={n_groups} {ret_h}")
        print(f"\n--- {task_desc} ---")
        _train_one(
            factor_root,
            base_root,
            fc_to_fn,
            train_dates,
            val_dates,
            out_dir,
            ret_col,
            n_groups,
            stride,
            params,
            num_boost_round,
            early_stopping_rounds,
            penalty_kwargs,
            verbose_eval,
            task_desc=task_desc,
            data_workers=data_workers,
            time_range=time_range,
        )

    print("\n所有模型训练完成。")
