"""
XGBoost 截面分层 — 代价敏感目标函数模块。

惩罚矩阵设计
------------
将 n_groups 个类别分为做空侧（0 .. mid-1）和做多侧（mid .. n_groups-1）。

P[i, j]:
  = 0                         i == j（正确分类，零惩罚）
  = 0                         i, j 均 < neighbor_zero（最差两组互错：零惩罚）
  = |i-j| × within_scale      同侧错分（轻惩罚，线性递增）
  = cross_base + |i-j|        跨越多空边界（重惩罚）

损失函数（期望代价）
------------------
  L_i = Σ_c P[y_i, c] · p_c(f_i)

梯度（对 softmax logit f_k 求偏导）
-----------------------------------
  ∂L_i/∂f_k = p_k · (P[y_i, k] − Σ_c P[y_i, c] · p_c)

Hessian 近似
-----------
  H_k ≈ max(p_k · (1 − p_k), 1e-6)
"""

import numpy as np


def build_penalty_matrix(
    n_groups: int = 10,
    within_scale: float = 0.5,
    cross_base: float = 5.0,
    neighbor_zero: int = 2,
) -> np.ndarray:
    """
    构建 (n_groups, n_groups) 惩罚矩阵。

    Parameters
    ----------
    n_groups      : 分类数（10 或 20）
    within_scale  : 同侧每步惩罚系数（默认 0.5）
    cross_base    : 跨边界基础惩罚（默认 5.0），实际值 = cross_base + |i-j|
    neighbor_zero : 最差的前 k 个组之间零惩罚（默认 2，对应 1、2 类互错不惩罚）

    Returns
    -------
    P : (n_groups, n_groups) float64 惩罚矩阵，对角线为 0
    """
    mid = n_groups // 2
    P = np.zeros((n_groups, n_groups), dtype=np.float64)
    for i in range(n_groups):
        for j in range(n_groups):
            dist = abs(i - j)
            if dist == 0:
                continue
            same_side = (i < mid) == (j < mid)
            if same_side:
                # 最差 neighbor_zero 个组彼此间：零惩罚
                if i < neighbor_zero and j < neighbor_zero:
                    P[i, j] = 0.0
                else:
                    P[i, j] = dist * within_scale
            else:
                P[i, j] = cross_base + dist
    return P


def make_cost_obj(penalty_matrix: np.ndarray):
    """
    工厂函数：返回绑定了惩罚矩阵的 XGBoost 自定义目标函数。

    用法
    ----
    obj = make_cost_obj(P)
    xgb.train(params, dtrain, obj=obj, ...)

    XGBoost 调用约定
    ----------------
    - y_pred  : (n_samples, n_classes) 或 (n_samples * n_classes,) logits，均兼容
    - 返回    : (grad, hess)，均为 (n_samples * n_classes,)
    """
    P = penalty_matrix.copy()
    n_cls = P.shape[0]

    def cost_obj(
        y_pred: np.ndarray,
        dtrain,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label().astype(int)
        n = len(labels)

        preds = y_pred.reshape(n, n_cls)

        # 数值稳定的 softmax
        preds = preds - preds.max(axis=1, keepdims=True)
        exp_p = np.exp(preds)
        probs = exp_p / exp_p.sum(axis=1, keepdims=True)   # (n, n_cls)

        # 每个样本的代价向量 P[y_i, :]
        costs = P[labels]                                    # (n, n_cls)

        # 期望代价 E_p[P[y, c]]
        expected = (costs * probs).sum(axis=1, keepdims=True)  # (n, 1)

        # 梯度：∂L/∂f_k = p_k · (P[y,k] − E[cost])
        grad = probs * (costs - expected)                    # (n, n_cls)

        # Hessian 近似：p_k · (1 − p_k)
        hess = np.maximum(probs * (1.0 - probs), 1e-6)      # (n, n_cls)

        return grad, hess  # (n, n_cls) — XGBoost 2.1+ 要求 (n_samples, n_classes)

    return cost_obj


def make_cost_eval(penalty_matrix: np.ndarray):
    """
    返回配套的验证集评估指标：期望代价均值（越小越好）。

    XGBoost 约定：返回 (metric_name: str, metric_value: float)。
    EarlyStopping 按 maximize=False 默认处理（越小越好）。
    """
    P = penalty_matrix.copy()
    n_cls = P.shape[0]

    def cost_eval(y_pred: np.ndarray, dtrain) -> tuple[str, float]:
        labels = dtrain.get_label().astype(int)
        n = len(labels)

        preds = y_pred.reshape(n, n_cls)
        preds = preds - preds.max(axis=1, keepdims=True)
        exp_p = np.exp(preds)
        probs = exp_p / exp_p.sum(axis=1, keepdims=True)

        costs = P[labels]
        expected = (costs * probs).sum(axis=1)   # (n,)
        return "cost", float(expected.mean())

    return cost_eval
