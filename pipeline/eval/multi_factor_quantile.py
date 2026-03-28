"""
多因子合成分层模块。

流程
----
1. 从各因子的 cs_ic_stats.csv（session=all）读取 IC 均值，
   筛选 |ic_mean| >= threshold 的因子列，按 ret_horizon 分组。
2. IC 权重归一化：w_i = ic_mean_i / Σ|ic_mean_j|（保留符号，|w| 之和=1）。
3. 逐日读取各因子宽表，计算截面分位数得分（0~1），再 IC 加权求和
   得到每只股票每时刻的合成分数。
4. 按合成分数分成 N_GROUPS=10 层，计算各层前向收益均值。
5. 输出与 cs_quantile 相同规格的汇总文件和图表。

输出目录
--------
{eval_root}/multi_factor_quantile/ret{100|200|300}/
    _weights.csv            筛选结果：factor_col, ic_mean, weight
    {date}.csv              Date, SampleTime, composite_mean, composite_std,
                            n_valid, g1..g10
    _daily.csv              每日各层均值
    _summary.csv            跨日整体均值
    _cum_daily.csv          跨日累计收益（含 long_short = g10-g1）
    _cum_tick_{date}.csv    日内 tick 级累计
    _chart_tick.png         跨日 tick 连续曲线图
"""

import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


N_GROUPS = 10

_RET_HORIZONS = {
    "ret100": "ret_fwd_100",
    "ret200": "ret_fwd_200",
    "ret300": "ret_fwd_300",
}

# ── IC 权重加载 ────────────────────────────────────────────────────────────────

def load_ic_weights(
    ic_stats_root: str,
    threshold: float = 0.02,
) -> dict[str, dict]:
    """
    读取所有因子的 cs_ic_stats.csv，筛选 session=all 且 |ic_mean| >= threshold。

    Returns
    -------
    weights : dict  ret_horizon → {
        'weights':       {factor_col: normalized_weight},
        'factor_names':  {factor_col: factor_name},   ← 用于定位数据目录
        'ic_means':      {factor_col: raw_ic_mean},
    }
    """
    csv_files = glob.glob(os.path.join(ic_stats_root, "*", "cs_ic_stats.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 cs_ic_stats.csv：{ic_stats_root}")

    dfs = []
    for path in csv_files:
        factor_name = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        df["factor_name"] = factor_name
        dfs.append(df)

    all_stats = pd.concat(dfs, ignore_index=True)
    all_stats = all_stats[all_stats["session"] == "all"].copy()

    result = {}
    for ret_h in _RET_HORIZONS:
        sub = all_stats[
            (all_stats["ret_horizon"] == ret_h) &
            (all_stats["ic_mean"].abs() >= threshold)
        ].copy()

        if sub.empty:
            result[ret_h] = {"weights": {}, "factor_names": {}, "ic_means": {}}
            continue

        total = sub["ic_mean"].abs().sum()
        weights      = {}
        factor_names = {}
        ic_means     = {}
        for _, row in sub.iterrows():
            fc = row["factor_col"]
            weights[fc]      = row["ic_mean"] / total
            factor_names[fc] = row["factor_name"]
            ic_means[fc]     = row["ic_mean"]

        result[ret_h] = {
            "weights":      weights,
            "factor_names": factor_names,
            "ic_means":     ic_means,
        }

    return result


def _save_weights(out_dir: str, ic_info: dict) -> None:
    """将本次筛选的权重表写入 _weights.csv。"""
    rows = [
        {"factor_col": fc, "ic_mean": ic_info["ic_means"][fc], "weight": ic_info["weights"][fc]}
        for fc in ic_info["weights"]
    ]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "_weights.csv"), index=False)


# ── 宽表读取 ───────────────────────────────────────────────────────────────────

def _build_wide_multi(
    factor_root: str,
    factor_names_by_col: dict[str, str],
    day: str,
) -> dict[str, pd.DataFrame]:
    """
    按 factor_name 分组读取单日所有股票文件，返回
    {factor_col: wide_df}，wide_df 的 index=SampleTime，columns=SecurityID。

    同一 factor_name 的所有 factor_col 共享一次 I/O（每股票文件只读一次）。
    """
    # 按 factor_name 分组收集需要的 factor_col
    name_to_cols: dict[str, list[str]] = {}
    for fc, fn in factor_names_by_col.items():
        name_to_cols.setdefault(fn, []).append(fc)

    # {factor_col: {secid: Series}}
    series_by_col: dict[str, dict[str, pd.Series]] = {
        fc: {} for fc in factor_names_by_col
    }

    for factor_name, cols in name_to_cols.items():
        day_dir = os.path.join(factor_root, factor_name, day)
        if not os.path.isdir(day_dir):
            continue

        files = sorted(
            f for f in os.listdir(day_dir)
            if f.endswith(".csv") and not f.startswith("_")
        )
        needed = ["SampleTime", "SecurityID"] + cols

        for fname in files:
            try:
                df = pd.read_csv(
                    os.path.join(day_dir, fname),
                    usecols=needed,
                    dtype={"SampleTime": str, "SecurityID": str},
                )
            except ValueError:
                # 该文件缺少某些列（不同因子类型的列不在同一文件中，跳过）
                continue
            if df.empty:
                continue
            secid   = df["SecurityID"].iloc[0]
            indexed = df.set_index("SampleTime")
            for fc in cols:
                if fc in indexed.columns:
                    series_by_col[fc][secid] = indexed[fc]

    # 组装宽表
    wide: dict[str, pd.DataFrame] = {}
    for fc, sdict in series_by_col.items():
        if sdict:
            wide[fc] = pd.DataFrame(sdict)

    return wide


def _build_ret_wide(
    factor_root: str,
    factor_name: str,
    day: str,
) -> pd.DataFrame | None:
    """
    读取某个 factor_name 目录下单日所有股票的三条收益率列，
    返回 dict {ret_col: wide_df}。
    只需任意一个因子目录即可（返回结构相同）。
    """
    day_dir = os.path.join(factor_root, factor_name, day)
    if not os.path.isdir(day_dir):
        return None

    files = sorted(
        f for f in os.listdir(day_dir)
        if f.endswith(".csv") and not f.startswith("_")
    )
    ret_cols = list(_RET_HORIZONS.values())
    needed   = ["SampleTime", "SecurityID"] + ret_cols

    series: dict[str, dict[str, pd.Series]] = {rc: {} for rc in ret_cols}
    for fname in files:
        try:
            df = pd.read_csv(
                os.path.join(day_dir, fname),
                usecols=needed,
                dtype={"SampleTime": str, "SecurityID": str},
            )
        except ValueError:
            continue
        if df.empty:
            continue
        secid   = df["SecurityID"].iloc[0]
        indexed = df.set_index("SampleTime")
        for rc in ret_cols:
            if rc in indexed.columns:
                series[rc][secid] = indexed[rc]

    if not any(series.values()):
        return None

    return {rc: pd.DataFrame(d) for rc, d in series.items()}


# ── 截面分位数与合成分 ────────────────────────────────────────────────────────

def _percentile_scores(f_mat: np.ndarray) -> np.ndarray:
    """
    f_mat : (T, N)  因子值矩阵（含 NaN）
    返回  : (T, N)  分位数得分矩阵，NaN 股票仍为 NaN，其余在 [0, 1]

    实现：对每行做 argsort(argsort(x))，仅在 isfinite 股票内排名，
    映射到 rank / n_valid。与 cs_quantile 的分组逻辑一致。
    """
    T, N = f_mat.shape
    scores = np.full((T, N), np.nan)
    for t in range(T):
        row   = f_mat[t]
        valid = np.isfinite(row)
        nv    = int(valid.sum())
        if nv == 0:
            continue
        sub_rank = np.argsort(np.argsort(row[valid]))   # 0 ~ nv-1
        scores[t, valid] = sub_rank / nv                # → [0, 1)
    return scores


def _composite_and_groups(
    wide_factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
    r_wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    wide_factors : {factor_col: wide_df (index=SampleTime, cols=SecurityID)}
    weights      : {factor_col: w_i}
    r_wide       : wide_df for one return horizon

    Returns
    -------
    DataFrame  index=SampleTime, cols=[composite_mean, composite_std, n_valid, g1..g10]
    """
    # 统一 index/columns（时间对齐）
    ref_index   = r_wide.index
    ref_columns = r_wide.columns
    T = len(ref_index)
    N = len(ref_columns)

    # 计算加权合成分矩阵 (T, N)
    composite = np.zeros((T, N))
    weight_sum = np.zeros((T, N))   # 累计有效权重（分母，用于处理部分因子 NaN）

    for fc, w in weights.items():
        if fc not in wide_factors:
            continue
        f_wide = wide_factors[fc].reindex(index=ref_index, columns=ref_columns)
        f_mat  = f_wide.to_numpy(np.float64)
        sc_mat = _percentile_scores(f_mat)                  # (T, N)

        valid_mask = np.isfinite(sc_mat)
        composite[valid_mask]   += w * sc_mat[valid_mask]
        weight_sum[valid_mask]  += abs(w)

    # 权重补偿：当某只股票某时刻有部分因子缺失时，归一化到实际有效权重
    # （若全部因子均缺失则保持 0，后续视为 NaN）
    can_use = weight_sum > 1e-9
    composite = np.where(can_use, composite / weight_sum, np.nan)

    # 收益率矩阵
    r_mat = r_wide.to_numpy(np.float64)

    # 十分层
    g_lists  = [[] for _ in range(N_GROUPS)]
    n_valids = []
    c_means  = []
    c_stds   = []

    for t in range(T):
        c_row = composite[t]
        r_row = r_mat[t]

        valid = np.isfinite(c_row)
        nv    = int(valid.sum())
        n_valids.append(nv)
        c_means.append(float(np.nanmean(c_row[valid])) if nv > 0 else np.nan)
        c_stds.append(float(np.nanstd(c_row[valid]))   if nv > 0 else np.nan)

        if nv < N_GROUPS:
            for g in range(N_GROUPS):
                g_lists[g].append(np.nan)
            continue

        c_valid = c_row[valid]
        r_valid = r_row[valid]
        ranks   = np.argsort(np.argsort(c_valid))
        groups  = (ranks * N_GROUPS // nv).clip(0, N_GROUPS - 1)

        for g in range(N_GROUPS):
            ret_g      = r_valid[groups == g]
            valid_ret  = ret_g[np.isfinite(ret_g)]
            g_lists[g].append(float(valid_ret.mean()) if len(valid_ret) > 0 else np.nan)

    out = pd.DataFrame(
        {"composite_mean": c_means, "composite_std": c_stds, "n_valid": n_valids},
        index=ref_index,
    )
    for g in range(N_GROUPS):
        out[f"g{g + 1}"] = g_lists[g]

    out.index.name = "SampleTime"
    return out


# ── 单日计算（供并行调用）────────────────────────────────────────────────────

def _compute_day(
    factor_root: str,
    out_dirs: dict[str, str],
    ic_info_by_horizon: dict[str, dict],
    day: str,
) -> str:
    """
    单日入口：读宽表 → 合成分 → 十分层 → 写 CSV。

    ic_info_by_horizon : {ret_horizon: {'weights':..., 'factor_names':...}}
    out_dirs           : {ret_horizon: output_directory}
    """
    # 收集本次需要的所有 factor_col → factor_name 映射（跨 horizon 合并）
    all_fc_to_fn: dict[str, str] = {}
    for info in ic_info_by_horizon.values():
        all_fc_to_fn.update(info["factor_names"])

    if not all_fc_to_fn:
        return day

    # 一次性读取所有因子宽表
    wide_factors = _build_wide_multi(factor_root, all_fc_to_fn, day)

    # 读取收益率宽表（遍历所有候选因子目录，取第一个成功的）
    ret_wides = None
    for fn in dict.fromkeys(all_fc_to_fn.values()):   # 保序去重
        ret_wides = _build_ret_wide(factor_root, fn, day)
        if ret_wides is not None:
            break
    if ret_wides is None:
        return day

    for ret_h, ret_col in _RET_HORIZONS.items():
        info    = ic_info_by_horizon.get(ret_h, {})
        weights = info.get("weights", {})
        if not weights or ret_col not in ret_wides:
            continue

        r_wide = ret_wides[ret_col]
        if r_wide.empty:
            continue

        df = _composite_and_groups(wide_factors, weights, r_wide)
        df = df.reset_index()
        df.insert(0, "Date", day)

        out_dir = out_dirs[ret_h]
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{day}.csv"), index=False)

    return day


# ── 汇总生成（与 cs_quantile 相同逻辑，适配10组）────────────────────────────

def _build_daily(csv_dir: str) -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return

    rows = []
    for f in files:
        day = os.path.splitext(os.path.basename(f))[0]
        df  = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in df.columns]
        if not g_cols:
            continue
        day_means = df[g_cols].mean()
        row = {"Date": day}
        for gc in g_cols:
            row[gc] = day_means.get(gc, np.nan)
        rows.append(row)

    if not rows:
        return
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, "_daily.csv"), index=False)


def _build_summary(csv_dir: str) -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return

    daily_means = []
    for f in files:
        df = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in df.columns]
        if g_cols:
            daily_means.append(df[g_cols].mean())

    if not daily_means:
        return

    overall = pd.DataFrame(daily_means).mean()
    summary = pd.DataFrame([overall.to_dict()])
    summary.insert(0, "label", "composite")
    summary.to_csv(os.path.join(csv_dir, "_summary.csv"), index=False)


def _build_cum_tick(csv_dir: str) -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        return

    file_iter = tqdm(files, desc="cum_tick") if tqdm else files
    for f in file_iter:
        day    = os.path.splitext(os.path.basename(f))[0]
        df     = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in df.columns]
        if not g_cols:
            continue

        sub = df[["SampleTime"] + g_cols].copy()
        sub[g_cols] = sub[g_cols].cumsum()
        sub["long_short"] = sub[f"g{N_GROUPS}"] - sub["g1"]

        sub.to_csv(os.path.join(csv_dir, f"_cum_tick_{day}.csv"), index=False)


def _build_cum_daily(csv_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(csv_dir, "_cum_tick_*.csv")))
    if not files:
        return

    rows = []
    file_iter = tqdm(files, desc="cum_daily") if tqdm else files
    for f in file_iter:
        m = re.search(r"_cum_tick_(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        day = m.group(1)
        df  = pd.read_csv(f, dtype={"SampleTime": str})
        g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in df.columns]
        if not g_cols:
            continue

        # 最后一行 = 日内累计总收益
        last      = df[g_cols + ["long_short"]].iloc[[-1]].copy()
        n_ticks   = int(df[g_cols[0]].notna().sum()) if g_cols else 0
        last["n_ticks"] = n_ticks
        last.insert(0, "Date", day)
        rows.append(last)

    if not rows:
        return

    all_last = (
        pd.concat(rows, ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    g_cols = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in all_last.columns]
    cum_cols = g_cols + ["n_ticks"]
    all_last[cum_cols]    = all_last[cum_cols].cumsum()
    all_last["long_short"] = all_last[f"g{N_GROUPS}"] - all_last["g1"]

    keep = ["Date"] + g_cols + ["long_short", "n_ticks"]
    all_last[keep].to_csv(os.path.join(csv_dir, "_cum_daily.csv"), index=False)


_GROUP_COLORS = [
    "#d62728", "#e07b2a", "#d4a017", "#9acd32", "#3cb371",
    "#2ca02c", "#17becf", "#1f77b4", "#7b52ab", "#8c564b",
]


def _build_cum_tick_chart(csv_dir: str) -> None:
    tick_files = sorted(glob.glob(os.path.join(csv_dir, "_cum_tick_*.csv")))
    if not tick_files:
        return

    daily_path = os.path.join(csv_dir, "_cum_daily.csv")
    if not os.path.exists(daily_path):
        return
    daily_df = pd.read_csv(daily_path, dtype={"Date": str}).set_index("Date")

    g_cols    = [f"g{g}" for g in range(1, N_GROUPS + 1) if f"g{g}" in daily_df.columns]
    all_cols  = g_cols + (["long_short"] if "long_short" in daily_df.columns else [])

    # 拼跨日 tick 数据
    dfs = []
    prev_date = None
    for f in tick_files:
        m = re.search(r"_cum_tick_(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        day = m.group(1)
        df  = pd.read_csv(f, dtype={"SampleTime": str})
        avail_cols = [c for c in all_cols if c in df.columns]
        df = df[["SampleTime"] + avail_cols].copy()
        if df.empty:
            prev_date = day
            continue
        if prev_date is not None and prev_date in daily_df.index:
            offset = daily_df.loc[prev_date, avail_cols]
            df[avail_cols] = df[avail_cols] + offset.values
        df.insert(0, "Date", day)
        dfs.append(df)
        prev_date = day

    if not dfs:
        return

    tick_df = pd.concat(dfs, ignore_index=True)
    x = np.arange(len(tick_df))

    day_starts = tick_df.groupby("Date", sort=False).apply(
        lambda g: g.index[0] - tick_df.index[0], include_groups=False
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, gc in enumerate(g_cols):
        if gc in tick_df.columns:
            ax.plot(x, tick_df[gc],
                    color=_GROUP_COLORS[i % len(_GROUP_COLORS)],
                    alpha=0.7, linewidth=0.7, label=gc)
    if "long_short" in tick_df.columns:
        ax.plot(x, tick_df["long_short"], color="black",
                linewidth=1.2, label=f"L/S(g{N_GROUPS}-g1)")

    tick_positions, tick_labels = [], []
    prev_month = None
    for date, pos in day_starts.items():
        month = date[:6]
        if month != prev_month:
            ax.axvline(pos, color="gray", linewidth=0.4, linestyle="--")
            tick_positions.append(pos)
            tick_labels.append(f"{date[:4]}-{date[4:6]}")
            prev_month = month

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax.set_ylabel("Cumulative Return (tick-level)")
    ax.set_title(f"Multi-Factor Composite  Cross-Day Tick Cumulative Return")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left", ncol=N_GROUPS + 1, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(csv_dir, "_chart_tick.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 批量入口 ───────────────────────────────────────────────────────────────────

def _worker(args) -> str:
    factor_root, out_dirs, ic_info_by_horizon, day = args
    return _compute_day(factor_root, out_dirs, ic_info_by_horizon, day)


def run_multi_factor_quantile(
    factor_root: str,
    eval_root: str,
    ic_stats_root: str,
    threshold: float = 0.02,
    dates: list[str] | None = None,
    max_workers: int | None = None,
):
    """
    Parameters
    ----------
    factor_root   : 因子数据根目录（如 result/factor）
    eval_root     : 评估结果输出根目录（如 result/eval）
    ic_stats_root : cs_ic_stats.csv 的根目录（如 result/ic_stats）
    threshold     : IC 均值绝对值筛选阈值，默认 0.02
    dates         : 指定日期列表；None 时从某个通过筛选的因子目录自动扫描
    max_workers   : 并行进程数
    """
    # 1. 加载权重
    ic_weights = load_ic_weights(ic_stats_root, threshold=threshold)

    # 打印筛选结果
    for ret_h, info in ic_weights.items():
        cols = list(info["weights"].keys())
        print(f"[{ret_h}] 通过筛选因子（{len(cols)} 个）：{cols}")
        for fc, w in info["weights"].items():
            print(f"    {fc:30s}  ic_mean={info['ic_means'][fc]:+.4f}  weight={w:+.4f}")

    # 2. 确定日期列表
    if dates is None:
        # 从任一有效因子目录扫描日期
        any_fn = next(
            (fn for info in ic_weights.values() for fn in info["factor_names"].values()),
            None,
        )
        if any_fn is None:
            print("没有通过筛选的因子，退出。")
            return
        scan_dir = os.path.join(factor_root, any_fn)
        dates = sorted(
            d for d in os.listdir(scan_dir)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(scan_dir, d))
        )

    # 3. 输出目录（每个 ret_horizon 一个子目录）
    base_out = os.path.join(eval_root, "multi_factor_quantile")
    out_dirs: dict[str, str] = {}
    for ret_h in _RET_HORIZONS:
        d = os.path.join(base_out, ret_h)
        os.makedirs(d, exist_ok=True)
        out_dirs[ret_h] = d
        _save_weights(d, ic_weights[ret_h])

    # 4. 并行逐日计算
    tasks = [(factor_root, out_dirs, ic_weights, day) for day in dates]

    if max_workers == 1:
        day_iter = tqdm(tasks, desc="Multi-Factor Quantile") if tqdm else tasks
        for t in day_iter:
            _worker(t)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs  = [pool.submit(_worker, t) for t in tasks]
            inner = (
                tqdm(as_completed(futs), total=len(futs), desc="Multi-Factor Quantile")
                if tqdm else as_completed(futs)
            )
            for fut in inner:
                fut.result()

    # 5. 汇总文件与图表
    for ret_h in _RET_HORIZONS:
        sub_dir = out_dirs[ret_h]
        _build_daily(sub_dir)
        _build_summary(sub_dir)
        _build_cum_tick(sub_dir)
        _build_cum_daily(sub_dir)
        _build_cum_tick_chart(sub_dir)
        print(f"[{ret_h}] 汇总完成：{sub_dir}")

    print(f"多因子分层计算完成：{base_out}")
