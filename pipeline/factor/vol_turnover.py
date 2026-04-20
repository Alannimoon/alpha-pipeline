"""
vol_turnover — 换手与MA特征（XGBoost输入特征，无需IC评价）

16个特征
--------
换手相关（需要 float_shares，8列）：

  vt_ret_N20/60/100/300
      = (CumVolume[t] - CumVolume[t-N]) / FREE_SHARES / (Price[t]/Price[t-N] - 1)
      分母 |Price[t]/Price[t-N] - 1| < 1e-8 时输出 NaN
      前 N tick 输出 NaN

  vt_high_ret / vt_low_ret
      对每个 t，在过去300 tick内枚举所有20 tick子窗口 [s, s+19]（s ∈ [t-299, t-20]）
      s* = argmax/argmin(CumVolume[s+19] - CumVolume[s])
      feat = Price[s*+19] / Price[s*] - 1

  vt_high_vs_now / vt_low_vs_now
      feat = Price[t] / nanmean(Price[s*:s*+20]) - 1
      （复用上面的 s*）

  前 299 tick 输出 NaN

MA（8列）：

  ma_20 / ma_60 / ma_100 / ma_300 / ma_600 / ma_900 / ma_1200 / ma_1500
      = nanmean(Price[t-N+1:t+1])，含当前tick的过去N tick均值
      全NaN窗口 → NaN；窗口不足 → NaN

数据来源：base/{date}.parquet + data/float_shares_filled.parquet
输出目录：result/factor/vol_turnover/{date}.parquet
"""

import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ── float_shares 模块级缓存（每个 worker 进程加载一次）────────────────────────

_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FLOAT_PATH = os.path.join(_ROOT, "data", "float_shares_filled.parquet")

_FS: dict[tuple[str, str], float] = {}


def _load_float_shares() -> None:
    df = pd.read_parquet(_FLOAT_PATH)
    for row in df.itertuples(index=False):
        _FS[(str(row.SecurityID), str(row.Date))] = float(row.FREE_SHARES)


_load_float_shares()

# ── 常量 ──────────────────────────────────────────────────────────────────────

VT_WINS          = [20, 60, 100, 300]
MA_WINS          = [20, 60, 100, 300, 600, 900, 1200, 1500]
EPS              = 1e-8
SRCH_W           = 300   # 特征5-8：向前搜索范围（tick）
SUB_W            = 20    # 特征5-8：子窗口大小（tick）
N_CAND           = SRCH_W - SUB_W   # = 280 个候选起始位置
MAX_INVALID_RATIO = 0.10  # 窗口内无效 tick 比例上限，与其他因子保持一致


# ── 主计算函数 ─────────────────────────────────────────────────────────────────

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日 DataFrame（来自 base/{date}.parquet）
    输出：只含 16 个特征列的 DataFrame，index 与输入对齐
    """
    n     = len(df)
    date  = str(df["Date"].iloc[0])
    secid = str(df["SecurityID"].iloc[0])

    can_use   = df["CanUsePrice"].to_numpy(bool)
    price_raw = df["Price"].to_numpy(np.float64)
    price_v   = np.where(can_use, price_raw, np.nan)   # 无效 tick → NaN
    cumvol    = df["CumVolume"].to_numpy(np.float64)
    free_sh   = _FS.get((secid, date), np.nan)

    out: dict[str, np.ndarray] = {}

    # ── 特征 1-4: vt_ret_N ───────────────────────────────────────────────────
    for N in VT_WINS:
        col = f"vt_ret_N{N}"
        if n <= N:
            out[col] = np.full(n, np.nan)
            continue
        vol_diff = cumvol[N:] - cumvol[:-N]        # (n-N,)
        ret      = price_v[N:] / price_v[:-N] - 1  # NaN when either endpoint invalid
        with np.errstate(invalid="ignore", divide="ignore"):
            feat = vol_diff / free_sh / ret
        feat[np.abs(ret) < EPS] = np.nan
        out[col] = np.concatenate([np.full(N, np.nan), feat])

    # ── 共用：累计价格和 + 有效计数（供特征5-8和MA复用）────────────────────
    valid    = np.isfinite(price_v).astype(np.float64)
    p_fill   = np.where(np.isfinite(price_v), price_v, 0.0)
    cum_p    = np.concatenate([[0.0], np.cumsum(p_fill)])   # (n+1,)
    cum_cnt  = np.concatenate([[0.0], np.cumsum(valid)])    # (n+1,)

    # ── 特征 5-8: vt_high/low ────────────────────────────────────────────────
    # t ≥ 299（0-indexed）才有效；tick t 对应 wins_vol 行 j = t - 299
    n_v58 = max(0, n - SRCH_W + 1)   # = max(0, n-299)

    if n_v58 > 0:
        # vol20[i] = 成交量在子窗口 [i, i+SUB_W) 内的增量
        vol20 = cumvol[SUB_W - 1:] - cumvol[:n - SUB_W + 1]  # (n-19,)

        # pm20[i] = nanmean(Price[i:i+SUB_W])，有效 tick 不足 90% 时为 NaN
        sw          = SUB_W
        min_v_sub   = np.ceil(sw * (1 - MAX_INVALID_RATIO))   # = 18
        p_sums = cum_p[sw:] - cum_p[:n - sw + 1]     # (n-19,)
        p_cnts = cum_cnt[sw:] - cum_cnt[:n - sw + 1]
        pm20   = np.where(p_cnts >= min_v_sub, p_sums / p_cnts, np.nan)

        # wins_vol[j] = vol20[j:j+N_CAND]，对应 tick t = j + 299
        # sliding_window_view 给出 (n-298, 280) 行，取前 n_v58 行（j = 0..n-300）
        wins_vol = sliding_window_view(vol20, N_CAND)[:n_v58]  # (n_v58, 280)

        rel_high = wins_vol.argmax(axis=1)   # (n_v58,)，窗口内偏移
        rel_low  = wins_vol.argmin(axis=1)

        j_arr  = np.arange(n_v58)
        s_high = j_arr + rel_high   # s* 在 price_v / pm20 中的绝对下标
        s_low  = j_arr + rel_low

        with np.errstate(invalid="ignore", divide="ignore"):
            hi_ret = price_v[s_high + SUB_W - 1] / price_v[s_high] - 1
            lo_ret = price_v[s_low  + SUB_W - 1] / price_v[s_low]  - 1
            p_now  = price_v[SRCH_W - 1:]   # price[t]，shape (n_v58,)
            hi_now = p_now / pm20[s_high] - 1
            lo_now = p_now / pm20[s_low]  - 1

        pad = np.full(SRCH_W - 1, np.nan)
        out["vt_high_ret"]    = np.concatenate([pad, hi_ret])
        out["vt_low_ret"]     = np.concatenate([pad, lo_ret])
        out["vt_high_vs_now"] = np.concatenate([pad, hi_now])
        out["vt_low_vs_now"]  = np.concatenate([pad, lo_now])
    else:
        for col in ["vt_high_ret", "vt_low_ret", "vt_high_vs_now", "vt_low_vs_now"]:
            out[col] = np.full(n, np.nan)

    # ── MA 特征 ──────────────────────────────────────────────────────────────
    for N in MA_WINS:
        col     = f"ma_{N}"
        ma      = np.full(n, np.nan)
        min_v_n = np.ceil(N * (1 - MAX_INVALID_RATIO))   # 有效 tick 下限
        if n >= N:
            sums   = cum_p[N:]   - cum_p[:n - N + 1]    # (n-N+1,)
            counts = cum_cnt[N:] - cum_cnt[:n - N + 1]
            ma[N - 1:] = np.where(counts >= min_v_n, sums / counts, np.nan)
        out[col] = ma

    return pd.DataFrame(out, index=df.index)
