"""
market_state — 市场状态特征（XGBoost 输入特征，无需 IC 评价）

7 个特征
--------
  ret_from_open        当前 WMID / 今日开盘价 - 1
  ret_from_prev_close  当前 WMID / 昨收前复权价 - 1
  time_since_open      距开盘交易秒数（下午段扣除午休 5400 s）
  high45_drawdown      过去 45 min 最高 WMID → 当前跌幅（≤ 0）
  time_since_high45    距过去 45 min 最高价 tick 的交易秒数
  low45_rally          过去 45 min 最低 WMID → 当前涨幅（≥ 0）
  time_since_low45     距过去 45 min 最低价 tick 的交易秒数

实现要点
--------
- 时间窗口：3 s 格数据中午休无 tick，滚动 900 ticks = 精确 45 min 交易时间
- time_since_open：wall clock 差值，下午段减去午休 5400 s
- time_since_high45 / time_since_low45：tick 距离 × 3 s，与 wall clock 等价
  （同理：全窗口内若有相同最高/最低价，取最近的那次）
- 前 899 tick（窗口未满）：high45/low45 系列输出 NaN
- CanUsePrice=False 时：ret_from_open / ret_from_prev_close / high45 系列均 NaN
  time_since_open 不依赖价格，全时段均输出有效值

数据来源：base/{date}.parquet
输出目录：result/factor/market_state/{date}.parquet
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ── 常量 ──────────────────────────────────────────────────────────────────────

WINDOW_45M = 900   # 3 s/tick × 900 = 2700 s = 45 min
OPEN_SEC   = 34200  # 09:30:00 从午夜起的秒数
NOON_GAP   = 5400   # 11:30:00 → 13:00:00 = 90 min 午休
PM_START   = 46800  # 13:00:00 从午夜起的秒数


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _sample_time_to_sec(st: pd.Series) -> np.ndarray:
    """'HH:MM:SS' → 从午夜起的秒数（整数）"""
    parts = st.str.split(":", expand=True).astype(int)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy(np.int64)


# ── 主计算函数 ─────────────────────────────────────────────────────────────────

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票单日 DataFrame（来自 base/{date}.parquet）
    输出：只含 7 个特征列的 DataFrame，index 与输入对齐
    """
    n         = len(df)
    can_use   = df["CanUsePrice"].to_numpy(bool)
    price_raw = df["Price"].to_numpy(np.float64)
    price_v   = np.where(can_use, price_raw, np.nan)   # 无效 tick → NaN

    # ── 特征 1：ret_from_open ─────────────────────────────────────────────────
    open_col = df["OpenPrice"].to_numpy(np.float64)
    valid_open = open_col[np.isfinite(open_col) & (open_col > 0)]
    open_val   = valid_open[0] if len(valid_open) > 0 else np.nan

    if np.isfinite(open_val) and open_val > 0:
        ret_from_open = np.where(can_use, price_v / open_val - 1, np.nan)
    else:
        ret_from_open = np.full(n, np.nan)

    # ── 特征 2：ret_from_prev_close ───────────────────────────────────────────
    pre_close = df["PreCloPrice"].to_numpy(np.float64)
    ret_from_prev_close = np.where(
        can_use & np.isfinite(pre_close) & (pre_close > 0),
        price_v / pre_close - 1,
        np.nan,
    )

    # ── 特征 3：time_since_open ───────────────────────────────────────────────
    wall_sec = _sample_time_to_sec(df["SampleTime"])
    is_pm    = wall_sec >= PM_START
    time_since_open = (wall_sec - OPEN_SEC - np.where(is_pm, NOON_GAP, 0)).astype(float)

    # ── 特征 4–7：45 min 滚动窗口 ─────────────────────────────────────────────
    W = WINDOW_45M

    if n >= W:
        # sentinel 填充，使 argmax/argmin 忽略无效 tick
        p_for_max = np.where(np.isfinite(price_v), price_v, -np.inf)
        p_for_min = np.where(np.isfinite(price_v), price_v,  np.inf)

        # shape (n-W+1, W)，index 0=最旧 tick，W-1=最新 tick
        wins_max = sliding_window_view(p_for_max, W)
        wins_min = sliding_window_view(p_for_min, W)
        wins_raw = sliding_window_view(price_v,   W)

        window_has_valid = np.any(np.isfinite(wins_raw), axis=1)   # (n-W+1,)
        roll_max_val     = wins_max.max(axis=1)                     # (n-W+1,)
        roll_min_val     = wins_min.min(axis=1)

        # 翻转窗口后取 argmax/argmin → 返回"从最新 tick 起的距离"（处理等值时取最近）
        # wins_max_flip[i, 0] 是最新 tick，wins_max_flip[i, W-1] 是最旧 tick
        wins_max_flip = np.flip(wins_max, axis=1)
        wins_min_flip = np.flip(wins_min, axis=1)
        ticks_since_high = wins_max_flip.argmax(axis=1).astype(float)   # ticks
        ticks_since_low  = wins_min_flip.argmin(axis=1).astype(float)

        # 当前 tick（窗口尾端）需 CanUsePrice=True 才计算 drawdown/rally
        cur_valid = can_use[W - 1:]

        _dd_valid = window_has_valid & cur_valid & np.isfinite(roll_max_val) & (roll_max_val > 0)
        _rl_valid = window_has_valid & cur_valid & np.isfinite(roll_min_val) & (roll_min_val > 0)

        high45_dd_tail   = np.where(_dd_valid, price_v[W - 1:] / roll_max_val - 1, np.nan)
        time_high45_tail = np.where(window_has_valid, ticks_since_high * 3.0, np.nan)
        low45_rl_tail    = np.where(_rl_valid, price_v[W - 1:] / roll_min_val - 1, np.nan)
        time_low45_tail  = np.where(window_has_valid, ticks_since_low  * 3.0, np.nan)

        pad = np.full(W - 1, np.nan)
        high45_drawdown   = np.concatenate([pad, high45_dd_tail])
        time_since_high45 = np.concatenate([pad, time_high45_tail])
        low45_rally       = np.concatenate([pad, low45_rl_tail])
        time_since_low45  = np.concatenate([pad, time_low45_tail])
    else:
        # 单日 tick 数不足 900（极少数停牌半天情况）
        high45_drawdown   = np.full(n, np.nan)
        time_since_high45 = np.full(n, np.nan)
        low45_rally       = np.full(n, np.nan)
        time_since_low45  = np.full(n, np.nan)

    return pd.DataFrame(
        {
            "ret_from_open":       ret_from_open,
            "ret_from_prev_close": ret_from_prev_close,
            "time_since_open":     time_since_open,
            "high45_drawdown":     high45_drawdown,
            "time_since_high45":   time_since_high45,
            "low45_rally":         low45_rally,
            "time_since_low45":    time_since_low45,
        },
        index=df.index,
    )
