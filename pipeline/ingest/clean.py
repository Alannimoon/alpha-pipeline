"""
清洗模块：在重采样数据基础上，删除不可用的股票日。

删除规则
--------
1. 全天 SuspendLike：该股票日所有 tick 的状态码均为 Suspend 类，整天停牌，无交易价值。
2. 手动覆盖列表（config/drop_overrides.csv）：人工核查后确认需要删除的案例。

仅标记、不自动删除
------------------
- 保留的股票日中 MaxGapSec > 60：写入 data/cleaned/_gap_review.csv 供人工复核。
  确认后手动加入 drop_overrides.csv，下次重跑 clean 时生效。

状态码分类（依据通联 L2 数据结构文档 V4.1）
-------------------------------------------
上交所 InstruStatus（mdl_4_4_0）：
  TradeLike   → TRADE（连续自动撮合）
  AuctionLike → OCALL / CCALL / ICALL / FCALL（各类集合竞价）
  SuspendLike → SUSP / HALT 及其他所有未列出状态

深交所 TradingPhaseCode（mdl_6_28_0，两位字符）：
  第二位 == "1"：全天停牌标志 → SuspendLike（优先判断，覆盖第一位）
  第一位（第二位为"0"时）：
    T → TradeLike（连续竞价）
    O / C → AuctionLike（开盘/收盘集合竞价）
    H → SuspendLike（临时停牌）
    B / S / E / A / V 及其他 → SuspendLike（休市、启动、闭市、盘后等）
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ── 状态码分类 ────────────────────────────────────────────────────────────────

# 上交所：明确为集合竞价的状态码
_SH_AUCTION = {"OCALL", "CCALL", "ICALL", "FCALL"}

def _classify_tick(status: str, market: str) -> str:
    """单个 tick 状态码 → TradeLike / AuctionLike / SuspendLike"""
    s = str(status).strip()
    if market == "SH":
        if s == "TRADE":
            return "TradeLike"
        if s in _SH_AUCTION:
            return "AuctionLike"
        return "SuspendLike"   # SUSP / HALT / START / BREAK / 其他

    elif market == "SZ":
        # 第二位 "1" = 全天停牌标志，优先
        if len(s) >= 2 and s[1] == "1":
            return "SuspendLike"
        c = s[0] if s else ""
        if c == "T":
            return "TradeLike"        # 连续竞价
        if c in ("O", "C"):
            return "AuctionLike"      # 开盘/收盘集合竞价
        return "SuspendLike"          # H(临时停牌) / B(休市) / S(启动) / E(闭市) / 其他

    return "SuspendLike"


def _detect_market(df: pd.DataFrame) -> tuple[str, str | None]:
    if "InstruStatus" in df.columns:
        return "SH", "InstruStatus"
    if "TradingPhaseCode" in df.columns:
        return "SZ", "TradingPhaseCode"
    return "UNK", None


def _is_all_day_suspend(df: pd.DataFrame) -> bool:
    """若该股票日所有 tick 均为 SuspendLike，返回 True。"""
    market, status_col = _detect_market(df)
    if status_col is None:
        return False
    return df[status_col].apply(lambda x: _classify_tick(x, market)).eq("SuspendLike").all()


# ── 手动覆盖列表 ──────────────────────────────────────────────────────────────

def load_drop_overrides(path: str) -> list[tuple[str, str]]:
    """
    读取 drop_overrides.csv，返回 (Date, SecurityID_or_wildcard) 列表。
    支持通配符：ALL_SZ / ALL_SH / ALL。
    """
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            date, secid = parts[0].strip(), parts[1].strip()
            if date == "Date":
                continue
            rows.append((date, secid))
    return rows


def _expand_overrides(override_rows: list, day: str, all_secids: list[str]) -> set[str]:
    """展开当天的手动覆盖条目，返回需删除的 SecurityID 集合。"""
    to_drop = set()
    for date, secid in override_rows:
        if date != day:
            continue
        if secid == "ALL":
            to_drop.update(all_secids)
        elif secid == "ALL_SZ":
            to_drop.update(s for s in all_secids if s[:1] in ("0", "3"))
        elif secid == "ALL_SH":
            to_drop.update(s for s in all_secids if s[:1] == "6")
        else:
            to_drop.add(secid.zfill(6))
    return to_drop


# ── 单文件判定 ────────────────────────────────────────────────────────────────

def _clean_one(args) -> dict:
    in_path, out_path, day, secid, override_drop = args
    base = {"Date": day, "SecurityID": secid}

    try:
        if secid in override_drop:
            return {**base, "Kept": False, "DropReason": "MANUAL_OVERRIDE", "MaxGapSec": None}

        df = pd.read_csv(in_path)
        if len(df) == 0:
            return {**base, "Kept": False, "DropReason": "EMPTY_FILE", "MaxGapSec": None}

        if _is_all_day_suspend(df):
            return {**base, "Kept": False, "DropReason": "ALL_DAY_SUSPEND", "MaxGapSec": None}

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)

        max_gap = df["GapSec"].max() if "GapSec" in df.columns else None
        return {
            **base,
            "Kept": True,
            "DropReason": "",
            "MaxGapSec": round(float(max_gap), 1) if pd.notna(max_gap) else None,
        }

    except Exception as e:
        return {**base, "Kept": False, "DropReason": f"ERROR: {e}", "MaxGapSec": None}


# ── 批量入口 ──────────────────────────────────────────────────────────────────

def run_clean(
    sampled_root: str,
    cleaned_root: str,
    override_csv: str,
    gap_threshold: float = 60.0,
    dates: list[str] | None = None,
    max_workers: int | None = None,
):
    """
    批量清洗。

    Parameters
    ----------
    sampled_root  : 重采样数据根目录
    cleaned_root  : 输出根目录
    override_csv  : drop_overrides.csv 路径
    gap_threshold : 保留的股票日中 MaxGapSec 超过此值的，写入 _gap_review.csv
    dates         : 指定日期列表；None 时自动扫描
    max_workers   : 并行进程数
    """
    os.makedirs(cleaned_root, exist_ok=True)
    override_rows = load_drop_overrides(override_csv)

    if dates is None:
        dates = sorted(
            d for d in os.listdir(sampled_root)
            if len(d) == 8 and d.isdigit()
            and os.path.isdir(os.path.join(sampled_root, d))
        )

    all_tasks = []
    for day in dates:
        in_dir  = os.path.join(sampled_root, day)
        out_dir = os.path.join(cleaned_root, day)
        stock_files = sorted(
            f for f in os.listdir(in_dir)
            if f.endswith(".csv") and not f.startswith("_")
        )
        if not stock_files:
            continue
        all_secids    = [os.path.splitext(f)[0].zfill(6) for f in stock_files]
        override_drop = _expand_overrides(override_rows, day, all_secids)
        os.makedirs(out_dir, exist_ok=True)
        for f in stock_files:
            all_tasks.append((
                os.path.join(in_dir, f),
                os.path.join(out_dir, f),
                day,
                os.path.splitext(f)[0].zfill(6),
                override_drop,
            ))

    if max_workers == 1:
        iter_ = tqdm(all_tasks, desc="clean") if tqdm else all_tasks
        results = [_clean_one(t) for t in iter_]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs  = [pool.submit(_clean_one, t) for t in all_tasks]
            iter_ = tqdm(as_completed(futs), total=len(futs), desc="clean") \
                    if tqdm else as_completed(futs)
            results = [f.result() for f in iter_]

    report = pd.DataFrame(results).sort_values(["Date", "SecurityID"]).reset_index(drop=True)

    # ── 清洗明细：仅包含被删除的股票日 ───────────────────────────────────────
    dropped_df = report[~report["Kept"]][["Date", "SecurityID", "DropReason"]].reset_index(drop=True)
    dropped_df.to_csv(os.path.join(cleaned_root, "_dropped.csv"), index=False)

    # ── 待复核：保留但 MaxGapSec > threshold 的股票日 ─────────────────────────
    gap_review = report[
        report["Kept"] &
        report["MaxGapSec"].notna() &
        (report["MaxGapSec"] > gap_threshold)
    ][["Date", "SecurityID", "MaxGapSec"]].reset_index(drop=True)
    gap_review_path = os.path.join(cleaned_root, "_gap_review.csv")
    gap_review.to_csv(gap_review_path, index=False)

    # ── 打印摘要 ───────────────────────────────────────────────────────────────
    total   = len(report)
    kept    = int(report["Kept"].sum())
    n_drop  = total - kept
    n_gap   = len(gap_review)

    print(f"清洗完成：共 {total} 个股票日，保留 {kept}，删除 {n_drop}")
    if n_drop:
        print("删除原因：")
        print(dropped_df["DropReason"].value_counts().to_string())
    if n_gap:
        print(f"\n待复核（MaxGapSec > {gap_threshold}s）：{n_gap} 个股票日 → {gap_review_path}")
    else:
        print("无待复核的大间隔股票日。")
