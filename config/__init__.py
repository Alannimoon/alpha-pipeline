"""
全局配置。
所有路径和参数集中在这里，修改配置只需改这一个文件。
"""
import os

# 项目根目录（config/ 的上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 数据路径 ────────────────────────────────────────────────────────────────
# 原始快照数据：ROOT/data/20250102/000001.csv, ROOT/data/20250103/...
RAW_ROOT = os.path.join(ROOT, "data")

# 各处理阶段的输出目录
SAMPLED_ROOT  = os.path.join(ROOT, "result", "sampled")
CLEANED_ROOT  = os.path.join(ROOT, "result", "cleaned")
BASE_ROOT     = os.path.join(ROOT, "result", "base")

# ── 采样参数 ────────────────────────────────────────────────────────────────
SAMPLE_FREQ = "3s"

# ── 交易时段 ────────────────────────────────────────────────────────────────
# 采样网格范围，均使用左闭右开区间，即不包含 11:30:00 和 14:57:00
AM_START = "09:30:00"
AM_END   = "11:30:00"   # 不含
PM_START = "13:00:00"
PM_END   = "14:57:00"   # 不含

# ── 清洗参数 ────────────────────────────────────────────────────────────────
# 人工确认的异常股票日列表（不删除时留空即可，文件可以只有注释行）
DROP_OVERRIDES_CSV = os.path.join(ROOT, "config", "drop_overrides.csv")

# MaxGapSec 超过此值的保留股票日会被输出到 _gap_review.csv 供人工复核（不自动删除）
GAP_REVIEW_THRESHOLD = 60.0

# ── 收益率参数 ───────────────────────────────────────────────────────────────
# 前向收益率窗口（单位：tick，3s/tick）
# 100 tick = 5分钟，200 tick = 10分钟，300 tick = 15分钟
RETURN_HORIZONS = [100, 200, 300]

# ── 因子参数 ─────────────────────────────────────────────────────────────────
FACTOR_ROOT = os.path.join(ROOT, "result", "factor")
EVAL_ROOT   = os.path.join(ROOT, "result", "eval")
