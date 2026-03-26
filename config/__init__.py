"""
全局配置。
所有路径和参数集中在这里，修改配置只需改这一个文件。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 数据路径 ─────────────────────────────────────────────────────────────────
# 原始分钟线数据：data_min/{secid}.csv（每只股票一个文件，含全年数据）
# 服务器路径示例：/data3/aiquanta/data/1min_data/2022
DATA_ROOT   = os.path.join(ROOT, "data_min")

BASE_ROOT   = os.path.join(ROOT, "result", "base")
FACTOR_ROOT = os.path.join(ROOT, "result", "factor")
EVAL_ROOT   = os.path.join(ROOT, "result", "eval")

# ── 收益率参数 ────────────────────────────────────────────────────────────────
# 前向收益率窗口（单位：tick，1 min/tick）
RETURN_HORIZONS = [5, 10, 15]

# 窗口名 → 列名映射（供 eval 模块使用）
RET_HORIZONS_MAP = {
    "ret5":  "ret_fwd_5",
    "ret10": "ret_fwd_10",
    "ret15": "ret_fwd_15",
}
