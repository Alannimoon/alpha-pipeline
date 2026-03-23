"""
Alpha Pipeline — 因子评估前端。

启动方式（在项目根目录下）：
    streamlit run app/main.py
"""

import os
import sys

# 确保项目根目录在 sys.path，使 config / pipeline 可正常导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from data import (
    available_factors, load_ic_stats, sort_factor_cols,
    load_ts_daily, load_cs_intraday,
)
from charts import ic_summary_chart, ts_daily_chart, cs_intraday_chart

# ── 页面配置 ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Alpha Pipeline", layout="wide")
st.title("Alpha Pipeline — 因子评估")

factors = available_factors()
if not factors:
    st.error("暂无因子结果，请先运行 `python run.py ic_stats --factor <name>`。")
    st.stop()

# ── 侧边栏 ────────────────────────────────────────────────────────────────────

st.sidebar.header("全局参数")
ic_type     = st.sidebar.radio("IC 类型", ["CS", "TS"]).lower()
ret_horizon = st.sidebar.selectbox("收益率窗口", ["ret100", "ret200", "ret300"])
session     = st.sidebar.selectbox("Session", ["all", "am", "pm"])

# ── Tab 布局 ──────────────────────────────────────────────────────────────────

tab_summary, tab_detail = st.tabs(["📊 IC 汇总", "📈 时序详情"])


# ── Tab 1：IC 汇总 ────────────────────────────────────────────────────────────

with tab_summary:
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.multiselect("因子", factors, default=factors)
    with col2:
        metric = st.radio("指标", ["ic_mean", "icir"], horizontal=True)

    if not selected:
        st.info("请至少选择一个因子。")
    else:
        dfs = {f: load_ic_stats(f, ic_type) for f in selected}
        dfs = {f: df for f, df in dfs.items() if not df.empty}
        if dfs:
            st.plotly_chart(
                ic_summary_chart(dfs, metric, ret_horizon, session),
                use_container_width=True,
            )
            with st.expander("查看数据表"):
                combined = pd.concat(
                    [df.assign(factor_name=f) for f, df in dfs.items()],
                    ignore_index=True,
                )
                sub = combined[
                    (combined["ret_horizon"] == ret_horizon)
                    & (combined["session"] == session)
                ].sort_values(["factor_name", "factor_window"])
                st.dataframe(sub, use_container_width=True)
        else:
            st.warning("所选因子暂无数据。")


# ── Tab 2：时序详情 ───────────────────────────────────────────────────────────

with tab_detail:
    col1, col2 = st.columns(2)
    with col1:
        factor = st.selectbox("因子", factors)
    with col2:
        stats_df = load_ic_stats(factor, ic_type)
        if stats_df.empty:
            st.warning("该因子暂无 ic_stats 结果。")
            st.stop()
        factor_cols = sort_factor_cols(stats_df["factor_col"].unique().tolist())
        factor_col  = st.selectbox("因子窗口", factor_cols)

    if ic_type == "ts":
        # TS-IC：展示跨日的每日均值趋势
        st.caption("每日所有股票 TS-IC 的截面均值，反映因子跨日稳定性。")
        daily_df = load_ts_daily(factor, ret_horizon, session)
        if daily_df.empty:
            st.warning("暂无数据。")
        else:
            st.plotly_chart(ts_daily_chart(daily_df, factor_col), use_container_width=True)
            st.caption(f"共 {len(daily_df)} 个交易日")

    else:
        # CS-IC：展示日内 IC 模式（同一时刻跨所有日期的均值）
        st.caption("每个时间点在所有日期上的 CS-IC 均值，反映因子在日内不同时段的截面预测力。")
        intraday_df = load_cs_intraday(factor, ret_horizon, session)
        if intraday_df.empty:
            st.warning("暂无数据。")
        else:
            st.plotly_chart(cs_intraday_chart(intraday_df, factor_col), use_container_width=True)
            st.caption(f"共 {len(intraday_df)} 个时间点")
