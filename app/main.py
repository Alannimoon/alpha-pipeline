"""
Alpha Pipeline — 因子评估前端。

启动方式（在项目根目录下）：
    streamlit run app/main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from data import (
    available_factors, available_cs_dates,
    load_ic_stats, sort_factor_cols,
    load_cs_daily_trend, load_cs_one_day,
)
from charts import ic_summary_chart, cs_daily_trend_chart, cs_intraday_chart

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

tab_summary, tab_cs = st.tabs(["📊 IC 汇总", "📈 截面详情"])


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


# ── Tab 2：截面详情 ───────────────────────────────────────────────────────────

with tab_cs:
    col1, col2, col3 = st.columns(3)

    with col1:
        factor = st.selectbox("因子", factors, key="cs_factor")

    with col2:
        stats_df = load_ic_stats(factor, "cs")
        if stats_df.empty:
            st.warning("该因子暂无 CS-IC 结果。")
            st.stop()
        factor_cols = sort_factor_cols(stats_df["factor_col"].unique().tolist())
        factor_col  = st.selectbox("因子窗口", factor_cols)

    with col3:
        dates = available_cs_dates(factor, ret_horizon, session)
        if not dates:
            st.warning("暂无数据。")
            st.stop()
        date_options = ["全部（跨日趋势）"] + dates
        date_sel = st.selectbox("日期", date_options)

    if date_sel == "全部（跨日趋势）":
        df = load_cs_daily_trend(factor, ret_horizon, session)
        if df.empty:
            st.warning("暂无数据。")
        else:
            st.caption(f"每个交易日日内 IC 均值，共 {len(df)} 个交易日。")
            st.plotly_chart(
                cs_daily_trend_chart(df, factor_col),
                use_container_width=True,
            )
    else:
        df = load_cs_one_day(factor, ret_horizon, session, date_sel)
        if df.empty:
            st.warning("该日期暂无数据。")
        else:
            st.caption(f"{date_sel} 日内各时刻的截面 IC，共 {len(df)} 个时间点。")
            st.plotly_chart(
                cs_intraday_chart(df, factor_col, date_sel),
                use_container_width=True,
            )
