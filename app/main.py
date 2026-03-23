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

from data import available_factors, load_ic_stats, load_daily_ic
from charts import ic_summary_chart, ic_timeseries_chart

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

tab_summary, tab_ts = st.tabs(["📊 IC 汇总", "📈 日度时序"])


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
            # 展示原始数据表（可折叠）
            with st.expander("查看数据表"):
                import pandas as pd
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


# ── Tab 2：日度时序 ───────────────────────────────────────────────────────────

with tab_ts:
    col1, col2 = st.columns(2)
    with col1:
        factor = st.selectbox("因子", factors)
    with col2:
        stats_df = load_ic_stats(factor, ic_type)
        if not stats_df.empty:
            factor_cols = sorted(stats_df["factor_col"].unique())
            factor_col  = st.selectbox("因子窗口", factor_cols)
        else:
            st.warning("该因子暂无 ic_stats 结果。")
            st.stop()

    daily_df = load_daily_ic(factor, ic_type, ret_horizon, session)
    if daily_df.empty:
        st.warning("暂无日度数据。")
    else:
        st.plotly_chart(
            ic_timeseries_chart(daily_df, factor_col, ic_type),
            use_container_width=True,
        )
        st.caption(f"共 {len(daily_df)} 个交易日")
