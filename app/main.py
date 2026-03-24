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
    available_factors, available_cs_dates, available_quantile_dates,
    load_ic_stats, sort_factor_cols,
    load_cs_daily_trend, load_cs_one_day,
    load_quantile_tick_cum, load_quantile_tick_one_day, load_quantile_daily_cum,
)
from charts import (
    ic_summary_chart, cs_daily_trend_chart, cs_intraday_chart,
    quantile_tick_cum_chart, quantile_intraday_cum_chart, quantile_daily_cum_chart,
)

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

tab_summary, tab_cs, tab_quantile = st.tabs(["📊 IC 汇总", "📈 截面详情", "📉 截面分层"])


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


# ── Tab 3：截面分层 ───────────────────────────────────────────────────────────

with tab_quantile:
    col1, col2 = st.columns(2)

    with col1:
        q_factor = st.selectbox("因子", factors, key="q_factor")
    with col2:
        q_stats = load_ic_stats(q_factor, "cs")
        if q_stats.empty:
            st.warning("该因子暂无 CS-IC 结果。")
            st.stop()
        q_factor_cols = sort_factor_cols(q_stats["factor_col"].unique().tolist())
        q_factor_col  = st.selectbox("因子窗口", q_factor_cols, key="q_factor_col")

    view_mode = st.radio(
        "视图", ["tick 级别累计", "日频累计"], horizontal=True, key="q_view"
    )

    # 截面分层始终使用全天数据（am/pm 是 all 的子集，跨日分半天无意义）
    _Q_SESSION = "all"

    if view_mode == "tick 级别累计":
        q_dates = available_quantile_dates(q_factor, ret_horizon, _Q_SESSION)
        if not q_dates:
            st.warning("暂无分层数据，请先运行 cs_quantile。")
            st.stop()
        date_options = ["全部（跨日）"] + q_dates
        q_tick_date = st.selectbox("日期", date_options, key="q_tick_date")

        if q_tick_date == "全部（跨日）":
            tick_df = load_quantile_tick_cum(q_factor, ret_horizon, _Q_SESSION, q_factor_col)
            if tick_df.empty:
                st.warning("暂无数据。")
            else:
                n_days  = tick_df["Date"].nunique()
                n_ticks = len(tick_df)
                st.caption(f"共 {n_days} 个交易日，{n_ticks} 个 tick。")
                st.plotly_chart(quantile_tick_cum_chart(tick_df), use_container_width=True)
        else:
            intraday_df = load_quantile_tick_one_day(
                q_factor, ret_horizon, _Q_SESSION, q_tick_date, q_factor_col
            )
            if intraday_df.empty:
                st.warning("该日期暂无数据。")
            else:
                st.caption(f"{q_tick_date} 日内累计，{len(intraday_df)} 个 tick（上午+下午）。")
                st.plotly_chart(
                    quantile_intraday_cum_chart(intraday_df, q_tick_date),
                    use_container_width=True,
                )
    else:
        daily_df = load_quantile_daily_cum(q_factor, ret_horizon, _Q_SESSION, q_factor_col)
        if daily_df.empty:
            st.warning("暂无分层数据，请先运行 cs_quantile。")
        else:
            st.caption(f"共 {len(daily_df)} 个交易日，跨日累计收益。")
            st.plotly_chart(quantile_daily_cum_chart(daily_df), use_container_width=True)
