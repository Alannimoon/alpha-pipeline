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
    load_quantile_pnl_stats, quantile_tick_chart_path,
    load_factor_meta,
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

tab_summary, tab_cs, tab_quantile, tab_meta = st.tabs(["📊 IC 汇总", "📈 截面详情", "📉 截面分层", "📖 因子说明"])


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

    # 截面分层始终使用全天数据（am/pm 是 all 的子集，跨日分半天无意义）
    _Q_SESSION = "all"

    view_mode = st.radio(
        "视图", ["单日 tick", "日频累计"], horizontal=True, key="q_view"
    )

    if view_mode == "单日 tick":
        q_dates = available_quantile_dates(q_factor, ret_horizon, _Q_SESSION)
        if not q_dates:
            st.warning("暂无分层数据，请先运行 cs_quantile。")
            st.stop()
        date_options = ["全部（跨日）"] + q_dates
        q_tick_date = st.selectbox("日期", date_options, key="q_tick_date")

        if q_tick_date == "全部（跨日）":
            img_path = quantile_tick_chart_path(q_factor, ret_horizon, _Q_SESSION, q_factor_col)
            if img_path is None:
                st.warning("暂无预渲染图片，请先运行 cs_quantile。")
            else:
                st.image(img_path, use_container_width=True)
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

    # ── PnL 统计 ──────────────────────────────────────────────────────────────
    pnl = load_quantile_pnl_stats(q_factor, ret_horizon, _Q_SESSION, q_factor_col)
    if pnl:
        st.divider()
        st.subheader("每 tick 平均收益（全周期汇总）")
        n_ticks = pnl.get("n_ticks")
        if n_ticks:
            st.caption(f"总有效 tick 数：{int(n_ticks):,}")
        cols = st.columns(7)
        labels = ["g1（最低）", "g2", "g3", "g4", "g5（最高）", "多空(g5-g1)", "单调性得分"]
        keys   = ["g1", "g2", "g3", "g4", "g5", "long_short", None]
        for col_ui, label, key in zip(cols, labels, keys):
            if key is not None:
                avg_val = pnl.get(f"avg_{key}")
                if avg_val is not None:
                    col_ui.metric(label, f"{avg_val:.4%}")
            else:
                g1, g2, g4, g5 = (pnl.get("g1"), pnl.get("g2"),
                                   pnl.get("g4"), pnl.get("g5"))
                if None not in (g1, g2, g4, g5) and (g2 - g4) != 0:
                    col_ui.metric(label, f"{(g1 - g5) / (g2 - g4):.4f}")


# ── Tab 4：因子说明 ────────────────────────────────────────────────────────────

with tab_meta:
    meta_df = load_factor_meta()
    if meta_df.empty:
        st.warning("暂无因子说明，请检查 config/factor_meta.csv。")
    else:
        search = st.text_input("搜索因子（名称 / 类别 / 关键词）", placeholder="输入关键词过滤…")
        if search:
            mask = meta_df.apply(
                lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1
            )
            filtered = meta_df[mask]
        else:
            filtered = meta_df

        if filtered.empty:
            st.info("无匹配结果。")
        else:
            col_labels = {
                "factor_name": "因子名",
                "full_name": "全称",
                "category": "类别",
                "description": "描述",
                "formula": "公式",
                "windows_min": "窗口（分钟）",
                "windows_ticks": "窗口（tick）",
                "inputs": "输入字段",
                "validity_conditions": "有效性条件",
                "notes": "备注",
            }
            for _, row in filtered.iterrows():
                st.subheader(f"{row['full_name']}（{row['factor_name']}）")
                st.caption(f"类别：{row['category']}")
                for col, label in col_labels.items():
                    if col in ("factor_name", "full_name", "category"):
                        continue
                    val = row.get(col, "")
                    if pd.notna(val) and str(val).strip():
                        st.markdown(f"**{label}**")
                        st.text(val)
                st.divider()
