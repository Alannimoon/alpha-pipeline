"""
Plotly 图表函数。
"""

import pandas as pd
import plotly.graph_objects as go

# 每个因子一种颜色，IC 实线 / RankIC 虚线
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def ic_summary_chart(
    dfs: dict[str, pd.DataFrame],
    metric: str,
    ret_horizon: str,
    session: str,
) -> go.Figure:
    """
    IC 汇总折线图：X = 因子窗口，Y = ic_mean 或 icir。
    dfs: {factor_name: stats_df}
    metric: "ic_mean" 或 "icir"
    """
    rank_metric = "rankic_mean" if metric == "ic_mean" else "rankic_ir"
    y_label = "IC mean" if metric == "ic_mean" else "ICIR"

    fig = go.Figure()
    for i, (factor_name, df) in enumerate(dfs.items()):
        sub = df[
            (df["ret_horizon"] == ret_horizon) & (df["session"] == session)
        ].sort_values("factor_window")
        color = _COLORS[i % len(_COLORS)]

        fig.add_trace(go.Scatter(
            x=sub["factor_window"], y=sub[metric],
            name=f"{factor_name} IC",
            mode="lines+markers",
            line=dict(color=color, dash="solid"),
        ))
        fig.add_trace(go.Scatter(
            x=sub["factor_window"], y=sub[rank_metric],
            name=f"{factor_name} RankIC",
            mode="lines+markers",
            line=dict(color=color, dash="dash"),
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="Factor window (min)",
        yaxis_title=y_label,
        height=420,
        margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def ts_daily_chart(daily_df: pd.DataFrame, factor_col: str) -> go.Figure:
    """
    TS-IC 跨日趋势图：X = 日期，Y = 当日所有股票的 TS-IC 均值。
    """
    ic_col  = f"ts_ic_{factor_col}"
    ric_col = f"ts_rankic_{factor_col}"

    fig = go.Figure()
    if ic_col in daily_df.columns:
        fig.add_trace(go.Scatter(
            x=daily_df["Date"], y=daily_df[ic_col],
            name="IC", line=dict(color="#1f77b4"),
        ))
    if ric_col in daily_df.columns:
        fig.add_trace(go.Scatter(
            x=daily_df["Date"], y=daily_df[ric_col],
            name="RankIC", line=dict(color="#ff7f0e", dash="dash"),
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="TS-IC（跨股票均值）",
        height=420,
        margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def cs_intraday_chart(intraday_df: pd.DataFrame, factor_col: str) -> go.Figure:
    """
    CS-IC 日内模式图：X = SampleTime，Y = 所有日期在该时刻的 IC 均值。
    展示因子在一天内不同时段的截面预测力。
    """
    ic_col  = f"ic_{factor_col}"
    ric_col = f"rankic_{factor_col}"

    fig = go.Figure()
    if ic_col in intraday_df.columns:
        fig.add_trace(go.Scatter(
            x=intraday_df["SampleTime"], y=intraday_df[ic_col],
            name="IC", line=dict(color="#1f77b4"),
        ))
    if ric_col in intraday_df.columns:
        fig.add_trace(go.Scatter(
            x=intraday_df["SampleTime"], y=intraday_df[ric_col],
            name="RankIC", line=dict(color="#ff7f0e", dash="dash"),
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="SampleTime",
        yaxis_title="CS-IC（跨日均值）",
        height=420,
        margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(tickangle=45, nticks=20),
    )
    return fig
