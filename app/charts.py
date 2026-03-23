"""
Plotly 图表函数。
"""

import pandas as pd
import plotly.graph_objects as go

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
    """
    rank_metric = "rankic_mean" if metric == "ic_mean" else "rankic_ir"
    y_label     = "IC mean"    if metric == "ic_mean" else "ICIR"

    fig = go.Figure()
    for i, (factor_name, df) in enumerate(dfs.items()):
        sub = df[
            (df["ret_horizon"] == ret_horizon) & (df["session"] == session)
        ].sort_values("factor_window")
        color = _COLORS[i % len(_COLORS)]

        fig.add_trace(go.Scatter(
            x=sub["factor_window"], y=sub[metric],
            name=f"{factor_name} IC",
            mode="lines+markers", line=dict(color=color, dash="solid"),
        ))
        fig.add_trace(go.Scatter(
            x=sub["factor_window"], y=sub[rank_metric],
            name=f"{factor_name} RankIC",
            mode="lines+markers", line=dict(color=color, dash="dash"),
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="Factor window (min)",
        yaxis_title=y_label,
        height=420, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


_GROUP_COLORS = ["#d62728", "#ff7f0e", "#8c8c8c", "#2ca02c", "#1f77b4"]
# g1(最低)=红, g2=橙, g3=灰, g4=绿, g5(最高)=蓝


def quantile_bar_chart(summary_df: pd.DataFrame) -> go.Figure:
    """
    截面分层柱状图：5个柱，X=组别，Y=跨所有日期和时刻的前向收益均值。
    g1=最低因子值组，g5=最高因子值组。
    """
    fig = go.Figure()
    for i, row in summary_df.iterrows():
        g = int(row["group"])
        fig.add_trace(go.Bar(
            x=[f"g{g}"],
            y=[row["mean_ret"]],
            name=f"g{g}",
            marker_color=_GROUP_COLORS[g - 1],
            showlegend=False,
        ))

    ls = summary_df["long_short"].iloc[0]
    fig.add_annotation(
        text=f"多空价差 g5-g1: {ls:.4%}",
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, align="right",
        font=dict(size=12),
    )
    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="分组（因子值由低到高）",
        yaxis_title="前向收益均值",
        yaxis_tickformat=".4%",
        height=480, margin=dict(t=30),
    )
    return fig


def quantile_daily_chart(daily_df: pd.DataFrame) -> go.Figure:
    """
    截面分层跨日趋势：5条组线 + 多空价差线，X=日期。
    """
    fig = go.Figure()
    for g in range(1, 6):
        col = f"g{g}"
        if col in daily_df.columns:
            fig.add_trace(go.Scatter(
                x=daily_df["Date"], y=daily_df[col],
                name=f"g{g}", line=dict(color=_GROUP_COLORS[g - 1]),
                opacity=0.6,
            ))
    if "long_short" in daily_df.columns:
        fig.add_trace(go.Scatter(
            x=daily_df["Date"], y=daily_df["long_short"],
            name="多空(g5-g1)", line=dict(color="black", width=2),
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="日度组均收益",
        yaxis_tickformat=".4%",
        height=560, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def quantile_intraday_chart(intraday_df: pd.DataFrame, day: str) -> go.Figure:
    """
    截面分层单日日内曲线：5条组线，X=SampleTime。
    """
    fig = go.Figure()
    for g in range(1, 6):
        col = f"g{g}"
        if col in intraday_df.columns:
            fig.add_trace(go.Scatter(
                x=intraday_df["SampleTime"], y=intraday_df[col],
                name=f"g{g}", line=dict(color=_GROUP_COLORS[g - 1]),
            ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="SampleTime",
        yaxis_title=f"组均收益  {day}",
        yaxis_tickformat=".4%",
        height=560, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(tickangle=45, nticks=20),
    )
    return fig


def cs_daily_trend_chart(df: pd.DataFrame, factor_col: str) -> go.Figure:
    """
    CS-IC 跨日趋势：X = 日期，Y = 当日所有时间点的 IC 均值。
    """
    ic_col  = f"ic_{factor_col}"
    ric_col = f"rankic_{factor_col}"

    fig = go.Figure()
    if ic_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[ic_col],
            name="IC", line=dict(color="#1f77b4"),
        ))
    if ric_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[ric_col],
            name="RankIC", line=dict(color="#ff7f0e", dash="dash"),
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="CS-IC（日内均值）",
        height=420, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def cs_intraday_chart(df: pd.DataFrame, factor_col: str, day: str) -> go.Figure:
    """
    CS-IC 单日日内曲线：X = SampleTime，Y = 各时刻的截面 IC。
    """
    ic_col  = f"ic_{factor_col}"
    ric_col = f"rankic_{factor_col}"

    fig = go.Figure()
    if ic_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["SampleTime"], y=df[ic_col],
            name="IC", line=dict(color="#1f77b4"),
        ))
    if ric_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["SampleTime"], y=df[ric_col],
            name="RankIC", line=dict(color="#ff7f0e", dash="dash"),
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    fig.update_layout(
        xaxis_title="SampleTime",
        yaxis_title=f"CS-IC  {day}",
        height=420, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(tickangle=45, nticks=20),
    )
    return fig
