"""Plotly chart builders using only actual controller and model records."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import CONTROLLER_COLOURS


def dark_layout(figure: go.Figure, *, height: int = 360) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        margin={"l": 20, "r": 20, "t": 45, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return figure


def scenario_heatmap(summary: pd.DataFrame, metric: str, held_out: set[str]) -> go.Figure:
    values = summary.pivot(
        index="configuration", columns="load_percent", values=metric
    ).sort_index()
    figure = go.Figure(
        go.Heatmap(
            z=values.to_numpy(),
            x=values.columns,
            y=values.index,
            colorscale="Blues",
            colorbar_title=metric,
        )
    )
    held_out_x: list[float] = []
    held_out_y: list[str] = []
    for configuration in values.index:
        for load in values.columns:
            matching = summary.loc[
                (summary["configuration"] == configuration) & (summary["load_percent"] == load),
                "scenario",
            ]
            if not matching.empty and matching.iloc[0] in held_out:
                held_out_x.append(load)
                held_out_y.append(configuration)
    if held_out_x:
        figure.add_trace(
            go.Scatter(
                x=held_out_x,
                y=held_out_y,
                mode="markers",
                name="Held out",
                marker={
                    "symbol": "square-open",
                    "size": 34,
                    "color": "#fbbf24",
                    "line": {"width": 3},
                },
            )
        )
    figure.update_xaxes(title="Load (%)")
    figure.update_yaxes(title="Configuration")
    return dark_layout(figure)


def replay_chart(trajectory: pd.DataFrame, outlet_limit: float) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=trajectory["step"],
            y=trajectory["outlet_temperature"],
            name="Outlet temperature",
            line={"color": "#fb7185"},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=trajectory["step"],
            y=trajectory["proposed_supply"],
            name="Supply setting",
            line={"color": "#38bdf8"},
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=trajectory["step"],
            y=trajectory["action"],
            name="Action",
            mode="markers",
            marker={"color": "#f59e0b", "size": 7},
        ),
        secondary_y=True,
    )
    figure.add_hline(
        y=outlet_limit, line_dash="dash", line_color="#fbbf24", annotation_text="Outlet limit"
    )
    figure.update_xaxes(title="Step")
    figure.update_yaxes(title="Outlet temperature (°C)", secondary_y=False)
    figure.update_yaxes(title="Supply setting (°C) / action", secondary_y=True)
    return dark_layout(figure, height=390)


def controller_scatter(summary: pd.DataFrame) -> go.Figure:
    grouped = summary.groupby("controller", as_index=False).mean(numeric_only=True)
    figure = px.scatter_3d(
        grouped,
        x="mean_p_dc",
        y="violation_rate",
        z="switching_rate",
        color="controller",
        color_discrete_map=CONTROLLER_COLOURS,
        hover_name="controller",
    )
    figure.update_layout(
        scene={
            "xaxis_title": "Mean P_DC",
            "yaxis_title": "Violation rate",
            "zaxis_title": "Switching rate",
        }
    )
    return dark_layout(figure, height=430)


def learning_curve_chart(curves: pd.DataFrame) -> go.Figure:
    figure = px.line(
        curves, x="step", y="reward", color="controller", color_discrete_map=CONTROLLER_COLOURS
    )
    figure.update_layout(yaxis_title="Training reward", xaxis_title="Training step")
    return dark_layout(figure)


def action_distribution_chart(trajectories: pd.DataFrame) -> go.Figure:
    labels = {0: "Decrease cooling", 1: "Maintain", 2: "Increase cooling"}
    counts = trajectories.groupby(["controller", "action"], as_index=False).size()
    counts["action_label"] = counts["action"].map(labels)
    figure = px.bar(
        counts,
        x="controller",
        y="size",
        color="action_label",
        barmode="stack",
        color_discrete_map={
            "Decrease cooling": "#fb7185",
            "Maintain": "#38bdf8",
            "Increase cooling": "#34d399",
        },
    )
    figure.update_layout(yaxis_title="Action count", xaxis_title="")
    return dark_layout(figure)


def paired_savings_chart(metrics: pd.DataFrame) -> go.Figure:
    baseline = metrics.loc[
        metrics["controller"] == "Rule-based", ["scenario", "episode", "mean_p_dc"]
    ].rename(columns={"mean_p_dc": "baseline_p_dc"})
    comparison = metrics.merge(baseline, on=["scenario", "episode"], how="inner")
    comparison = comparison.loc[comparison["controller"] != "Rule-based"].copy()
    comparison["saving"] = comparison["baseline_p_dc"] - comparison["mean_p_dc"]
    grouped = (
        comparison.groupby("controller", as_index=False)["saving"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["ci"] = 1.96 * grouped["std"].fillna(0) / np.sqrt(grouped["count"].clip(lower=1))
    figure = go.Figure(
        go.Bar(
            x=grouped["controller"],
            y=grouped["mean"],
            error_y={"type": "data", "array": grouped["ci"]},
            marker_color=[
                CONTROLLER_COLOURS.get(value, "#38bdf8") for value in grouped["controller"]
            ],
        )
    )
    figure.add_hline(y=0, line_color="#94a3b8")
    figure.update_layout(yaxis_title="Paired P_DC saving vs rule-based")
    return dark_layout(figure)
