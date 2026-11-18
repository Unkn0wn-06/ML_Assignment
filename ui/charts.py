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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 45, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return figure


def scenario_heatmap(summary: pd.DataFrame, metric: str, held_out: set[str]) -> go.Figure:
    loads = [25, 50, 75, 100]
    configs = ["Baseline", "Plate", "PlateWall", "Wall"]
    values = summary.pivot_table(
        index="configuration", columns="load_percent", values=metric
    ).reindex(index=configs, columns=loads)
    labels = [
        [
            (
                "n/a"
                if pd.isna(value)
                else format(value, ".1%" if metric == "violation_rate" else ".2f")
            )
            for value in row
        ]
        for row in values.to_numpy()
    ]
    figure = go.Figure(
        go.Heatmap(
            z=values.to_numpy(),
            x=list(range(4)),
            y=list(range(4)),
            colorscale=[[0, "#26334b"], [0.01, "#156078"], [0.4, "#32a7a9"], [1, "#dfc45a"]],
            showscale=False,
        )
    )
    for row, config in enumerate(configs):
        for col, load in enumerate(loads):
            figure.add_annotation(
                x=col, y=row, text=labels[row][col], showarrow=False, bgcolor="#24304f"
            )
            if f"{config}Test_{load}percent" in held_out:
                figure.add_shape(
                    type="rect",
                    x0=col - 0.49,
                    x1=col + 0.49,
                    y0=row - 0.49,
                    y1=row + 0.49,
                    line_color="#38bdf8",
                )
    figure.update_xaxes(tickvals=list(range(4)), ticktext=[f"{load}%" for load in loads])
    figure.update_yaxes(tickvals=list(range(4)), ticktext=configs, autorange="reversed")
    return dark_layout(figure)


def replay_chart(trajectory, outlet_limit):
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{}], [{"secondary_y": True}]])
    for field, label, row, secondary, colour in [
        ("outlet_temperature", "Outlet", 1, False, "#fb7185"),
        ("proposed_supply", "Supply", 2, False, "#38bdf8"),
        ("action", "Action", 2, True, "#f59e0b"),
    ]:
        figure.add_scatter(
            x=trajectory.step,
            y=trajectory[field],
            name=label,
            mode="lines+markers",
            line_color=colour,
            line_shape="hv" if secondary else "linear",
            row=row,
            col=1,
            secondary_y=secondary,
        )
    figure.add_hrect(
        y0=outlet_limit - 1,
        y1=outlet_limit,
        fillcolor="#f59e0b",
        opacity=0.15,
        line_width=0,
        row=1,
        col=1,
    )
    figure.add_hline(y=outlet_limit, line_dash="dash", row=1, col=1)
    figure.update_yaxes(title="Outlet °C", row=1, col=1)
    figure.update_yaxes(title="Supply °C", row=2, col=1, secondary_y=False)
    figure.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=["Decrease", "Maintain", "Increase"],
        range=[-0.2, 2.2],
        row=2,
        col=1,
        secondary_y=True,
    )
    figure.update_xaxes(title="Step", row=2, col=1)
    return dark_layout(figure)


def controller_scatter(summary: pd.DataFrame) -> go.Figure:
    grouped = summary.groupby("controller", as_index=False).mean(numeric_only=True)
    figure = px.scatter_3d(
        grouped,
        x="mean_p_dc",
        y="violation_rate",
        z="switching_rate",
        color="controller",
        color_discrete_map=CONTROLLER_COLOURS,
        text="controller",
        hover_name="controller",
    )
    figure.update_scenes(camera_eye={"x":1.6,"y":1.5,"z":1.3},xaxis_showgrid=False,yaxis_showgrid=False,zaxis_showgrid=False,yaxis_tickformat=".0%",zaxis_tickformat=".0%")
    figure.update_layout(
        scene={
            "xaxis_title": "P_DC (kW)",
            "yaxis_title": "Violation rate",
            "zaxis_title": "Switching rate",
        }
    )
    return dark_layout(figure, height=430)


def learning_curve_chart(curves: pd.DataFrame) -> go.Figure:
    figure = px.line(
        curves, x="step", y="reward", line_group="seed" if "seed" in curves else None, color="controller", color_discrete_map=CONTROLLER_COLOURS
    )
    figure.update_traces(opacity=0.22)
    for controller, sample in curves.groupby("controller"):
        paths = (
            sample.pivot_table(index="step", columns="seed", values="reward")
            if "seed" in sample
            else sample.set_index("step")[["reward"]]
        )
        smooth = paths.rolling(10, min_periods=1).mean()
        mean = smooth.mean(axis=1)
        colour = CONTROLLER_COLOURS[controller]
        if smooth.shape[1] > 1:
            spread = smooth.std(axis=1)
            figure.add_scatter(x=mean.index, y=mean - spread, line_width=0, showlegend=False)
            figure.add_scatter(
                x=mean.index,
                y=mean + spread,
                fill="tonexty",
                line_color=colour,
                opacity=0.25,
                showlegend=False,
            )
        figure.add_scatter(x=mean.index, y=mean, line_color=colour, name=controller + " · rolling")
    figure.update_layout(yaxis_title="Training reward", xaxis_title="Training step")
    return dark_layout(figure)


def action_distribution_chart(trajectories: pd.DataFrame) -> go.Figure:
    labels = {0: "Decrease cooling", 1: "Maintain", 2: "Increase cooling"}
    counts = trajectories.groupby(["controller", "action"], as_index=False).size()
    counts["size"] = 100 * counts["size"] / counts.groupby("controller")["size"].transform("sum")
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
    figure.update_layout(yaxis_title="Action share (%)", xaxis_title="")
    return dark_layout(figure)


def paired_savings_chart(trajectories: pd.DataFrame) -> go.Figure:
    keys = ["seed","scenario","episode","step"]
    baseline = trajectories.loc[trajectories.controller=="Rule-based",[*keys,"P_DC"]]
    paired = trajectories.loc[trajectories.controller!="Rule-based"].merge(baseline,on=keys,suffixes=("","_baseline"),validate="many_to_one")
    paired["saving"] = (paired.P_DC_baseline - paired.P_DC) * 2 / 60
    comparison = paired.groupby(["controller","seed","scenario","episode"],as_index=False).saving.sum()
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
    figure.update_layout(yaxis_title="Paired energy saving (kWh), common steps")
    return dark_layout(figure)
