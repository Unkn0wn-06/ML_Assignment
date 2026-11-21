"""Plotly chart builders using only actual controller and model records."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import CONTROLLER_COLOURS, TEMPERATURE_RAMP


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


def scenario_heatmap(
    summary: pd.DataFrame, metric: str, held_out: set[str], limit: float = 29.0
) -> go.Figure:
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
                else format(value, ".1%" if metric == "episode_violation_rate" else ".2f")
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
            colorscale=[
                [point, color]
                for i, color in enumerate(TEMPERATURE_RAMP)
                for point in (i / 7, (i + 1) / 7)
            ],
            zmin=limit - 6 if metric == "mean_outlet_temperature" else 0,
            zmax=limit + 1 if metric == "mean_outlet_temperature" else None,
            showscale=False,
            customdata=summary.pivot_table(
                index="configuration", columns="load_percent", values="mean_steps_survived"
            ).reindex(index=configs, columns=loads)
            if "mean_steps_survived" in summary
            else None,
            hovertemplate="Value %{z:.3g}<br>Mean steps survived %{customdata:.1f}<extra></extra>",
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


def replay_chart(trajectory, outlet_limit, comparison=None):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    for field, label, row, secondary, colour in [
        ("outlet_temperature", "Outlet", 1, False, "#fb7185"),
        ("proposed_supply", "Supply", 2, False, "#38bdf8"),
        ("action", "Action", 3, True, "#f59e0b"),
    ]:
        figure.add_scatter(
            x=trajectory.step,
            y=trajectory[field],
            name=label,
            mode="lines+markers",
            line_color=CONTROLLER_COLOURS.get(trajectory.controller.iloc[0], colour)
            if "controller" in trajectory
            else colour,
            line_shape="hv" if secondary else "linear",
            row=row,
            col=1,
        )
    if comparison is not None:
        for field, label, row, secondary in [
            ("outlet_temperature", "Outlet", 1, False),
            ("proposed_supply", "Supply", 2, False),
            ("action", "Action", 3, True),
        ]:
            name = comparison.controller.iloc[0]
            figure.add_scatter(
                x=comparison.step,
                y=comparison[field],
                name=f"{name} {label}",
                line_color=CONTROLLER_COLOURS[name],
                line_dash="dot",
                line_shape="hv" if secondary else "linear",
                row=row,
                col=1,
            )
    if "controller" in trajectory and (
        trajectory.controller.str.startswith("Rule-based").any()
        or (comparison is not None and comparison.controller.str.startswith("Rule-based").any())
    ):
        for threshold in [min(24, outlet_limit - 3), outlet_limit - 1]:
            figure.add_hline(y=threshold, line_dash="dot", opacity=0.5, row=1, col=1)
    if "horizon" in trajectory:
        last = trajectory.iloc[-1]
        figure.update_xaxes(range=[0, last.horizon])
        note = "Reached horizon" if last.step == last.horizon else "Reached end of scenario data"
        if last.get("terminated", False):
            note = f"Ended at step {last.step}: outlet {last.outlet_temperature:.2f} °C > limit {outlet_limit:.2f} °C"
        figure.add_vline(x=last.step, line_dash="dot", row=1, col=1)
        figure.add_annotation(x=0.5, y=0.68, xref="paper", yref="paper", text=note, showarrow=False)
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
    figure.update_yaxes(title="Supply °C", row=2, col=1)
    figure.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=["Decrease", "Maintain", "Increase"],
        range=[-0.2, 2.2],
        row=3,
        col=1,
    )
    figure.update_xaxes(title="Step", row=3, col=1)
    for rows in [trajectory, comparison]:
        if rows is None or "supply_clipped" not in rows:
            continue
        clipped = rows[rows.supply_clipped]
        if len(clipped):
            figure.add_scatter(
                x=clipped.step,
                y=clipped.proposed_supply,
                mode="markers",
                marker_symbol="x",
                name=f"{rows.controller.iloc[0]} clipped",
                row=2,
                col=1,
            )
    for bound in ["supply_lower", "supply_upper"]:
        if bound in trajectory:
            figure.add_hline(
                y=trajectory[bound].iloc[0], line_dash="dot", opacity=0.55, row=2, col=1
            )
    return dark_layout(figure, height=530)


def learning_curve_chart(curves: pd.DataFrame) -> go.Figure:
    figure = px.line(
        curves,
        x="step",
        y="reward",
        line_group="seed" if "seed" in curves else None,
        color="controller",
        color_discrete_map=CONTROLLER_COLOURS,
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


def paired_savings(metrics):
    keys = ["scenario", "start_row", "seed", "episode", "horizon"]
    full = metrics.loc[metrics.steps_survived == metrics.horizon]
    base = full.loc[full.controller == "Rule-based"]
    rows = []
    for controller in metrics.controller.unique():
        if controller == "Rule-based":
            continue
        pairs = full.loc[full.controller == controller].merge(
            base, on=keys, suffixes=("", "_base"), validate="one_to_one"
        )
        saved = (pairs.mean_p_dc_base - pairs.mean_p_dc).mean()
        rows.append(
            {
                "Controller": controller,
                "Pairs": len(pairs),
                "Saving (kW)": saved,
                "Saving (%)": 100 * saved / pairs.mean_p_dc_base.mean(),
                "Any violation difference (pp)": 100
                * (pairs.any_violation.mean() - pairs.any_violation_base.mean()),
            }
        )
    return pd.DataFrame(rows)


def training_chart(curves, metric):
    figure = go.Figure()
    for name, group in curves.groupby("controller"):
        color = CONTROLLER_COLOURS[name]
        values = group.pivot(index="step", columns="seed", values=metric).dropna(how="all")
        if metric == "reward_vs_baseline":
            for seed in values:
                figure.add_scatter(
                    x=values.index,
                    y=values[seed],
                    mode="lines",
                    line_color=color,
                    opacity=0.12,
                    legendgroup=name,
                    showlegend=False,
                )
            values = values.rolling(40, min_periods=1).mean()
        mean = values.mean(axis=1)
        if metric.startswith("eval_") and values.shape[1] > 1:
            spread = values.std(axis=1)
            figure.add_scatter(
                x=mean.index, y=mean - spread, line_width=0, showlegend=False, legendgroup=name
            )
            figure.add_scatter(
                x=mean.index,
                y=mean + spread,
                line_width=0,
                fill="tonexty",
                fillcolor=color,
                opacity=0.17,
                showlegend=False,
                legendgroup=name,
            )
        figure.add_scatter(
            x=mean.index,
            y=mean,
            name=name,
            legendgroup=name,
            line_color=color,
            mode="lines+markers" if metric.startswith("eval_") else "lines",
        )
        if metric == "epsilon":
            fallback = group.groupby("step").eval_fallback.mean().dropna()
            figure.add_scatter(
                x=fallback.index,
                y=fallback,
                name=name + " · fallback",
                line_dash="dot",
                line_color=color,
                mode="lines+markers",
            )
    if metric == "reward_vs_baseline":
        figure.add_hline(y=0, line_dash="dash", annotation_text="Fixed setpoint")
    figure.update_xaxes(title="Training episode")
    return dark_layout(figure, height=260 if metric == "epsilon" else 350)


def final_difference(curves):
    from scipy.stats import t

    notes = []
    for name, group in curves.groupby("controller"):
        final = (
            group.sort_values("step")
            .groupby("seed")
            .tail(40)
            .groupby("seed")
            .reward_vs_baseline.mean()
        )
        mean = final.mean()
        if len(final) < 2:
            notes.append(
                f"{name}: final mean {mean:+.3f}; one seed cannot estimate a between-seed interval."
            )
            continue
        half = t.ppf(0.975, len(final) - 1) * final.sem()
        finding = (
            "No measurable improvement over fixed setpoint"
            if mean - half <= 0 <= mean + half
            else (
                "Higher reward than fixed setpoint"
                if mean > 0
                else "Lower reward than fixed setpoint"
            )
        )
        notes.append(
            f"{name}: {finding} (last 40 episodes, mean {mean:+.3f}, 95% CI {mean - half:+.3f} to {mean + half:+.3f})."
        )
    return " ".join(notes)
