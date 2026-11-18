"""Control-room dashboard for honest AAU cooling-controller experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_centre_rl.data import STATE_COLUMNS, load_csv_directory
from data_centre_rl.experiments import (
    CONTROLLERS,
    ControllerResult,
    ExperimentContext,
    diagnostics_frame,
    prepare_experiment,
    run_controller,
)
from ui.charts import (
    action_distribution_chart,
    controller_scatter,
    dark_layout,
    learning_curve_chart,
    paired_savings_chart,
    replay_chart,
    scenario_heatmap,
)
from ui.room3d import render_room
from ui.theme import apply_control_room_theme

st.set_page_config(page_title="Cooling Control Room", page_icon="❄️", layout="wide")
apply_control_room_theme(st)


@st.cache_data(show_spinner=False)
def load_data(data_directory: str) -> pd.DataFrame:
    return load_csv_directory(data_directory)


def enabled_controllers() -> tuple[list[str], list[str]]:
    if importlib.util.find_spec("stable_baselines3"):
        return list(CONTROLLERS), []
    return list(CONTROLLERS[:4]), ["DQN", "PPO"]


def summarise(results: dict[str, ControllerResult], data: pd.DataFrame) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    metrics = pd.concat([result.scenario_metrics for result in results.values()], ignore_index=True)
    metadata = data.groupby("scenario", as_index=False).agg(
        configuration=("configuration", "first"), load_percent=("load_fraction", "first")
    )
    metadata["load_percent"] *= 100
    summary = metrics.merge(metadata, on="scenario", how="left")
    summary["violation_rate"] = summary["violation_count"] / summary["steps"].clip(lower=1)
    return summary


def delta(summary: pd.DataFrame, controller: str, column: str) -> str | None:
    baseline = summary.loc[summary["controller"] == "Rule-based", column]
    current = summary.loc[summary["controller"] == controller, column]
    if baseline.empty or current.empty or controller == "Rule-based":
        return None
    return f"{current.mean() - baseline.mean():+.3f} vs rule-based"


def safety_status(maximum: float, limit: float) -> tuple[str, str]:
    if maximum <= limit - 1:
        return "GREEN", "safety-green"
    if maximum <= limit:
        return "AMBER", "safety-amber"
    return "RED", "safety-red"


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby("controller", as_index=False).agg(
        mean_p_dc=("mean_p_dc", "mean"),
        std_p_dc=("mean_p_dc", "std"),
        mean_pue=("mean_pue", "mean"),
        mean_cop=("mean_cop", "mean"),
        max_outlet_temperature=("max_outlet_temperature", "max"),
        violation_rate=("violation_rate", "mean"),
        max_violation=("max_violation", "max"),
        switching_rate=("switching_rate", "mean"),
        reward=("total_reward", "mean"),
    )
    grouped["P_DC (mean ± std)"] = grouped.apply(
        lambda row: f"{row.mean_p_dc:.3f} ± {0 if pd.isna(row.std_p_dc) else row.std_p_dc:.3f}",
        axis=1,
    )
    return grouped


def pue_surface(context: ExperimentContext, configuration: str, outlet: float) -> go.Figure:
    supply_low, supply_high = context.supply_bounds
    loads = sorted(context.training_data["load_fraction"].unique())
    supply = [supply_low + index * (supply_high - supply_low) / 20 for index in range(21)]
    state = (
        context.training_data.loc[
            context.training_data["configuration"] == configuration, STATE_COLUMNS
        ]
        .mean()
        .to_dict()
    )
    grid = pd.DataFrame(
        [
            {
                **state,
                "Outlet Avg Temp": outlet,
                "Cooling Supply Average": setting,
                "configuration": configuration,
                "load_fraction": load,
            }
            for load in loads
            for setting in supply
        ]
    )
    grid["PUE"] = context.model.predict(grid)["PUE"]
    surface = grid.pivot(index="Cooling Supply Average", columns="load_fraction", values="PUE")
    observed = context.training_data.loc[context.training_data["configuration"] == configuration]
    figure = go.Figure(
        go.Surface(
            z=surface.to_numpy(),
            x=surface.columns * 100,
            y=surface.index,
            colorscale="Blues",
            opacity=0.85,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=observed["load_fraction"] * 100,
            y=observed["Cooling Supply Average"],
            z=observed["PUE"],
            mode="markers",
            name="Observed AAU",
            marker={"size": 3, "color": "#fbbf24"},
        )
    )
    figure.update_layout(
        scene={
            "xaxis_title": "Load (%)",
            "yaxis_title": "Supply setting (°C)",
            "zaxis_title": "Predicted PUE",
        }
    )
    return dark_layout(figure, height=460)


st.markdown('<div class="control-kicker">AAU · DATA-DRIVEN SIMULATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="control-title">Cooling control room</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="safety-banner"><strong>Simulation only.</strong> AAU files contain no logged control actions; cooling supply temperature is a proxy. Never use this dashboard to control live cooling equipment.</div>',
    unsafe_allow_html=True,
)

available, disabled = enabled_controllers()
with st.sidebar:
    st.header("Experiment")
    data_directory = st.text_input("AAU data folder", "data/raw/AAU")
    controllers = st.multiselect("Controllers", available, default=["Rule-based", "Q-Learning"])
    if disabled:
        st.info('DQN and PPO require `python -m pip install -e ".[deep-rl]"`.')
    split_name = st.selectbox(
        "Evaluation split", ["All scenarios", "Load generalisation: hold out 100%"]
    )
    outlet_limit = st.slider("Outlet limit (°C)", 20.0, 35.0, 27.0, 0.5)
    horizon = st.slider("Episode horizon", 5, 200, 50, 5)
    repetitions = st.slider("Episodes per scenario", 1, 10, 1)
    tabular_episodes = st.slider("Tabular RL episodes", 25, 1_000, 250, 25)
    deep_timesteps = st.slider("Deep RL timesteps", 1_000, 100_000, 20_000, 1_000)
    trees = st.slider("Transition-model trees", 20, 200, 50, 10)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    run = st.button(
        "Run experiment", type="primary", use_container_width=True, disabled=not controllers
    )

data: pd.DataFrame | None = None
data_error: str | None = None
try:
    if Path(data_directory).is_dir():
        data = load_data(data_directory)
    else:
        data_error = "AAU data folder not found. Run `python scripts/prepare_data.py --archive ..\\sources\\AAU.tar.xz` first."
except (FileNotFoundError, ValueError) as error:
    data_error = str(error)

if run and data is not None:
    progress = st.sidebar.progress(0, text="Preparing split and transition model...")
    try:
        with st.spinner("Running the learned simulator..."):
            context = prepare_experiment(data, split_name=split_name, trees=trees, seed=int(seed))
            results: dict[str, ControllerResult] = {}
            for index, controller in enumerate(controllers):

                def report(
                    current: int,
                    total: int,
                    reward: float,
                    *,
                    offset=index,
                    controller_name=controller,
                ) -> None:
                    fraction = (offset + current / max(total, 1)) / len(controllers)
                    progress.progress(
                        min(fraction, 1.0), text=f"Training {controller_name}: {current}/{total}"
                    )

                results[controller] = run_controller(
                    context,
                    controller=controller,
                    outlet_limit=outlet_limit,
                    horizon=horizon,
                    episodes_per_scenario=repetitions,
                    tabular_episodes=tabular_episodes,
                    deep_timesteps=deep_timesteps,
                    seed=int(seed),
                    progress_callback=report,
                )
            st.session_state.update(
                context=context, controller_results=results, outlet_limit=outlet_limit
            )
    except (ImportError, RuntimeError, ValueError) as error:
        st.error(str(error))
    finally:
        progress.empty()
elif run:
    st.error(data_error or "Choose a valid AAU data folder.")

overview_tab, compare_tab, diagnostics_tab, source_tab = st.tabs(
    ["Overview", "Compare controllers", "Model diagnostics", "Source data"]
)
context: ExperimentContext | None = st.session_state.get("context")
results: dict[str, ControllerResult] = st.session_state.get("controller_results", {})
summary = summarise(results, data) if data is not None and results else pd.DataFrame()

with overview_tab:
    if summary.empty:
        st.info(
            "Run one or more controllers from the Experiment panel to populate this control-room overview."
        )
    else:
        controller = st.selectbox("Overview controller", list(results))
        selected = summary.loc[summary["controller"] == controller]
        status, status_class = safety_status(
            selected["max_outlet_temperature"].max(), st.session_state["outlet_limit"]
        )
        columns = st.columns(5)
        columns[0].metric(
            "Mean P_DC",
            f"{selected['mean_p_dc'].mean():.3f}",
            delta(summary, controller, "mean_p_dc"),
        )
        columns[1].metric(
            "Mean PUE", f"{selected['mean_pue'].mean():.3f}", delta(summary, controller, "mean_pue")
        )
        columns[2].metric(
            "Mean outlet",
            f"{selected['mean_outlet_temperature'].mean():.2f} °C",
            delta(summary, controller, "mean_outlet_temperature"),
        )
        columns[3].metric(
            "Violation rate",
            f"{selected['violation_rate'].mean():.1%}",
            delta(summary, controller, "violation_rate"),
        )
        columns[4].markdown(
            f'<span class="safety-chip {status_class}">SAFETY · {status}</span>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            metric = st.selectbox(
                "Scenario heatmap metric", ["violation_rate", "mean_p_dc", "mean_pue"]
            )
            st.plotly_chart(
                scenario_heatmap(selected, metric, context.held_out_scenarios),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(controller_scatter(summary), use_container_width=True)
        trajectory = results[controller].trajectories
        scenario = st.selectbox("Replay scenario", sorted(trajectory["scenario"].unique()))
        episode = st.selectbox(
            "Replay episode",
            sorted(trajectory.loc[trajectory["scenario"] == scenario, "episode"].unique()),
        )
        replay = trajectory.loc[
            (trajectory["scenario"] == scenario) & (trajectory["episode"] == episode)
        ]
        st.plotly_chart(
            replay_chart(replay, st.session_state["outlet_limit"]), use_container_width=True
        )
        surface_column, room_column = st.columns(2)
        with surface_column:
            configuration = st.selectbox(
                "Surface configuration", sorted(context.training_data["configuration"].unique())
            )
            surface_outlet = st.slider("Surface outlet temperature (°C)", 15.0, 35.0, 25.0, 0.5)
            st.plotly_chart(
                pue_surface(context, configuration, surface_outlet), use_container_width=True
            )
        with room_column:
            configuration = context.evaluation_data.loc[
                context.evaluation_data["scenario"] == scenario, "configuration"
            ].iloc[0]
            render_room(replay, configuration)

with compare_tab:
    if summary.empty:
        st.info("Run two or more controllers to compare physical metrics, actions, and reward.")
    else:
        table = comparison_table(summary)
        visible = table[
            [
                "controller",
                "P_DC (mean ± std)",
                "mean_pue",
                "mean_cop",
                "max_outlet_temperature",
                "violation_rate",
                "max_violation",
                "switching_rate",
                "reward",
            ]
        ]
        st.subheader("Physical metrics alongside reward")
        st.dataframe(
            visible.style.highlight_min(
                subset=["mean_pue", "max_outlet_temperature", "violation_rate", "max_violation", "switching_rate"],
                color="#14532d",
            ).highlight_max(subset=["mean_cop", "reward"], color="#14532d"),
            hide_index=True,
            use_container_width=True,
        )
        trajectories = pd.concat(
            [result.trajectories for result in results.values()], ignore_index=True
        )
        curves = pd.concat(
            [result.learning_curve for result in results.values()], ignore_index=True
        )
        left, right = st.columns(2)
        with left:
            if not curves.empty:
                st.plotly_chart(learning_curve_chart(curves), use_container_width=True)
                st.caption(
                    "Learning curves show the selected seed. Run repeated seeds for report-quality uncertainty bands."
                )
        with right:
            st.plotly_chart(action_distribution_chart(trajectories), use_container_width=True)
        if "Rule-based" in results and len(results) > 1:
            st.plotly_chart(paired_savings_chart(summary), use_container_width=True)
        st.download_button(
            "Download comparison table",
            visible.to_csv(index=False).encode(),
            "controller_comparison.csv",
            "text/csv",
        )
        st.download_button(
            "Download raw trajectories",
            trajectories.to_csv(index=False).encode(),
            "controller_trajectories.csv",
            "text/csv",
        )

with diagnostics_tab:
    if context is None:
        st.info("Run an experiment to inspect the transition model.")
    else:
        st.caption(
            "In-sample errors are optimistic. Use the held-out split for report-quality assessment."
        )
        scores = diagnostics_frame(context.diagnostics)
        st.plotly_chart(
            dark_layout(
                px.bar(
                    scores,
                    x="target",
                    y="mae",
                    color="partition",
                    barmode="group",
                    template="plotly_dark",
                )
            ),
            use_container_width=True,
        )
        target = st.selectbox(
            "Predicted-vs-actual target",
            sorted(context.diagnostics.evaluation_predictions["target"].unique()),
        )
        prediction = context.diagnostics.evaluation_predictions.query("target == @target")
        figure = px.scatter(
            prediction, x="actual", y="predicted", color="scenario", template="plotly_dark"
        )
        minimum, maximum = (
            min(prediction["actual"].min(), prediction["predicted"].min()),
            max(prediction["actual"].max(), prediction["predicted"].max()),
        )
        figure.add_shape(
            type="line",
            x0=minimum,
            y0=minimum,
            x1=maximum,
            y1=maximum,
            line={"dash": "dash", "color": "#fbbf24"},
        )
        st.plotly_chart(dark_layout(figure), use_container_width=True)
        left, right = st.columns(2)
        with left:
            errors = context.diagnostics.held_out_by_scenario.pivot(
                index="scenario", columns="target", values="mae"
            )
            st.plotly_chart(
                dark_layout(
                    go.Figure(
                        go.Heatmap(
                            z=errors.to_numpy(),
                            x=errors.columns,
                            y=errors.index,
                            colorscale="Blues",
                        )
                    )
                ),
                use_container_width=True,
            )
        with right:
            importance = context.diagnostics.feature_importances
            st.plotly_chart(
                dark_layout(
                    px.bar(
                        importance,
                        x="importance",
                        y="feature",
                        orientation="h",
                        template="plotly_dark",
                    )
                ),
                use_container_width=True,
            )
        clipped = sum(result.proposed_supply_out_of_range_steps for result in results.values())
        st.metric("Proposed supply settings clipped to training range", clipped)

with source_tab:
    if data is None:
        st.info(data_error or "Provide a valid AAU data folder to inspect source data.")
    else:
        interval = data.groupby("scenario")["time"].diff().dropna().median()
        columns = st.columns(4)
        columns[0].metric("Rows", f"{len(data):,}")
        columns[1].metric("Missing values", int(data.isna().sum().sum()))
        columns[2].metric("Duplicate rows", int(data.duplicated().sum()))
        columns[3].metric("Median interval", f"{interval:.2f}" if pd.notna(interval) else "N/A")
        configurations = st.multiselect(
            "Configurations",
            sorted(data["configuration"].unique()),
            default=sorted(data["configuration"].unique()),
        )
        loads = st.multiselect(
            "Load (%)",
            sorted((data["load_fraction"] * 100).unique()),
            default=sorted((data["load_fraction"] * 100).unique()),
        )
        filtered = data.loc[
            data["configuration"].isin(configurations) & (data["load_fraction"] * 100).isin(loads)
        ]
        if filtered.empty:
            st.info("No rows match the selected configuration and load filters.")
        else:
            selected_columns = st.multiselect(
                "Columns",
                list(filtered.columns),
                default=["time", "scenario", "Outlet Avg Temp", "Cooling Supply Average", "P_DC", "PUE"],
            )
            st.dataframe(filtered[selected_columns], use_container_width=True, height=280)
            st.download_button(
                "Download filtered source data",
                filtered.to_csv(index=False).encode(),
                "aau_filtered.csv",
                "text/csv",
            )
            st.warning(
                "Cooling supply temperature is an observed proxy only; AAU files contain no logged control-action field."
            )
            supply = px.scatter(
                filtered,
                x="load_fraction",
                y="Cooling Supply Average",
                color="configuration",
                template="plotly_dark",
            )
            supply.update_xaxes(tickformat=".0%", title="Load")
            st.plotly_chart(dark_layout(supply), use_container_width=True)
            series_column = st.selectbox(
                "Time-series column",
                [column for column in filtered.columns if column not in {"scenario", "configuration"}],
            )
            preview_scenario = st.selectbox("Preview scenario", sorted(filtered["scenario"].unique()))
            st.plotly_chart(
                dark_layout(
                    px.line(
                        filtered.loc[filtered["scenario"] == preview_scenario],
                        x="time",
                        y=series_column,
                        template="plotly_dark",
                    )
                ),
                use_container_width=True,
            )
