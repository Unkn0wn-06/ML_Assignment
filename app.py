"""Control-room dashboard for honest AAU cooling-controller experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from time import monotonic

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_centre_rl.data import STATE_COLUMNS, load_csv_directory
from data_centre_rl.experiments import (
    CONTROLLERS,
    ControllerResult,
    ExperimentContext,
    action_sensitivity,
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
from ui.theme import CONTROLLER_COLOURS, apply_control_room_theme, card

st.set_page_config(page_title="Cooling Control Room", page_icon="❄️", layout="wide")
apply_control_room_theme(st)


@st.cache_data(show_spinner=False)
def load_data(data_directory: str) -> pd.DataFrame:
    return load_csv_directory(data_directory)


px.defaults.color_discrete_sequence = list(CONTROLLER_COLOURS.values())

cached_context = st.cache_resource(show_spinner=False)(prepare_experiment)


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
        
        mean_pue=("mean_pue", "mean"),
        mean_cop=("mean_cop", "mean"),
        max_outlet_temperature=("max_outlet_temperature", "max"),
        violation_rate=("violation_rate", "mean"),
        max_violation=("max_violation", "max"),
        switching_rate=("switching_rate", "mean"),
        reward=("total_reward", "mean"),
    )
    return grouped


def pue_surface(context: ExperimentContext, configuration: str, outlet: float) -> go.Figure:
    local_supply=context.training_data.loc[context.training_data.configuration==configuration,"Cooling Supply Average"]
    supply_low=max(local_supply.min(),context.supply_bounds[0])
    supply_high=min(local_supply.max(),context.supply_bounds[1])
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
    ranges = observed.groupby("load_fraction")["Cooling Supply Average"].agg(["min", "max"])
    supported = grid.join(ranges, on="load_fraction")
    flagged = supported.loc[
        ~supported["Cooling Supply Average"].between(supported["min"], supported["max"])
    ]
    observed = observed.sample(min(200, len(observed)), random_state=42)
    figure = go.Figure(
        go.Surface(
            z=surface.to_numpy(),
            x=surface.columns * 100,
            y=surface.index,
            colorscale="Viridis",
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
            marker={"size": 2, "color": "#fbbf24", "opacity": 0.45},
        )
    )
    if not flagged.empty:
        figure.add_scatter3d(
            x=flagged.load_fraction * 100,
            y=flagged["Cooling Supply Average"],
            z=flagged.PUE,
            mode="markers",
            marker={"symbol": "x", "size": 3, "color": "#fb7185"},
            name="Outside local range",
        )
    figure.update_scenes(camera_eye={"x": 1.6, "y": 1.6, "z": 1.25})
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
    st.caption("CONTROLLERS")
    controllers = st.multiselect(
        "Controllers", available, default=["Rule-based", "Q-Learning", "SARSA"]
    )
    if disabled:
        st.info('DQN and PPO require `python -m pip install -e ".[deep-rl]"`.')
    st.caption("SPLIT · use a hold-out for report numbers")
    split_name = st.selectbox(
        "Evaluation split",
        [
            "All scenarios",
            "Load generalisation: hold out 100%",
            *[
                f"Hold out configuration: {name}"
                for name in ["Baseline", "Plate", "PlateWall", "Wall"]
            ],
        ],
    )
    st.caption("SIMULATION")
    safe = st.checkbox("Start at or below the limit", True)
    outlet_limit = st.slider("Outlet limit (°C)", 24.0, 32.0, 29.0, 0.25)
    horizon = st.slider("Episode horizon", 5, 200, 30, 5)
    repetitions = st.slider("Episodes per scenario", 1, 10, 1)
    st.caption("TRAINING")
    tabular_episodes = st.slider("Tabular RL episodes", 25, 1_000, 25, 25)
    deep_timesteps = st.slider("Deep RL timesteps", 1_000, 100_000, 20_000, 1_000)
    trees = st.slider("Transition-model trees", 20, 200, 50, 10)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    seed_count = st.number_input("Consecutive training seeds", 1, 10, 1)
    run = st.button("Run experiment", type="primary", width="stretch", disabled=not controllers)

    if st.button("Clear results"):
        for key in ("context", "controller_results", "run_folder", "infeasible", "run_cache"):
            st.session_state.pop(key, None)

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
    live = st.sidebar.empty()
    started = monotonic()
    try:
        with st.spinner("Running the learned simulator..."):
            context = cached_context(data, split_name=split_name, trees=trees, seed=int(seed))
            results: dict[str, ControllerResult] = {}
            cache = st.session_state.setdefault("run_cache", {})
            for index, controller in enumerate(controllers):
                history = []

                def report(
                    current: int,
                    total: int,
                    reward: float,
                    *,
                    offset=index,
                    controller_name=controller,
                    history=history,
                ) -> None:
                    fraction = (offset + current / max(total, 1)) / len(controllers)
                    history.append(reward)
                    if current % max(1, total // 40) and current != total:
                        return
                    live.line_chart(history[-100:])
                    progress.progress(
                        min(fraction, 1.0),
                        text=f"Training {controller_name}: {current / total:.0%} · {monotonic() - started:.0f}s",
                    )

                runs = []
                for training_seed in range(int(seed), int(seed + seed_count)):
                    key = (
                        id(context),
                        controller,
                        outlet_limit,
                        horizon,
                        repetitions,
                        tabular_episodes,
                        deep_timesteps,
                        training_seed,
                        safe,
                    )
                    if key not in cache:
                        cache[key] = (
                            context,
                            run_controller(
                                context,
                                controller=controller,
                                outlet_limit=outlet_limit,
                                horizon=horizon,
                                episodes_per_scenario=repetitions,
                                tabular_episodes=tabular_episodes,
                                deep_timesteps=deep_timesteps,
                                seed=training_seed,
                                progress_callback=report,
                                safe_starts=safe,
                            ),
                        )
                    runs.append(cache[key][1])
                results[controller] = ControllerResult(
                    controller,
                    pd.concat([r.scenario_metrics for r in runs]),
                    pd.concat([r.trajectories for r in runs]),
                    pd.concat([r.learning_curve for r in runs]),
                    sum(r.training_seconds for r in runs),
                    sum(r.proposed_supply_out_of_range_steps for r in runs),
                )
            st.session_state.update(
                context=context,
                controller_results=results,
                outlet_limit=outlet_limit,
                run_folder=data_directory,
                infeasible=sorted(
                    set(context.evaluation_data.scenario)
                    - set(results[controllers[0]].trajectories.get("scenario", []))
                ),
            )
    except (ImportError, RuntimeError, ValueError) as error:
        st.error(str(error))
    finally:
        progress.empty()
        live.empty()
elif run:
    st.error(data_error or "Choose a valid AAU data folder.")

overview_tab, compare_tab, diagnostics_tab, source_tab = st.tabs(
    ["Overview", "Compare controllers", "Model diagnostics", "Source data"]
)
context: ExperimentContext | None = st.session_state.get("context")
results: dict[str, ControllerResult] = st.session_state.get("controller_results", {})
if data_directory != st.session_state.get("run_folder") or data is None:
    context, results = None, {}
summary = (
    summarise(results, context.evaluation_data)
    if results and any(not result.scenario_metrics.empty for result in results.values())
    else pd.DataFrame()
)
if context:
    missing = st.session_state["infeasible"]
    st.caption(
        f"{context.split_name} · {len(missing)} scenarios infeasible at this limit"
        + (" · " + ", ".join(missing) if missing else "")
    )

with overview_tab:
    if summary.empty:
        st.info(
            "Run one or more controllers from the Experiment panel to populate this control-room overview."
        )
    else:
        selectors = st.columns(3)
        controller = selectors[0].selectbox("Overview controller", list(results))
        selected = summary.loc[summary["controller"] == controller]
        status, status_class = safety_status(
            selected["max_outlet_temperature"].max(), st.session_state["outlet_limit"]
        )
        columns = st.columns(5)
        columns[0].metric(
            "Mean P_DC (kW)",
            f"{selected['mean_p_dc'].mean():.3f}",
            delta(summary, controller, "mean_p_dc"),
        )
        columns[1].metric(
            "Mean PUE", f"{selected['mean_pue'].mean():.2f}", delta(summary, controller, "mean_pue")
        )
        columns[2].metric(
            "Mean outlet",
            f"{selected['mean_outlet_temperature'].mean():.1f} °C",
            delta(summary, controller, "mean_outlet_temperature"),
        )
        columns[3].metric(
            "Violation rate",
            f"{selected['violation_rate'].mean():.1%}",
            delta(summary, controller, "violation_rate"),
        )
        columns[4].metric(
            "Safety status",
            {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴"}[status] + " " + status.title(),
        )
        trajectory = results[controller].trajectories
        scenario = selectors[1].selectbox(
            "Replay scenario", sorted(trajectory["scenario"].unique())
        )
        replay_seed = (
            selectors[1].selectbox("Replay seed", sorted(trajectory.seed.unique()))
            if trajectory.seed.nunique() > 1
            else trajectory.seed.iloc[0]
        )
        trajectory = trajectory.loc[trajectory.seed == replay_seed]
        episode = selectors[2].selectbox(
            "Replay episode",
            sorted(trajectory.loc[trajectory["scenario"] == scenario, "episode"].unique()),
        )
        replay = trajectory.loc[
            (trajectory["scenario"] == scenario) & (trajectory["episode"] == episode)
        ]
        room_column, surface_column = st.columns([1.3, 1])
        with card(
            surface_column,
            "Transition surface",
            "Predicted PUE at the chosen outlet temperature; dots are observed AAU rows.",
        ):
            config_slot, outlet_slot = st.columns(2)
            configuration = config_slot.selectbox(
                "Surface configuration", sorted(context.training_data["configuration"].unique())
            )
            surface_outlet = outlet_slot.slider(
                "Surface outlet temperature (°C)", 15.0, 35.0, 25.0, 0.5
            )
            st.plotly_chart(pue_surface(context, configuration, surface_outlet), width="stretch")
        with card(room_column, "Server room", "Replay of simulated average outlet temperature."):
            configuration = context.evaluation_data.loc[
                context.evaluation_data["scenario"] == scenario, "configuration"
            ].iloc[0]
            render_room(replay, configuration)
        heat, episode_panel, trade = st.columns(3)
        with card(heat, "Scenario map", "Grey n/a cells have no evaluated steps."):
            metric = st.selectbox("Metric", ["violation_rate", "mean_p_dc", "mean_pue"])
            st.plotly_chart(scenario_heatmap(selected, metric, context.held_out_scenarios))
        with card(episode_panel, "Episode replay", "Warning band starts 1 °C below the limit."):
            st.plotly_chart(replay_chart(replay, st.session_state["outlet_limit"]))
        with card(
            trade, "Controller trade-offs", "Marker size in the projection reflects switching rate."
        ):
            st.plotly_chart(controller_scatter(summary))
            points = summary.groupby("controller", as_index=False).mean(numeric_only=True)
            points["marker_size"] = points.switching_rate + 0.02
            st.plotly_chart(
                dark_layout(
                    px.scatter(
                        points,
                        x="mean_p_dc",
                        y="violation_rate",
                        size="marker_size",
                        color="controller",
                        color_discrete_map=CONTROLLER_COLOURS,
                        labels={"mean_p_dc": "Power (kW)", "violation_rate": "Violation rate"},
                    )
                )
            )

with compare_tab:
    if summary.empty:
        st.info("Run two or more controllers to compare physical metrics, actions, and reward.")
    else:
        with card(
            st,
            "Controller performance",
            "Episode averages across evaluated scenarios. Green highlights the best value, including ties.",
        ):
            table = comparison_table(summary)
            visible = table[
                [
                    "controller",
                    "mean_p_dc",
                    "mean_pue",
                    "mean_cop",
                    "max_outlet_temperature",
                    "violation_rate",
                    "max_violation",
                    "switching_rate",
                    "reward",
                ]
            ]

            st.dataframe(
                visible.style.format(
                    {
                        "mean_p_dc": "{:.3f}",
                        "mean_pue": "{:.2f}",
                        "mean_cop": "{:.2f}",
                        "max_outlet_temperature": "{:.1f}",
                        "max_violation": "{:.1f}",
                        "violation_rate": "{:.1%}",
                        "switching_rate": "{:.1%}",
                        "reward": "{:.1f}",
                    }
                )
                .highlight_min(
                    subset=[
                        "mean_p_dc",
                        "mean_pue",
                        "max_outlet_temperature",
                        "violation_rate",
                        "max_violation",
                        "switching_rate",
                    ],
                    color="#14532d",
                )
                .highlight_max(subset=["mean_cop", "reward"], color="#14532d"),
                column_config={
                    "mean_p_dc": "Power (kW)",
                    "mean_pue": "PUE",
                    "mean_cop": "CoP",
                    "max_outlet_temperature": "Peak outlet (°C)",
                    "max_violation": "Peak excess (°C)",
                    "switching_rate": "Switching",
                    "violation_rate": "Violations",
                    "reward": "Return",
                },
                hide_index=True,
                width="stretch",
            )
        trajectories = pd.concat(
            [result.trajectories for result in results.values()], ignore_index=True
        )
        curves = pd.concat(
            [result.learning_curve for result in results.values()], ignore_index=True
        )
        left, right = st.columns(2)
        with card(left, "Learning curves", "Rolling reward over ten training observations."):
            if not curves.empty:
                st.plotly_chart(learning_curve_chart(curves), width="stretch")
                st.caption(
                    "Single seed · faint raw rewards and 10-point rolling mean."
                    if curves.seed.nunique() == 1
                    else "Shaded band spans ±1 standard deviation across seeds."
                )
        with card(right, "Action distribution", "Share of evaluated steps for each controller."):
            st.plotly_chart(action_distribution_chart(trajectories), width="stretch")
        if summary.episode.max() < 1:
            st.info("Raise Episodes per scenario to at least 2 for paired savings intervals.")
        elif "Rule-based" in results and len(results) > 1:
            card(
                st,
                "Paired energy saving",
                "Common steps from matched starts, approximate 95% interval.",
            ).plotly_chart(paired_savings_chart(trajectories), width="stretch")
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
        if not context.held_out_scenarios:
            st.info(
                "All scenarios uses the same rows for fitting and evaluation. Both MAE bars are identical by design."
            )
        sensitivity = action_sensitivity(context)
        with card(
            st,
            "Action sensitivity",
            "Near-zero changes give RL little opportunity to improve on the rule-based controller in this simulator.",
        ):
            for panel, target, unit in zip(
                st.columns(2), ["Outlet Avg Temp", "P_DC"], ["°C", "kW"], strict=True
            ):
                panel.plotly_chart(
                    dark_layout(
                        px.bar(
                            sensitivity.assign(shift=sensitivity["shift"].astype(str)),
                            x="scenario",
                            y=target,
                            color="shift",
                            barmode="group",
                            labels={target: f"Change ({unit})", "shift": "Supply step °C"},
                            color_discrete_sequence=["#38bdf8", "#f59e0b"],
                        )
                    )
                )
        with card(st, "Prediction error", "Mean absolute error in each target’s original units."):
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
                width="stretch",
            )
        with card(
            st, "Predicted versus observed", "The dashed diagonal denotes perfect agreement."
        ):
            target = st.selectbox(
                "Predicted-vs-actual target",
                sorted(context.diagnostics.evaluation_predictions["target"].unique()),
            )
            prediction = context.diagnostics.evaluation_predictions.query("target == @target")
            figure = px.scatter(
                prediction,
                x="actual",
                y="predicted",
                color_discrete_sequence=["#38bdf8"],
                template="plotly_dark",
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
            st.plotly_chart(dark_layout(figure), width="stretch")
        left, right = st.columns(2)
        with card(left, "Error by scenario", "Mean absolute error by predicted target."):
            errors = context.diagnostics.held_out_by_scenario.pivot(
                index="scenario", columns="target", values="mae"
            )
            st.plotly_chart(
                dark_layout(
                    go.Figure(
                        go.Heatmap(
                            z=errors.to_numpy(),
                            texttemplate="%{z:.3f}",
                            x=errors.columns,
                            y=errors.index,
                            colorscale="Blues",
                        )
                    )
                ),
                width="stretch",
            )
        with card(
            right, "Feature importance", "Relative contribution to forest split impurity reduction."
        ):
            importance = context.diagnostics.feature_importances.copy()
            importance["feature"] = importance.feature.str.replace(
                r"^(numeric__|configuration__configuration_)", "", regex=True
            )
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
                width="stretch",
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
        columns[3].metric("Median interval", f"{interval:g} min" if pd.notna(interval) else "N/A")
        st.caption("Supply temperature is an observed proxy for the cooling action.")
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
            with card(
                st, "AAU observations", "Filter the rows and columns to inspect or download."
            ):
                selected_columns = st.multiselect(
                    "Columns",
                    list(filtered.columns),
                    default=[
                        "time",
                        "scenario",
                        "Outlet Avg Temp",
                        "Cooling Supply Average",
                        "P_DC",
                        "PUE",
                    ],
                )
                st.dataframe(
                    filtered[selected_columns].style.format(precision=3),
                    width="stretch",
                    height=280,
                )
                st.download_button(
                    "Download filtered source data",
                    filtered[selected_columns].to_csv(index=False).encode(),
                    "aau_filtered.csv",
                    "text/csv",
                )

            with card(
                st, "Cooling supply by load", "Each box summarises a configuration and load."
            ):
                supply = px.box(
                    filtered,
                    x="load_fraction",
                    y="Cooling Supply Average",
                    color="configuration",
                    template="plotly_dark",
                )
                supply.update_xaxes(tickformat=".0%", title="Load")
                st.plotly_chart(dark_layout(supply), width="stretch")
            with card(
                st, "Time series", "Each trace follows the original two-minute measurements."
            ):
                metrics = st.multiselect(
                    "Time-series metrics",
                    [
                        name
                        for name in filtered.select_dtypes("number")
                        if name not in {"time", "load_fraction"}
                    ],
                    default=["Outlet Avg Temp"],
                )
                scenario = st.selectbox("Preview scenario", sorted(filtered.scenario.unique()))
                preview = filtered.loc[filtered.scenario == scenario].assign(
                    hours=lambda rows: rows.time / 60
                )
                if metrics:
                    st.plotly_chart(
                        dark_layout(
                            px.line(preview, x="hours", y=metrics, labels={"hours": "Time (hours)"})
                        )
                    )
