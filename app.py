"""Control-room dashboard for honest AAU cooling-controller experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from time import monotonic

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_centre_rl.data import STATE_COLUMNS, load_csv_directory
from data_centre_rl.experiments import (
    BOUNDS,
    CONTROLLERS,
    ControllerResult,
    ExperimentContext,
    action_sensitivity,
    diagnostics_frame,
    prepare_experiment,
    run_controller,
)
from data_centre_rl.transition_model import VARIANTS
from ui.charts import (
    action_distribution_chart,
    dark_layout,
    final_difference,
    learning_curve_chart,
    paired_savings,
    replay_chart,
    scenario_heatmap,
    training_chart,
)
from ui.room3d import render_room
from ui.theme import CONTROLLER_COLOURS, TEMPERATURE_RAMP, apply_control_room_theme, card

st.set_page_config(page_title="Cooling RL", page_icon="❄️", layout="wide")
apply_control_room_theme(st)


@st.cache_data(show_spinner=False)
def load_data(data_directory: str) -> pd.DataFrame:
    return load_csv_directory(data_directory)


px.defaults.color_discrete_sequence = list(CONTROLLER_COLOURS.values())

cached_context = st.cache_resource(show_spinner=False)(prepare_experiment)


def enabled_controllers() -> tuple[list[str], list[str]]:
    if importlib.util.find_spec("stable_baselines3"):
        return list(CONTROLLERS), []
    return [name for name in CONTROLLERS if name not in {"DQN", "PPO"}], ["DQN", "PPO"]


def summarise(results: dict[str, ControllerResult], data: pd.DataFrame) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    metrics = pd.concat([result.scenario_metrics for result in results.values()], ignore_index=True)
    metadata = data.groupby("scenario", as_index=False).agg(
        configuration=("configuration", "first"), load_percent=("load_fraction", "first")
    )
    metadata["load_percent"] *= 100
    summary = metrics.merge(metadata, on="scenario", how="left")
    summary["episode_violation_rate"] = summary.any_violation.astype(float)
    summary["mean_steps_survived"] = summary.steps_survived
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
        episode_violation_rate=("episode_violation_rate", "mean"),
        max_violation=("max_violation", "max"),
        mean_steps_survived=("steps_survived", "mean"),
        switching_rate=("switching_rate", "mean"),
        violating_step_share=("violating_step_share", "mean"),
        reward=("return_per_step", "mean"),
    )
    rates = summary.groupby(["controller", "seed"])[["violation_count", "steps"]].sum()
    grouped["violating_step_share"] = grouped.controller.map(
        (rates.violation_count / rates.steps).groupby(level=0).mean()
    )
    return grouped


def pue_surface(context: ExperimentContext, configuration: str, outlet: float) -> go.Figure:
    local_supply = context.training_data.loc[
        context.training_data.configuration == configuration, "Cooling Supply Average"
    ]
    supply_low = max(local_supply.min(), context.supply_bounds[0])
    supply_high = min(local_supply.max(), context.supply_bounds[1])
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
st.markdown('<div class="control-title">Cooling RL</div>', unsafe_allow_html=True)
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
        "Controllers", available, default=["Rule-based", "Q-Learning", "SARSA", *BOUNDS]
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
    variant = st.selectbox(
        "Simulator variant",
        VARIANTS,
        help="Levels predicts absolute outcomes. Deltas learns temperature changes. Monotone boosting assumes warmer supply cannot lower predicted outlet or return temperature.",
    )
    safe = st.checkbox("Start at or below the limit", True)
    outlet_limit = st.slider("Outlet limit (°C)", 24.0, 32.0, 29.0, 0.25)
    horizon = st.slider("Episode horizon", 5, 200, 50, 5)
    repetitions = st.slider("Episodes per scenario", 1, 10, 3)
    st.caption("TRAINING")
    with st.expander("Advanced tabular settings"):
        release_band = st.number_input("Rule release band (°C)", 0.0, 2.0, 0.5)
        min_hold_steps = st.number_input("Minimum hold steps", 1, 100, 5)
        legacy_state = (
            st.selectbox("Tabular state", ["Safety state", "Legacy 5-bin state"])
            == "Legacy 5-bin state"
        )
        checkpoint_config = json.loads((Path(__file__).parent / "configs/base.json").read_text())
        training_options = {
            "alpha": st.number_input("Alpha", 0.01, 1.0, 0.1),
            "gamma": st.number_input("Gamma", 0.0, 1.0, 0.9),
            "epsilon_start": st.number_input("Starting exploration", 0.0, 1.0, 1.0),
            "epsilon_end": st.number_input("Final exploration", 0.0, 1.0, 0.05),
            "decay_fraction": st.number_input("Exploration decay fraction", 0.01, 1.0, 0.6),
            "min_visits": st.number_input("Minimum trusted visits", 1, 1000, 10),
        }
        legacy_termination = st.checkbox("Legacy termination")
        tie_margin = st.number_input(
            "Near-tie margin (reward units)", min_value=0.0, value=0.05, step=0.01
        )
        zero_initialisation = st.checkbox("Legacy zero initial Q-values")
    tabular_episodes = st.slider("Tabular RL episodes", 25, 1_000, 300, 25)
    deep_timesteps = st.slider("Deep RL timesteps", 1_000, 100_000, 20_000, 1_000)
    trees = st.slider("Transition-model trees", 20, 200, 50, 10)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    seed_count = st.number_input("Consecutive training seeds", 1, 10, 3)
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
            context = cached_context(
                data, split_name=split_name, variant=variant, trees=trees, seed=int(seed)
            )
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
                        release_band,
                        min_hold_steps,
                        legacy_state,
                        tuple(training_options.items()),
                        checkpoint_config["sarsa" if controller == "SARSA" else "q_learning"][
                            "eval_every"
                        ],
                        legacy_termination,
                        tie_margin,
                        zero_initialisation,
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
                                release_band=release_band,
                                min_hold_steps=min_hold_steps,
                                legacy_state=legacy_state,
                                training_options={
                                    **training_options,
                                    "eval_every": checkpoint_config[
                                        "sarsa" if controller == "SARSA" else "q_learning"
                                    ]["eval_every"],
                                },
                                legacy_termination=legacy_termination,
                                tie_margin=tie_margin,
                                zero_initialisation=zero_initialisation,
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
    ["Viewer", "Compare", "Diagnostics", "Source data"]
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
        selectors = st.columns([1, 0.65, 1.5, 0.65, 1])
        controller = selectors[0].selectbox(
            "Viewer controller",
            list(results),
            index=list(results).index("Rule-based") if "Rule-based" in results else 0,
        )
        selected = summary.loc[summary["controller"] == controller]
        trajectory = results[controller].trajectories
        replay_seed = (
            selectors[1].selectbox("Replay seed", sorted(trajectory.seed.unique()))
            if trajectory.seed.nunique() > 1
            else trajectory.seed.iloc[0]
        )
        trajectory = trajectory.loc[trajectory.seed == replay_seed]
        full_load = selected.loc[selected.load_percent.eq(100)]
        held = full_load.loc[full_load.scenario.isin(context.held_out_scenarios)]
        candidates = held if not held.empty else full_load if not full_load.empty else selected
        default_scenario = candidates.groupby("scenario").any_violation.mean().idxmax()
        episodes = selected.loc[
            (selected.scenario == default_scenario) & (selected.seed == replay_seed)
        ]
        default_episode = (
            episodes.sort_values(["violation_degree_steps", "episode"], ascending=[False, True])
            .iloc[0]
            .episode
        )
        stress = (
            context.evaluation_data.assign(
                hot=lambda rows: rows["Outlet Avg Temp"] > st.session_state["outlet_limit"] - 1
            )
            .groupby("scenario")
            .hot.mean()
        )
        scenarios = sorted(trajectory.scenario.unique())
        scenario = selectors[2].selectbox(
            "Replay scenario",
            scenarios,
            index=scenarios.index(default_scenario),
            format_func=lambda name: f"{name} · {'stressed' if stress[name] > 0.1 else 'calm'}",
        )
        episode = selectors[3].selectbox(
            "Replay episode",
            sorted(trajectory.loc[trajectory["scenario"] == scenario, "episode"].unique()),
            index=int(default_episode) if scenario == default_scenario else 0,
        )
        replay = trajectory.loc[
            (trajectory["scenario"] == scenario) & (trajectory["episode"] == episode)
        ]
        outcome = selected.loc[
            (selected.scenario == scenario)
            & (selected.episode == episode)
            & (selected.seed == replay_seed)
        ].iloc[0]
        other = selectors[4].selectbox(
            "Compare with", [name for name in results if name != controller] or ["None"]
        )
        comparison = None
        if other != "None":
            traces = results[other].trajectories
            comparison = traces.loc[
                (traces.scenario == scenario)
                & (traces.episode == episode)
                & (traces.seed == replay_seed)
                & (traces.start_row == outcome.start_row)
            ]
        render_room(
            replay,
            comparison,
            results,
            selected,
            context.evaluation_data,
            outcome,
            st.session_state["outlet_limit"],
        )
        with st.expander("Run metrics and analysis charts"):
            status, status_class = safety_status(
                selected["max_outlet_temperature"].max(), st.session_state["outlet_limit"]
            )
            columns = st.columns(6)
            columns[5].metric(
                "Steps survived",
                f"{selected.steps_survived.mean():.1f}",
                help=f"Mean executed steps out of {int(selected.horizon.iloc[0])}.",
            )
            columns[0].metric(
                "Mean P_DC (kW)",
                f"{selected['mean_p_dc'].mean():.3f}",
                delta(summary, controller, "mean_p_dc"),
            )
            columns[1].metric(
                "Mean PUE",
                f"{selected['mean_pue'].mean():.2f}",
                delta(summary, controller, "mean_pue"),
            )
            columns[2].metric(
                "Mean outlet",
                f"{selected['mean_outlet_temperature'].mean():.1f} °C",
                delta(summary, controller, "mean_outlet_temperature"),
            )
            columns[3].metric(
                "Episodes violated",
                f"{selected['episode_violation_rate'].mean():.1%}",
                delta(summary, controller, "episode_violation_rate"),
                help="Share of episodes with any outlet temperature above the limit.",
            )
            columns[4].metric(
                "Safety status",
                {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴"}[status] + " " + status.title(),
            )
            with card(
                st,
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
                st.plotly_chart(
                    pue_surface(context, configuration, surface_outlet), width="stretch"
                )
            heat, trade = st.columns(2)
            with card(heat, "Scenario map", "Grey n/a cells have no evaluated steps."):
                metric = st.selectbox(
                    "Metric",
                    [
                        "episode_violation_rate",
                        "mean_steps_survived",
                        "mean_p_dc",
                        "mean_pue",
                        "mean_outlet_temperature",
                    ],
                )
                st.plotly_chart(
                    scenario_heatmap(
                        selected,
                        metric,
                        context.held_out_scenarios,
                        st.session_state["outlet_limit"],
                    )
                )
            with card(
                trade,
                "Controller trade-offs",
                "Bars span one standard deviation across seeds. Marker size shows switching rate.",
            ):
                seeds = summary.groupby(["controller", "seed"])[
                    ["mean_p_dc", "episode_violation_rate", "switching_rate"]
                ].mean()
                points = seeds.groupby(level=0).mean()
                spread = seeds.groupby(level=0).std().fillna(0)
                zoom = st.checkbox("Zoom to controllers", value=True)
                figure = go.Figure()
                for i, (name, point) in enumerate(points.iterrows()):
                    figure.add_scatter(
                        x=[point.mean_p_dc],
                        y=[point.episode_violation_rate],
                        mode="markers",
                        name=name,
                        marker={
                            "size": 9 + 30 * point.switching_rate,
                            "color": CONTROLLER_COLOURS[name],
                        },
                        error_x={"array": [spread.loc[name, "mean_p_dc"]]},
                        error_y={"array": [spread.loc[name, "episode_violation_rate"]]},
                    )
                    figure.add_annotation(
                        x=point.mean_p_dc,
                        y=point.episode_violation_rate,
                        text=name,
                        ax=35 if i % 2 else -35,
                        ay=(-1) ** i * (20 + 8 * i),
                        arrowhead=0,
                    )
                figure.update_xaxes(
                    title="Mean power (kW)", rangemode="normal" if zoom else "tozero"
                )
                figure.update_yaxes(
                    title="Episodes with any violation",
                    tickformat=".0%",
                    rangemode="normal" if zoom else "tozero",
                )
                figure.update_layout(showlegend=False)
                st.plotly_chart(dark_layout(figure))
            with card(st, "Episode replay", "Warning band starts 1 °C below the limit."):
                st.caption(
                    f"{scenario} · {controller} · start row {outcome.start_row} · {outcome.steps_survived}/{outcome.horizon} steps · {outcome.end_reason} · {int(replay.action.diff().dropna().ne(0).sum())} switches · {int(replay.supply_clipped.sum())} clipped"
                )
                st.plotly_chart(replay_chart(replay, st.session_state["outlet_limit"], comparison))

with compare_tab:
    if not summary.empty and (summary.legacy_termination.any() or summary.legacy_state.any()):
        st.warning("Legacy setting: not comparable with the default results")
    if summary.empty:
        st.info("Run two or more controllers to compare physical metrics, actions, and reward.")
    else:
        with card(
            st,
            "Controller performance",
            "Episode averages across evaluated scenarios. Green highlights the best value, including ties.",
        ):
            table = comparison_table(summary)
            if context.model.variant == VARIANTS[1] and "Rule-based" in results:
                baseline = table.loc[table.controller == "Rule-based"].iloc[0]
                for name in ["Q-Learning", "SARSA"]:
                    row = table.loc[table.controller == name]
                    if row.empty:
                        continue
                    power_saving = 100 * (1 - row.mean_p_dc.iloc[0] / baseline.mean_p_dc)
                    hot_change = 100 * (
                        row.violating_step_share.iloc[0] - baseline.violating_step_share
                    )
                    if power_saving > 0 and hot_change > 0:
                        st.caption(
                            f"{name} power–safety trade-off versus hysteresis: {power_saving:.2f}% lower mean power and {hot_change:.2f} percentage points more violating steps."
                        )
            visible = table[
                [
                    "controller",
                    "mean_p_dc",
                    "episode_violation_rate",
                    "violating_step_share",
                    "mean_steps_survived",
                    "reward",
                    "switching_rate",
                    "mean_pue",
                    "mean_cop",
                    "max_outlet_temperature",
                    "max_violation",
                ]
            ]

            st.dataframe(
                visible.style.format(
                    {
                        "mean_p_dc": "{:.3f}",
                        "mean_steps_survived": "{:.1f}",
                        "mean_pue": "{:.2f}",
                        "mean_cop": "{:.2f}",
                        "max_outlet_temperature": "{:.1f}",
                        "max_violation": "{:.1f}",
                        "episode_violation_rate": "{:.1%}",
                        "switching_rate": "{:.1%}",
                        "violating_step_share": "{:.1%}",
                        "reward": "{:.2f}",
                    }
                )
                .highlight_min(
                    subset=[
                        "mean_p_dc",
                        "mean_pue",
                        "max_outlet_temperature",
                        "episode_violation_rate",
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
                    "mean_steps_survived": "Mean steps survived",
                    "episode_violation_rate": st.column_config.ProgressColumn(
                        "Episodes with any violation", min_value=0, max_value=1, format="percent"
                    ),
                    "violating_step_share": "Violating steps (%)",
                    "reward": "Return / step",
                },
                hide_index=True,
                width="stretch",
            )
        if any(name in results for name in BOUNDS):
            with card(
                st,
                "Cooling bounds",
                "Grey dashed lines show the two constant-action policies on the same starts.",
            ):
                metric = st.selectbox(
                    "Compare bound metric",
                    ["episode_violation_rate", "mean_p_dc", "mean_pue"],
                    format_func={
                        "episode_violation_rate": "Episodes with any violation",
                        "mean_p_dc": "Mean facility power (kW)",
                        "mean_pue": "Mean PUE",
                    }.get,
                )
                by_seed = summary.groupby(["controller", "seed"], as_index=False)[metric].mean()
                bound_fig = go.Figure()
                for name, group in by_seed.groupby("controller"):
                    if name in BOUNDS and len(group) == 1:
                        bound_fig.add_hline(
                            y=group[metric].iloc[0],
                            line_dash="dash",
                            line_color=CONTROLLER_COLOURS[name],
                        )
                    bound_fig.add_scatter(
                        x=group.seed,
                        y=group[metric],
                        name=name,
                        mode="lines+markers",
                        line={
                            "color": CONTROLLER_COLOURS[name],
                            "dash": "dash" if name in BOUNDS else "solid",
                        },
                    )
                bound_fig.update_xaxes(title="Seed", dtick=1)
                if metric == "episode_violation_rate":
                    bound_fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(dark_layout(bound_fig))
        trajectories = pd.concat(
            [result.trajectories for result in results.values()], ignore_index=True
        )
        curves = pd.concat(
            [result.learning_curve for result in results.values()], ignore_index=True
        )
        left, right = st.columns(2)
        with card(
            left,
            "Training diagnostics",
            "Tabular training episodes use matched fixed-setpoint starts.",
        ):
            if "reward_vs_baseline" in curves:
                tabular = curves.dropna(subset=["reward_vs_baseline"])
                st.markdown("**Reward vs fixed setpoint**")
                st.plotly_chart(training_chart(tabular, "reward_vs_baseline"))
                st.caption(final_difference(tabular))
                st.markdown("**Greedy evaluation on fixed starts**")
                metric = st.selectbox(
                    "Checkpoint metric",
                    [
                        "eval_reward",
                        "eval_violating_steps",
                        "eval_power",
                        "eval_switching",
                        "eval_fallback",
                    ],
                    format_func={
                        "eval_reward": "Reward per step",
                        "eval_violating_steps": "Violating-step share",
                        "eval_power": "Mean facility power (kW)",
                        "eval_switching": "Switching rate",
                        "eval_fallback": "Fallback-to-maintain share",
                    }.get,
                )
                st.plotly_chart(training_chart(tabular, metric))
                st.caption(
                    "Twelve fixed training-scenario starts. Bands show ±1 standard deviation across seeds when available."
                )
                st.markdown("**Exploration and fallback to maintain**")
                st.plotly_chart(training_chart(tabular, "epsilon"))
            deep = (
                curves[curves.controller.isin(["DQN", "PPO"])] if "controller" in curves else curves
            )
            if not deep.empty:
                st.plotly_chart(learning_curve_chart(deep))
        with card(right, "Action distribution", "Share of evaluated steps for each controller."):
            st.plotly_chart(action_distribution_chart(trajectories), width="stretch")
        if "Rule-based" in results and len(results) > 1:
            with card(
                st,
                "Paired power and safety",
                "Full-horizon episodes with matching starts. Differences below ~1% are not practically meaningful.",
            ):
                st.dataframe(paired_savings(summary), hide_index=True)
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
        tabular_rows = []
        for name, result in results.items():
            if "states_visited" in result.scenario_metrics:
                per_seed = result.scenario_metrics.groupby("seed").first()
                tabular_rows.append(
                    {
                        "Controller": name,
                        "States visited (seed mean)": per_seed.states_visited.mean(),
                        "Median trusted visits (seed mean)": per_seed.median_trusted_visits.mean(),
                        "Fallback decisions (%)": 100 * result.trajectories.fallback.mean(),
                        "Switching (%)": 100 * result.scenario_metrics.switching_rate.mean(),
                    }
                )
        if tabular_rows:
            st.subheader("Tabular policy diagnostics")
            st.dataframe(pd.DataFrame(tabular_rows).style.format(precision=2), hide_index=True)
        if not context.held_out_scenarios:
            st.info(
                "All scenarios uses the same rows for fitting and evaluation. Both MAE bars are identical by design."
            )
        with card(
            st,
            "Simulator comparison",
            "Validation uses the selected hold-out, or the 100% load hold-out for an all-scenarios run.",
        ):
            split = (
                context.split_name
                if context.held_out_scenarios
                else "Load generalisation: hold out 100%"
            )
            diagnostics = []
            responses = []
            for choice in VARIANTS:
                if split.startswith("Load") and data.load_fraction.nunique() < 2:
                    st.info("Load comparison requires training and held-out loads.")
                    break
                candidate = cached_context(
                    data,
                    split_name=split,
                    variant=choice,
                    trees=context.model.n_estimators,
                    seed=context.model.random_state,
                )
                effects = action_sensitivity(candidate)
                responses.append(
                    effects.assign(
                        variant=choice, spurious=(effects.applied < 0) & (effects.P_DC < 0)
                    )
                )
                diagnostics.append(
                    {
                        "variant": choice,
                        **candidate.diagnostics.held_out_mae,
                        "Mean |Δ outlet|": effects["Outlet Avg Temp"].abs().mean(),
                        "Mean |Δ kW|": effects.P_DC.abs().mean(),
                    }
                )
            if diagnostics:
                scores = pd.DataFrame(diagnostics).set_index("variant")
                st.dataframe(
                    scores.rename_axis("Simulator variant")
                    .rename(
                        columns={
                            "P_DC": "Facility power (kW)",
                            "Outlet Avg Temp": "Outlet temperature (°C)",
                            "Cooling Return Average": "Return temperature (°C)",
                        }
                    )
                    .style.format("{:.3f}")
                )
                for choice in VARIANTS[1:]:
                    worse = [
                        target
                        for target in context.diagnostics.in_sample_mae
                        if scores.loc[choice, target] > 1.10 * scores.iloc[0][target]
                    ]
                    if worse:
                        st.caption(
                            f"{choice}: held-out MAE exceeds the levels model by >10% for {', '.join(worse)}."
                        )
                details = pd.concat(responses, ignore_index=True)
                st.caption(
                    "Flagged scenarios predict lower power when supply is lowered. This may be a spurious correlation."
                )
                flags = details.loc[details.spurious]
                for variant, flagged in flags.groupby("variant"):
                    st.write(f"⚠ {variant}: {', '.join(flagged.scenario.unique())}")
                if flags.empty:
                    st.write("✓ No scenarios flagged.")
                response = (
                    details.assign(outlet_change=details["Outlet Avg Temp"].abs())
                    .groupby(["variant", "shift"], as_index=False)
                    .outlet_change.mean()
                )
                response["shift"] = response["shift"].map(lambda shift: f"{shift:+.1f} °C")
                st.plotly_chart(
                    dark_layout(
                        px.bar(
                            response,
                            x="shift",
                            y="outlet_change",
                            color="variant",
                            barmode="group",
                            labels={
                                "shift": "Supply shift (°C)",
                                "variant": "Simulator variant",
                                "outlet_change": "Mean absolute outlet change (°C)",
                            },
                        )
                    )
                )
                st.caption(
                    "Each bar averages the absolute predicted outlet response across evaluated scenarios. Small changes indicate limited simulated cooling leverage."
                )
                st.caption(
                    "Single-state spikes from a random forest are steps in its predictions and should not be read as physical responses."
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
                            colorscale=TEMPERATURE_RAMP,
                        )
                    )
                ),
                width="stretch",
            )
        with card(
            right, "Feature importance", "Relative contribution to forest split impurity reduction."
        ):
            importance = context.diagnostics.feature_importances.copy()
            if importance.empty:
                st.caption("Forest impurity importance is unavailable for this boosting model.")
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
                low, median, high = filtered["Outlet Avg Temp"].quantile([0, 0.5, 1])
                st.progress(
                    float((median - low) / (high - low)) if high > low else 0.0,
                    text=f"Outlet range {low:.1f}–{high:.1f} °C · median {median:.1f} °C",
                )
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
