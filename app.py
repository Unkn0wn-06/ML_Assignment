"""Interactive dashboard for AAU cooling-controller experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_centre_rl.baselines import RuleBasedPolicy
from data_centre_rl.data import load_csv_directory, make_transition_table
from data_centre_rl.environment import DataCenterCoolingEnv
from data_centre_rl.evaluation import evaluate_policy
from data_centre_rl.training import model_action, train_dqn, train_ppo, train_tabular
from data_centre_rl.transition_model import TransitionModel

st.set_page_config(page_title="Data Centre Cooling RL", page_icon="❄️", layout="wide")
st.title("Data Centre Cooling RL")
st.caption("AAU data-driven simulator for academic evaluation — not for live cooling control.")
st.warning(
    "The AAU files do not identify actual cooling-control actions. Results describe a learned "
    "simulation and must be validated before making any engineering decision.",
    icon="⚠️",
)


@st.cache_data(show_spinner=False)
def load_data(data_directory: str) -> pd.DataFrame:
    return load_csv_directory(data_directory)


def run_experiment(
    data: pd.DataFrame,
    algorithm: str,
    trees: int,
    horizon: int,
    repetitions: int,
    episodes: int,
    timesteps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    transitions = make_transition_table(data)
    transition_model = TransitionModel(random_state=seed, n_estimators=trees).fit(transitions)
    environment = DataCenterCoolingEnv(data, transition_model, episode_horizon=horizon)
    if algorithm == "Rule-based baseline":
        action_fn = RuleBasedPolicy().predict
    elif algorithm in {"Q-Learning", "SARSA"}:
        method = "q_learning" if algorithm == "Q-Learning" else "sarsa"
        agent = train_tabular(
            environment, data, algorithm=method, episodes=episodes, bins=5, seed=seed
        )
        action_fn = lambda observation: agent.select_action(observation, explore=False)
    else:
        trainer = train_dqn if algorithm == "DQN" else train_ppo
        agent = trainer(environment, total_timesteps=timesteps, seed=seed)
        action_fn = lambda observation: model_action(agent, observation)
    results = evaluate_policy(
        environment, action_fn, episodes_per_scenario=repetitions, seed=seed
    )
    scenario_metadata = data.groupby("scenario", as_index=False).agg(
        configuration=("configuration", "first"), load_percent=("load_fraction", "first")
    )
    scenario_metadata["load_percent"] *= 100
    return results.merge(scenario_metadata, on="scenario", how="left"), transition_model.score(transitions)


with st.sidebar:
    st.header("Experiment")
    data_directory = st.text_input("AAU data folder", "data/raw/AAU")
    algorithm = st.selectbox(
        "Controller", ["Rule-based baseline", "Q-Learning", "SARSA", "DQN", "PPO"]
    )
    trees = st.slider("Transition-model trees", min_value=20, max_value=200, value=50, step=10)
    horizon = st.slider("Episode horizon", min_value=5, max_value=200, value=50, step=5)
    repetitions = st.slider("Episodes per scenario", min_value=1, max_value=10, value=1)
    episodes = st.slider("Tabular-RL episodes", min_value=25, max_value=1_000, value=250, step=25)
    timesteps = st.slider("Deep-RL timesteps", min_value=1_000, max_value=100_000, value=20_000, step=1_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    run = st.button("Run experiment", type="primary", use_container_width=True)

if run:
    data_path = Path(data_directory)
    if not data_path.is_dir():
        st.error("AAU data folder not found. Run `python scripts/prepare_data.py --archive ..\\sources\\AAU.tar.xz` first.")
    else:
        try:
            with st.spinner("Loading AAU scenarios, fitting the transition model, and evaluating the controller..."):
                data = load_data(str(data_path))
                results, transition_mae = run_experiment(
                    data, algorithm, trees, horizon, repetitions, episodes, timesteps, int(seed)
                )
            st.session_state["results"] = results
            st.session_state["transition_mae"] = transition_mae
            st.session_state["algorithm"] = algorithm
        except ImportError as error:
            st.error(f"{error}. Install the optional deep-RL package: `python -m pip install -e \".[deep-rl]\"`.")
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))

if "results" not in st.session_state:
    st.info("Choose settings in the sidebar and select **Run experiment** to view the results.")
    st.stop()

results = st.session_state["results"].copy()
summary = results.groupby(["scenario", "configuration", "load_percent"], as_index=False).mean(numeric_only=True)
summary["violation_rate"] = summary["violation_count"] / summary["steps"].clip(lower=1)

st.subheader(f"{st.session_state['algorithm']} results")
metric_one, metric_two, metric_three, metric_four = st.columns(4)
metric_one.metric("Mean data-centre power", f"{summary['mean_p_dc'].mean():.3f}")
metric_two.metric("Mean PUE", f"{summary['mean_pue'].mean():.3f}")
metric_three.metric("Mean outlet temperature", f"{summary['mean_outlet_temperature'].mean():.2f} °C")
metric_four.metric("Safety-violation rate", f"{summary['violation_rate'].mean():.1%}")

results_tab, diagnostics_tab, source_tab = st.tabs(["Scenario results", "Model diagnostics", "Source data"])
with results_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Mean data-centre power by scenario")
        st.bar_chart(summary.set_index("scenario")["mean_p_dc"], color="#1976D2")
        st.markdown("#### Mean outlet temperature by scenario")
        st.bar_chart(summary.set_index("scenario")["mean_outlet_temperature"], color="#E65100")
    with right:
        st.markdown("#### PUE by scenario")
        st.bar_chart(summary.set_index("scenario")["mean_pue"], color="#6A1B9A")
        st.markdown("#### Safety-violation rate by scenario")
        st.bar_chart(summary.set_index("scenario")["violation_rate"], color="#C62828")
    st.markdown("#### Scenario comparison")
    visible_columns = [
        "scenario", "configuration", "load_percent", "mean_p_dc", "mean_pue", "mean_cop",
        "mean_outlet_temperature", "max_outlet_temperature", "violation_count",
        "max_violation", "switching_rate", "total_reward",
    ]
    st.dataframe(summary[visible_columns].sort_values(["configuration", "load_percent"]), use_container_width=True)
    st.download_button(
        "Download scenario results (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="cooling_controller_results.csv",
        mime="text/csv",
    )

with diagnostics_tab:
    st.markdown("#### In-sample transition-model mean absolute error")
    mae = pd.Series(st.session_state["transition_mae"], name="MAE")
    st.bar_chart(mae, color="#00897B")
    st.caption(
        "These are in-sample diagnostics only. Use the held-out scenario designs in PROJECT_GUIDANCE.md "
        "for report-quality generalisation results."
    )

with source_tab:
    data = load_data(data_directory)
    st.write(f"Loaded {len(data):,} rows across {data['scenario'].nunique()} AAU scenarios.")
    st.dataframe(data.head(100), use_container_width=True)

