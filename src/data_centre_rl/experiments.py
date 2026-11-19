"""Leak-aware experiment orchestration shared by the dashboard and future scripts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from .baselines import FixedPolicy, RuleBasedPolicy
from .data import TARGET_COLUMNS, make_transition_table, scenario_split
from .environment import DataCenterCoolingEnv
from .evaluation import EvaluationResult, evaluate_policy
from .training import model_action, train_dqn, train_ppo, train_tabular
from .transition_model import VARIANTS, TransitionModel

CONTROLLERS = ("Fixed setpoint", "Rule-based", "Q-Learning", "SARSA", "DQN", "PPO")


@dataclass
class ModelDiagnostics:
    """Training and held-out transition-model diagnostics."""

    in_sample_mae: dict[str, float]
    held_out_mae: dict[str, float]
    in_sample_by_scenario: pd.DataFrame
    held_out_by_scenario: pd.DataFrame
    feature_importances: pd.DataFrame
    evaluation_predictions: pd.DataFrame


@dataclass
class ExperimentContext:
    """Fitted transition model and data partitions for one split definition."""

    split_name: str
    training_data: pd.DataFrame
    evaluation_data: pd.DataFrame
    held_out_scenarios: set[str]
    model: TransitionModel
    diagnostics: ModelDiagnostics
    supply_bounds: tuple[float, float]


@dataclass
class ControllerResult:
    """A controller's real simulator results, learning curve, and run metadata."""

    controller: str
    scenario_metrics: pd.DataFrame
    trajectories: pd.DataFrame
    learning_curve: pd.DataFrame
    training_seconds: float
    proposed_supply_out_of_range_steps: int
    agent: object | None = None


def make_split(data: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Return training/evaluation data for supported whole-scenario splits."""
    if split_name == "All scenarios":
        return data.copy(), data.copy(), set()
    if split_name == "Load generalisation: hold out 100%":
        held_out = set(data.loc[data["load_fraction"] == 1.0, "scenario"])
        training, evaluation = scenario_split(data, held_out)
        return training, evaluation, held_out
    if split_name.startswith("Hold out configuration: "):
        configuration = split_name.removeprefix("Hold out configuration: ")
        held_out = set(data.loc[data["configuration"] == configuration, "scenario"])
        training, evaluation = scenario_split(data, held_out)
        return training, evaluation, held_out
    raise ValueError(f"Unsupported split: {split_name}")


def prepare_experiment(
    data: pd.DataFrame,
    *,
    split_name: str,
    variant: str = VARIANTS[0],
    trees: int,
    seed: int,
) -> ExperimentContext:
    """Fit a transition model exclusively on training scenarios for one split."""
    training_data, evaluation_data, held_out = make_split(data, split_name)
    train_transitions = make_transition_table(training_data)
    evaluation_transitions = make_transition_table(evaluation_data)
    model = TransitionModel(random_state=seed, n_estimators=trees, variant=variant).fit(
        train_transitions
    )
    in_sample = model.score(train_transitions)
    evaluation_mae = model.score(evaluation_transitions)
    diagnostics = ModelDiagnostics(
        in_sample_mae=in_sample,
        held_out_mae=evaluation_mae,
        in_sample_by_scenario=model.score_by_scenario(train_transitions),
        held_out_by_scenario=model.score_by_scenario(evaluation_transitions),
        feature_importances=model.feature_importances(),
        evaluation_predictions=model.predicted_vs_actual(evaluation_transitions),
    )
    return ExperimentContext(
        split_name=split_name,
        training_data=training_data,
        evaluation_data=evaluation_data,
        held_out_scenarios=held_out,
        model=model,
        diagnostics=diagnostics,
        supply_bounds=model.observed_supply_bounds(training_data),
    )


def _environment(
    context: ExperimentContext,
    *,
    outlet_limit: float,
    horizon: int,
    training: bool,
    safe_starts: bool,
):
    data = context.training_data if training else context.evaluation_data
    return DataCenterCoolingEnv(
        data,
        context.model,
        outlet_limit=outlet_limit,
        episode_horizon=horizon,
        safe_starts=safe_starts,
        supply_bounds=context.supply_bounds,
    )


def run_controller(
    context: ExperimentContext,
    *,
    controller: str,
    outlet_limit: float,
    horizon: int,
    episodes_per_scenario: int,
    tabular_episodes: int,
    deep_timesteps: int,
    seed: int,
    safe_starts: bool = True,
    tie_margin: float = 0.05,
    zero_initialisation: bool = False,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> ControllerResult:
    """Train (where needed) and evaluate one controller under a prepared split."""
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller: {controller}")
    train_env = _environment(
        context, outlet_limit=outlet_limit, horizon=horizon, training=True, safe_starts=safe_starts
    )
    if not train_env.scenarios and controller not in {"Fixed setpoint", "Rule-based"}:
        raise ValueError(f"No feasible training scenarios at {outlet_limit:g} °C.")
    started = perf_counter()
    learning_curve = pd.DataFrame(columns=["step", "reward"])
    agent = None
    if controller == "Fixed setpoint":
        agent = FixedPolicy()
        action_fn = agent.predict
    elif controller == "Rule-based":
        agent = RuleBasedPolicy(
            warning_temperature=outlet_limit - 1, lower_temperature=min(24, outlet_limit - 3)
        )
        action_fn = agent.predict
    elif controller in {"Q-Learning", "SARSA"}:
        algorithm = "q_learning" if controller == "Q-Learning" else "sarsa"
        agent = train_tabular(
            train_env,
            context.training_data,
            algorithm=algorithm,
            episodes=tabular_episodes,
            bins=5,
            tie_margin=tie_margin,
            zero_initialisation=zero_initialisation,
            seed=seed,
            progress_callback=progress_callback,
        )
        learning_curve = agent.learning_curve
        action_fn = lambda observation: agent.select_action(observation, explore=False)
    else:
        trainer = train_dqn if controller == "DQN" else train_ppo
        agent = trainer(
            train_env,
            total_timesteps=deep_timesteps,
            seed=seed,
            progress_callback=progress_callback,
        )
        learning_curve = agent.learning_curve
        action_fn = lambda observation: model_action(agent, observation)
    training_seconds = perf_counter() - started
    evaluation = evaluate_policy(
        _environment(
            context,
            outlet_limit=outlet_limit,
            horizon=horizon,
            training=False,
            safe_starts=safe_starts,
        ),
        action_fn,
        episodes_per_scenario=episodes_per_scenario,
        seed=seed,
        return_trajectories=True,
        explain=getattr(agent, "explain", None)
        if controller not in {"DQN", "PPO"}
        else lambda observation: explain_deep(agent, observation),
    )
    assert isinstance(evaluation, EvaluationResult)
    metrics = evaluation.scenario_metrics.copy()
    metrics["seed"] = seed
    metrics["controller"] = controller
    trajectories = evaluation.trajectories.copy()
    trajectories["seed"] = seed
    trajectories["controller"] = controller
    clipped_steps = int(trajectories["supply_clipped"].sum()) if not trajectories.empty else 0
    return ControllerResult(
        controller=controller,
        scenario_metrics=metrics,
        trajectories=trajectories,
        learning_curve=learning_curve.assign(controller=controller, seed=seed),
        training_seconds=training_seconds,
        proposed_supply_out_of_range_steps=clipped_steps,
        agent=agent,
    )


def diagnostics_frame(diagnostics: ModelDiagnostics) -> pd.DataFrame:
    """Format in-sample and evaluation MAE for a compact diagnostics chart."""
    records = []
    for target in TARGET_COLUMNS:
        records.extend(
            [
                {
                    "target": target,
                    "partition": "In-sample",
                    "mae": diagnostics.in_sample_mae[target],
                },
                {
                    "target": target,
                    "partition": "Evaluation",
                    "mae": diagnostics.held_out_mae[target],
                },
            ]
        )
    return pd.DataFrame(records)


def action_sensitivity(context):
    """One-factor changes around each scenario's median state, within training bounds."""
    from .data import STATE_COLUMNS

    groups = ["scenario", "configuration", "load_fraction"]
    typical = context.evaluation_data.groupby(groups, as_index=False)[STATE_COLUMNS].median()
    supply = "Cooling Supply Average"
    typical[supply] = typical[supply].clip(*context.supply_bounds)
    targets = ["Outlet Avg Temp", "P_DC"]
    baseline = context.model.predict(typical)[targets]
    records = []
    for shift in (-0.5, 0.5):
        changed = typical.copy()
        changed[supply] = (changed[supply] + shift).clip(*context.supply_bounds)
        differences = context.model.predict(changed)[targets] - baseline
        records.append(
            differences.assign(
                scenario=typical.scenario, shift=shift, applied=changed[supply] - typical[supply]
            )
        )
    return pd.concat(records, ignore_index=True)


def explain_deep(model, observation):
    import torch

    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(observation)
        if hasattr(model, "q_net"):
            values = model.q_net(tensor)
            label = "Q"
        else:
            values = model.policy.get_distribution(tensor).distribution.probs
            label = "Probability"
        return (
            label
            + " (decrease / maintain / increase): "
            + " / ".join(f"{value:.2f}" for value in values.flatten().tolist())
        )
