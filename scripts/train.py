"""Train and evaluate a baseline or RL controller from extracted AAU data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from data_centre_rl.baselines import RuleBasedPolicy
from data_centre_rl.data import load_csv_directory, make_transition_table
from data_centre_rl.environment import DataCenterCoolingEnv, RewardWeights
from data_centre_rl.evaluation import evaluate_policy
from data_centre_rl.training import model_action, train_dqn, train_ppo, train_tabular
from data_centre_rl.transition_model import TransitionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/base.json"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--algorithm", choices=["baseline", "q_learning", "sarsa", "dqn", "ppo"], required=True
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--episode-horizon", type=int, default=100)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    training = config["q_learning"]
    data = load_csv_directory(arguments.data_dir)
    transitions = make_transition_table(data)
    model = TransitionModel(random_state=arguments.seed, n_estimators=arguments.trees).fit(
        transitions
    )
    env = DataCenterCoolingEnv(
        data,
        model,
        episode_horizon=arguments.episode_horizon,
        outlet_limit=config["outlet_limit"],
        cooling_step=config["cooling_step"],
        safe_starts=config["safe_starts"],
        reward_weights=RewardWeights(**config["reward_weights"]),
        termination_excess=config.get("termination_excess", 1.5),
        termination_penalty_per_step=config.get("termination_penalty_per_step"),
        legacy_termination=config.get("legacy_termination", False),
    )
    arguments.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (arguments.artifacts_dir / "transition_mae.json").write_text(
        json.dumps(model.score(transitions), indent=2)
    )
    joblib.dump(model, arguments.artifacts_dir / "transition_model.joblib")
    if arguments.algorithm == "baseline":
        action_fn = RuleBasedPolicy(
            lower=min(24, env.outlet_limit - 3), warning=env.outlet_limit - 1
        ).predict
    elif arguments.algorithm in {"q_learning", "sarsa"}:
        agent = train_tabular(
            env,
            data,
            algorithm=arguments.algorithm,
            episodes=arguments.episodes or training.get("episodes", 300),
            bins=training["bins"],
            seed=arguments.seed,
            **{
                key: training[key]
                for key in (
                    "alpha",
                    "gamma",
                    "epsilon_start",
                    "epsilon_end",
                    "min_visits",
                    "decay_fraction",
                    "legacy_state",
                )
                if key in training
            },
        )
        joblib.dump(agent, arguments.artifacts_dir / f"{arguments.algorithm}.joblib")
        action_fn = agent.predict
    else:
        trainer = train_dqn if arguments.algorithm == "dqn" else train_ppo
        agent = trainer(env, total_timesteps=arguments.timesteps, seed=arguments.seed)
        agent.save(str(arguments.artifacts_dir / arguments.algorithm))
        action_fn = lambda observation: model_action(agent, observation)
    results = evaluate_policy(
        env, action_fn, episodes_per_scenario=arguments.evaluation_episodes, seed=arguments.seed
    )
    results["termination_penalty_per_step"] = env.termination_penalty_per_step
    results["legacy_termination"] = env.legacy_termination
    results.to_csv(arguments.artifacts_dir / f"{arguments.algorithm}_evaluation.csv", index=False)
    print(results.groupby("scenario").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
