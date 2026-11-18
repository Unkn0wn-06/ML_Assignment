"""Consistent, physical-metric evaluation for all controller types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    """Aggregated episode metrics with optional simulator trajectories."""

    scenario_metrics: pd.DataFrame
    trajectories: pd.DataFrame


def evaluate_policy(
    env,
    action_fn: Callable[[np.ndarray], int],
    *,
    episodes_per_scenario: int = 3,
    seed: int = 42,
    return_trajectories: bool = False,
) -> pd.DataFrame | EvaluationResult:
    """Evaluate a controller from matched seeded initial states per scenario."""
    records: list[dict[str, float | str]] = []
    trajectory_records: list[dict[str, float | int | str | bool]] = []
    for scenario_index, scenario in enumerate(env.scenarios):
        for episode in range(episodes_per_scenario):
            observation, _ = env.reset(seed=seed + scenario_index * 100 + episode, options={"scenario": scenario})
            total_reward, actions, outcomes = 0.0, [], []
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                action = action_fn(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                actions.append(action)
                outcomes.append(info)
                if return_trajectories:
                    trajectory_records.append(
                        {
                            "scenario": scenario,
                            "episode": episode,
                            "step": step,
                            "action": action,
                            "reward": reward,
                            **info,
                        }
                    )
                step += 1
            result = pd.DataFrame(outcomes)
            records.append({
                "scenario": scenario, "episode": episode, "total_reward": total_reward,
                "mean_p_dc": result["P_DC"].mean(), "mean_pue": result["PUE"].mean(),
                "mean_cop": result["CoP"].mean(),
                "mean_outlet_temperature": result["outlet_temperature"].mean(),
                "max_outlet_temperature": result["outlet_temperature"].max(),
                "violation_count": int((result["violation"] > 0).sum()),
                "max_violation": result["violation"].max(),
                "switching_rate": float(np.mean(np.diff(actions) != 0)) if len(actions) > 1 else 0.0,
                "steps": len(actions),
            })
    metrics = pd.DataFrame(records)
    if return_trajectories:
        return EvaluationResult(metrics, pd.DataFrame(trajectory_records))
    return metrics

