"""Training helpers for tabular and Stable-Baselines3 agents."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
import pandas as pd

from .tabular import Discretizer, SafetyStateEncoder, TabularAgent


def fit_discretizer(data, bins: int = 5) -> Discretizer:
    columns = [
        "Inlet Temp Avg",
        "Outlet Avg Temp",
        "Cooling Supply Average",
        "Cooling Return Average",
        "P_IT",
    ]
    return Discretizer(bins=bins).fit(data[columns].to_numpy())


ProgressCallback = Callable[[int, int, float], None]


def train_tabular(
    env,
    training_data,
    *,
    algorithm: str,
    episodes: int,
    bins: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    legacy_state: bool = False,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    decay_fraction: float = 0.6,
    min_visits: int = 10,
    tie_margin: float = 0.05,
    zero_initialisation: bool = False,
    learning_callback: ProgressCallback | None = None,
) -> TabularAgent:
    """Train Q-Learning or SARSA with linearly decayed epsilon exploration."""
    if algorithm not in {"q_learning", "sarsa"}:
        raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
    weights = env.reward_weights
    penalties = (
        weights.energy * training_data.P_DC.clip(lower=0)
        + weights.pue * training_data.PUE.clip(lower=0)
        + weights.temperature
        * (training_data["Outlet Avg Temp"] - env.outlet_limit).clip(lower=0) ** 2
    )
    initial = 0.0 if zero_initialisation else -float(penalties.mean())
    encoder = (
        fit_discretizer(training_data, bins)
        if legacy_state
        else SafetyStateEncoder(env.outlet_limit, env.supply_bounds)
    )
    agent = TabularAgent(
        encoder,
        alpha=alpha,
        gamma=gamma,
        min_visits=min_visits,
        seed=seed,
        tie_margin=tie_margin,
        initial_value=initial,
    )
    learning_records: list[dict[str, float | int]] = []
    for episode in range(episodes):
        agent.reset()
        fraction = min(1.0, episode / max(1, round(episodes * decay_fraction) - 1))
        agent.epsilon = epsilon_start + (epsilon_end - epsilon_start) * fraction
        observation, _ = env.reset(seed=seed + episode)
        state = agent.encode(observation)
        action = agent.select_action(observation)
        terminated = truncated = False
        episode_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            next_observation, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            episode_reward += reward
            done = terminated or truncated
            next_state = agent.encode(next_observation)
            if algorithm == "q_learning":
                agent.update_q_learning(state, action, reward, next_state, done)
                next_action = agent.select_action(next_observation)
            else:
                next_action = agent.select_action(next_observation)
                agent.update_sarsa(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
        episode_reward /= steps
        learning_records.append({"step": episode + 1, "reward": episode_reward})
        if progress_callback:
            progress_callback(episode + 1, episodes, episode_reward)
        if learning_callback:
            learning_callback(episode + 1, episodes, episode_reward)
    agent.epsilon = 0.0
    agent.reset()
    agent.learning_curve = pd.DataFrame(learning_records)
    return agent


def _make_sb3_callback(total_timesteps: int, progress_callback: ProgressCallback | None):
    """Create a lightweight callback without importing SB3 until it is requested."""
    from stable_baselines3.common.callbacks import BaseCallback

    class LearningCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.learning_curve: list[dict[str, float | int]] = []
            self.started = perf_counter()

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", [])
            reward = float(np.mean(rewards)) if len(rewards) else 0.0
            self.learning_curve.append({"step": self.num_timesteps, "reward": reward})
            if progress_callback:
                progress_callback(self.num_timesteps, total_timesteps, reward)
            return True

    return LearningCallback()


def train_dqn(
    env,
    *,
    total_timesteps: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
):
    """Train a DQN, importing the optional deep-RL dependency only when used."""
    try:
        from stable_baselines3 import DQN
    except ImportError as error:
        raise ImportError(
            "Install dependencies with `python -m pip install -e .` to use DQN"
        ) from error
    model = DQN(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=1e-3,
        buffer_size=20_000,
        learning_starts=500,
        batch_size=64,
        target_update_interval=250,
        verbose=0,
    )
    callback = _make_sb3_callback(total_timesteps, progress_callback)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.learning_curve = pd.DataFrame(callback.learning_curve)
    return model


def train_ppo(
    env,
    *,
    total_timesteps: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
):
    """Train discrete-action PPO with a small MLP policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise ImportError(
            "Install dependencies with `python -m pip install -e .` to use PPO"
        ) from error
    model = PPO(
        "MlpPolicy", env, seed=seed, learning_rate=3e-4, n_steps=256, batch_size=64, verbose=0
    )
    callback = _make_sb3_callback(total_timesteps, progress_callback)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.learning_curve = pd.DataFrame(callback.learning_curve)
    return model


def model_action(model, observation: np.ndarray) -> int:
    action, _ = model.predict(observation, deterministic=True)
    return int(action)
