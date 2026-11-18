"""Training helpers for tabular and Stable-Baselines3 agents."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
import pandas as pd

from .tabular import Discretizer, TabularAgent


def fit_discretizer(data, bins: int = 5) -> Discretizer:
    columns = ["Inlet Temp Avg", "Outlet Avg Temp", "Cooling Supply Average", "Cooling Return Average", "P_IT"]
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
    learning_callback: ProgressCallback | None = None,
) -> TabularAgent:
    """Train Q-Learning or SARSA with linearly decayed epsilon exploration."""
    if algorithm not in {"q_learning", "sarsa"}:
        raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
    agent = TabularAgent(fit_discretizer(training_data, bins), seed=seed)
    learning_records: list[dict[str, float | int]] = []
    for episode in range(episodes):
        agent.epsilon = max(0.05, 1.0 - episode / max(1, episodes - 1) * 0.95)
        observation, _ = env.reset(seed=seed + episode)
        action = agent.select_action(observation)
        terminated = truncated = False
        episode_reward = 0.0
        while not (terminated or truncated):
            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            if algorithm == "q_learning":
                agent.update_q_learning(observation, action, reward, next_observation, done)
                next_action = agent.select_action(next_observation)
            else:
                next_action = agent.select_action(next_observation)
                agent.update_sarsa(observation, action, reward, next_observation, next_action, done)
            observation, action = next_observation, next_action
        learning_records.append({"step": episode + 1, "reward": episode_reward})
        if progress_callback:
            progress_callback(episode + 1, episodes, episode_reward)
        if learning_callback:
            learning_callback(episode + 1, episodes, episode_reward)
    agent.epsilon = 0.0
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
        raise ImportError("Install dependencies with `python -m pip install -e .` to use DQN") from error
    model = DQN("MlpPolicy", env, seed=seed, learning_rate=1e-3, buffer_size=20_000, learning_starts=500, batch_size=64, target_update_interval=250, verbose=0)
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
        raise ImportError("Install dependencies with `python -m pip install -e .` to use PPO") from error
    model = PPO("MlpPolicy", env, seed=seed, learning_rate=3e-4, n_steps=256, batch_size=64, verbose=0)
    callback = _make_sb3_callback(total_timesteps, progress_callback)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.learning_curve = pd.DataFrame(callback.learning_curve)
    return model


def model_action(model, observation: np.ndarray) -> int:
    action, _ = model.predict(observation, deterministic=True)
    return int(action)

