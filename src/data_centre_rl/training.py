"""Training helpers for tabular and Stable-Baselines3 agents."""

from __future__ import annotations

import numpy as np

from .tabular import Discretizer, TabularAgent


def fit_discretizer(data, bins: int = 5) -> Discretizer:
    columns = ["Inlet Temp Avg", "Outlet Avg Temp", "Cooling Supply Average", "Cooling Return Average", "P_IT"]
    return Discretizer(bins=bins).fit(data[columns].to_numpy())


def train_tabular(env, training_data, *, algorithm: str, episodes: int, bins: int, seed: int) -> TabularAgent:
    """Train Q-Learning or SARSA with linearly decayed epsilon exploration."""
    if algorithm not in {"q_learning", "sarsa"}:
        raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
    agent = TabularAgent(fit_discretizer(training_data, bins), seed=seed)
    for episode in range(episodes):
        agent.epsilon = max(0.05, 1.0 - episode / max(1, episodes - 1) * 0.95)
        observation, _ = env.reset(seed=seed + episode)
        action = agent.select_action(observation)
        terminated = truncated = False
        while not (terminated or truncated):
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if algorithm == "q_learning":
                agent.update_q_learning(observation, action, reward, next_observation, done)
                next_action = agent.select_action(next_observation)
            else:
                next_action = agent.select_action(next_observation)
                agent.update_sarsa(observation, action, reward, next_observation, next_action, done)
            observation, action = next_observation, next_action
    agent.epsilon = 0.0
    return agent


def train_dqn(env, *, total_timesteps: int, seed: int):
    """Train a DQN, importing the optional deep-RL dependency only when used."""
    try:
        from stable_baselines3 import DQN
    except ImportError as error:
        raise ImportError("Install dependencies with `python -m pip install -e .` to use DQN") from error
    model = DQN("MlpPolicy", env, seed=seed, learning_rate=1e-3, buffer_size=20_000, learning_starts=500, batch_size=64, target_update_interval=250, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


def train_ppo(env, *, total_timesteps: int, seed: int):
    """Train discrete-action PPO with a small MLP policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise ImportError("Install dependencies with `python -m pip install -e .` to use PPO") from error
    model = PPO("MlpPolicy", env, seed=seed, learning_rate=3e-4, n_steps=256, batch_size=64, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


def model_action(model, observation: np.ndarray) -> int:
    action, _ = model.predict(observation, deterministic=True)
    return int(action)

