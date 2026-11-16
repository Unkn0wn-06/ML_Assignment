"""Small, inspectable Q-Learning and SARSA implementations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Discretizer:
    """Equal-width discretiser fitted on training observations only."""

    bins: int = 5

    def fit(self, observations: np.ndarray) -> Discretizer:
        array = np.asarray(observations, dtype=float)
        self.lower = array.min(axis=0)
        self.upper = array.max(axis=0)
        self.width = np.maximum((self.upper - self.lower) / self.bins, 1e-8)
        return self

    def transform(self, observation: np.ndarray) -> tuple[int, ...]:
        indices = np.floor((np.asarray(observation) - self.lower) / self.width).astype(int)
        return tuple(np.clip(indices, 0, self.bins - 1).tolist())


class TabularAgent:
    def __init__(
        self,
        discretizer: Discretizer,
        *,
        alpha: float = 0.15,
        gamma: float = 0.98,
        epsilon: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(3, dtype=float)
        )
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        state = self.discretizer.transform(observation)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))
        return int(np.argmax(self.q_values[state]))

    def update_q_learning(
        self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool
    ) -> None:
        state = self.discretizer.transform(observation)
        next_state = self.discretizer.transform(next_observation)
        target = reward if done else reward + self.gamma * self.q_values[next_state].max()
        self.q_values[state][action] += self.alpha * (target - self.q_values[state][action])

    def update_sarsa(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_action: int,
        done: bool,
    ) -> None:
        state = self.discretizer.transform(observation)
        next_state = self.discretizer.transform(next_observation)
        target = reward if done else reward + self.gamma * self.q_values[next_state][next_action]
        self.q_values[state][action] += self.alpha * (target - self.q_values[state][action])

