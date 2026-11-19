"""Small, inspectable Q-Learning and SARSA implementations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial

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
        tie_margin: float = .05,
        initial_value: float = 0.,
    ) -> None:
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            partial(np.full,3,initial_value,dtype=float)
        )
        self.visits=defaultdict(partial(np.zeros,3,dtype=int))
        self.tie_margin=tie_margin
        self.last_decision=None
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        state = self.discretizer.transform(observation)
        if not explore:
            counts=self.visits.get(state)
            if counts is None or not counts.any():
                self.last_decision="Unseen state · no visited actions → maintain"
                return 1
            eligible=np.flatnonzero(counts)
            values=self.q_values[state]
            ordered=eligible[np.argsort(-values[eligible],kind="stable")]
            best=int(ordered[0])
            gap=values[best]-values[ordered[1]] if len(ordered)>1 else None
            near=gap is not None and (gap==0 or gap<self.tie_margin)
            if near and counts[1] and (values[best]==values[1] or values[best]-values[1]<self.tie_margin):
                best=1
            scores=" / ".join(f"{value:.2f}" if count else "unvisited" for value,count in zip(values,counts,strict=True))
            margin=f"{gap:.3f}" if gap is not None else "n/a"
            self.last_decision=f"Q (decrease / maintain / increase): {scores} · gap {margin}" + (" · near tie" if near else "") + f" → {('decrease','maintain','increase')[best]}"
            return best
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))
        return int(np.argmax(self.q_values[state]))

    def explain(self, observation):
        return self.last_decision

    def update_q_learning(
        self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool
    ) -> None:
        state = self.discretizer.transform(observation)
        next_state = self.discretizer.transform(next_observation)
        self.visits[state][action] += 1
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
        self.visits[state][action] += 1
        target = reward if done else reward + self.gamma * self.q_values[next_state][next_action]
        self.q_values[state][action] += self.alpha * (target - self.q_values[state][action])

