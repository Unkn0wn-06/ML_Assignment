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


@dataclass
class SafetyStateEncoder:
    outlet_limit: float
    supply_bounds: tuple[float, float]

    def transform(self, observation, previous=None, last_action=1):
        margin = self.outlet_limit - float(observation[1])
        change = 0.0 if previous is None else float(observation[1]) - float(previous[1])
        low, high = self.supply_bounds
        fraction = (float(observation[2]) - low) / max(high - low, 1e-8)
        return (int(np.searchsorted([0, .5, 1, 2, 4], margin)),
                0 if change < -.1 else 2 if change > .1 else 1,
                0 if fraction < 1/3 else 2 if fraction >= 2/3 else 1, last_action)


class TabularAgent:
    def __init__(
        self,
        discretizer: Discretizer,
        *,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        seed: int = 42,
        tie_margin: float = 0.05,
        min_visits: int = 10,
        initial_value: float = 0.0,
    ) -> None:
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            partial(np.full, 3, initial_value, dtype=float)
        )
        self.visits = defaultdict(partial(np.zeros, 3, dtype=int))
        self.min_visits = min_visits
        self.reset()
        self.tie_margin = tie_margin
        self.last_decision = None
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.previous = None
        self.last_action = 1
        self.fallback = False

    def encode(self, observation):
        if isinstance(self.discretizer, SafetyStateEncoder):
            return self.discretizer.transform(observation, self.previous, self.last_action)
        return self.discretizer.transform(observation)

    def select_action(self, observation, explore=True):
        state = self.encode(observation)
        action = self.choose(state, explore)
        self.previous = observation.copy()
        self.last_action = action
        return action

    def predict(self, observation):
        return self.select_action(observation, explore=False)

    def choose(self, state, explore):
        if not explore:
            counts = self.visits.get(state, np.zeros(3, dtype=int))
            eligible = np.flatnonzero(counts >= self.min_visits)
            self.fallback = not len(eligible)
            trust = ", ".join(f"{n} {'trusted' if n >= self.min_visits else 'not trusted'}" for n in counts)
            if self.fallback:
                self.last_decision = f"Visits: {trust} → maintain (fallback)"
                return 1
            values = self.q_values[state]
            ordered = eligible[np.argsort(-values[eligible], kind="stable")]
            best = int(ordered[0])
            gap = values[best] - values[ordered[1]] if len(ordered) > 1 else None
            threshold = max(self.tie_margin, .02 * abs(values[best]))
            near = gap is not None and gap <= threshold
            if counts[1] >= self.min_visits and values[best] - values[1] <= threshold:
                best = 1
            scores = " / ".join(
                f"{value:.2f}" if count else "unvisited"
                for value, count in zip(values, counts, strict=True)
            )
            margin = f"{gap:.3f}" if gap is not None else "n/a"
            self.last_decision = (
                f"Q (decrease / maintain / increase): {scores} · gap {margin} · visits: {trust}"
                + (" · near tie" if near else "")
                + f" → {('decrease', 'maintain', 'increase')[best]}"
            )
            return best
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))
        return int(np.argmax(self.q_values[state]))

    def explain(self, observation):
        return self.last_decision

    def update_q_learning(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        self.visits[state][action] += 1
        target = reward if done else reward + self.gamma * self.q_values[next_state].max()
        self.q_values[state][action] += self.alpha * (target - self.q_values[state][action])

    def update_sarsa(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_action: int,
        done: bool,
    ) -> None:
        self.visits[state][action] += 1
        target = reward if done else reward + self.gamma * self.q_values[next_state][next_action]
        self.q_values[state][action] += self.alpha * (target - self.q_values[state][action])
