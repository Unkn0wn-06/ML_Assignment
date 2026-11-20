"""Gymnasium environment backed by a learned AAU transition model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .data import STATE_COLUMNS
from .transition_model import TransitionModel


@dataclass(frozen=True)
class RewardWeights:
    energy: float = 1.0
    pue: float = 0.5
    temperature: float = 10.0
    switching: float = 0.05


class DataCenterCoolingEnv(gym.Env[np.ndarray, int]):
    """Experimental cooling-control environment with bounded discrete actions.

    Actions are 0=decrease cooling (raise supply setpoint), 1=maintain, and
    2=increase cooling (lower supply setpoint). It is not for real-world control.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}
    action_deltas = np.array([1.0, 0.0, -1.0], dtype=np.float64)

    def __init__(
        self,
        scenario_data: pd.DataFrame,
        transition_model: TransitionModel,
        *,
        safe_starts: bool = True,
        outlet_limit: float = 29.0,
        cooling_step: float = 0.5,
        episode_horizon: int = 100,
        reward_weights: RewardWeights | None = None,
        scenarios: list[str] | None = None,
        termination_excess: float = 1.5,
        termination_penalty_per_step: float | None = None,
        legacy_termination: bool = False,
        supply_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        required = {"scenario", "configuration", "load_fraction", *STATE_COLUMNS}
        missing = required.difference(scenario_data.columns)
        if missing:
            raise ValueError(f"Scenario data lacks columns: {sorted(missing)}")
        self.data = scenario_data.sort_values(["scenario", "time"]).reset_index(drop=True)
        self.transition_model = transition_model
        self.outlet_limit = outlet_limit
        self.cooling_step = cooling_step
        self.episode_horizon = episode_horizon
        self.reward_weights = reward_weights or RewardWeights()
        self.termination_excess = termination_excess
        self.legacy_termination = legacy_termination
        self.termination_penalty_per_step = (
            np.ceil(
                self.reward_weights.energy * self.data.P_DC.max()
                + self.reward_weights.pue * self.data.PUE.max()
            )
            if termination_penalty_per_step is None
            else termination_penalty_per_step
        )
        self.supply_bounds = supply_bounds or transition_model.observed_supply_bounds(self.data)
        available = sorted(self.data["scenario"].unique())
        self.scenarios = scenarios or available
        unknown = set(self.scenarios).difference(available)
        if unknown:
            raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
        self.frames = {
            scenario: self.data.loc[self.data["scenario"] == scenario].reset_index(drop=True)
            for scenario in self.scenarios
        }
        if any(len(frame) < 2 for frame in self.frames.values()):
            raise ValueError("Every scenario needs at least two rows")

        self.records = {name: frame.to_dict("records") for name, frame in self.frames.items()}
        self.start_rows = {}
        for scenario, frame in self.frames.items():
            candidates = np.arange(len(frame) - 1)
            if safe_starts:
                candidates = candidates[
                    frame["Outlet Avg Temp"].to_numpy()[candidates] <= outlet_limit
                ]
            full_length = candidates[candidates <= max(0, len(frame) - episode_horizon - 1)]
            self.start_rows[scenario] = full_length if len(full_length) else candidates
        self.infeasible = [name for name, rows in self.start_rows.items() if not len(rows)]
        self.scenarios = [name for name in self.scenarios if name not in self.infeasible]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.full(len(STATE_COLUMNS), -1_000_000.0, dtype=np.float32),
            high=np.full(len(STATE_COLUMNS), 1_000_000.0, dtype=np.float32),
            dtype=np.float32,
        )
        self._frame: pd.DataFrame | None = None
        self._index = 0
        self._steps = 0
        self._last_action = 1
        self._state: dict[str, float] = {}

    def _observation(self) -> np.ndarray:
        return np.array([self._state[column] for column in STATE_COLUMNS], dtype=np.float32)

    def _info(self, prediction: dict[str, float] | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "scenario": self._rows[self._index]["scenario"],
            "configuration": self._rows[self._index]["configuration"],
            "load_fraction": float(self._rows[self._index]["load_fraction"]),
            "outlet_temperature": float(self._state["Outlet Avg Temp"]),
        }
        if prediction:
            info.update({key: float(value) for key, value in prediction.items()})
        return info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        selected = (options or {}).get("scenario")
        if not self.scenarios or selected in self.infeasible:
            raise ValueError("Infeasible at this limit: no eligible initial state")
        if selected is None:
            selected = self.np_random.choice(self.scenarios).item()
        if selected not in self.frames:
            raise ValueError(f"Unknown scenario: {selected}")
        self._frame = self.frames[selected]
        self._rows = self.records[selected]
        self._index = int(self.np_random.choice(self.start_rows[selected]))
        self._steps = 0
        self._last_action = 1
        row = self._rows[self._index]
        self._state = {column: float(row[column]) for column in STATE_COLUMNS}
        return self._observation(), {**self._info(), "start_row": self._index}

    def step(self, action: int):
        if self._frame is None:
            raise RuntimeError("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        current = self._rows[self._index]
        unclipped_supply = (
            self._state["Cooling Supply Average"] + self.action_deltas[action] * self.cooling_step
        )
        proposed_supply = float(np.clip(unclipped_supply, *self.supply_bounds))
        features = {
            **self._state,
            "Cooling Supply Average": proposed_supply,
            "configuration": current["configuration"],
            "load_fraction": float(current["load_fraction"]),
        }
        prediction = self.transition_model.predict_one(features)
        next_index = min(self._index + 1, len(self._frame) - 1)
        exogenous = self._rows[next_index]
        self._state = {
            "Inlet Temp Avg": float(exogenous["Inlet Temp Avg"]),
            "Outlet Avg Temp": float(prediction["Outlet Avg Temp"]),
            "Cooling Supply Average": proposed_supply,
            "Cooling Return Average": float(prediction["Cooling Return Average"]),
            "P_IT": float(exogenous["P_IT"]),
        }
        violation = max(0.0, self._state["Outlet Avg Temp"] - self.outlet_limit)
        p_dc = max(0.0, float(prediction["P_DC"]))
        pue = max(0.0, float(prediction["PUE"]))
        serious = violation > self.termination_excess
        terminated = bool(violation > 0 if self.legacy_termination else serious)
        remaining = max(0, self.episode_horizon - self._steps - 1)
        components = {
            "termination_penalty": self.termination_penalty_per_step * remaining
            if terminated and not self.legacy_termination
            else 0.0,
            "energy_penalty": self.reward_weights.energy * p_dc,
            "pue_penalty": self.reward_weights.pue * pue,
            "temperature_penalty": self.reward_weights.temperature * violation**2,
            "switching_penalty": self.reward_weights.switching * float(action != self._last_action),
        }
        reward = -sum(components.values())
        self._index = next_index
        self._steps += 1
        self._last_action = action
        truncated = bool(self._steps >= self.episode_horizon or self._index >= len(self._frame) - 1)
        info = self._info(prediction)
        info.update({key: self._state[key] for key in ("Inlet Temp Avg", "P_IT")})
        info.update(components)
        info.update(
            terminated=terminated,
            terminated_serious=bool(terminated and serious),
            supply_lower=self.supply_bounds[0],
            supply_upper=self.supply_bounds[1],
        )
        info["violation"] = violation
        info["proposed_supply"] = proposed_supply
        info["unclipped_supply"] = float(unclipped_supply)
        info["supply_clipped"] = bool(not np.isclose(unclipped_supply, proposed_supply))
        return self._observation(), float(reward), terminated, truncated, info
