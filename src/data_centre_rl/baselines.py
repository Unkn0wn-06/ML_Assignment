"""Transparent fixed and rule-based controllers for fair comparison."""

from __future__ import annotations

import numpy as np


class FixedPolicy:
    """Always maintain the current cooling setting."""

    def predict(self, observation: np.ndarray) -> int:
        return 1


    def explain(self, observation):
        return "Fixed setpoint → maintain"


class RuleBasedPolicy:
    """Hysteresis controller using only the observed outlet temperature."""

    def __init__(self, warning_temperature: float = 28.0, lower_temperature: float = 24.0) -> None:
        if lower_temperature >= warning_temperature:
            raise ValueError("lower_temperature must be below warning_temperature")
        self.warning_temperature = warning_temperature
        self.lower_temperature = lower_temperature

    def predict(self, observation: np.ndarray) -> int:
        outlet_temperature = float(observation[1])
        if outlet_temperature >= self.warning_temperature:
            return 2
        if outlet_temperature <= self.lower_temperature:
            return 0
        return 1

    def explain(self, observation):
        outlet = float(observation[1])
        action = self.predict(observation)
        condition = [f"≤ {self.lower_temperature:g}", f"inside {self.lower_temperature:g}–{self.warning_temperature:g}", f"≥ {self.warning_temperature:g}"][action]
        return f"{outlet:.2f} °C {condition} °C → {('decrease','maintain','increase')[action]}"

