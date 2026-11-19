"""Transparent fixed and rule-based controllers for fair comparison."""

from __future__ import annotations

import numpy as np


class FixedPolicy:
    """Always maintain the current cooling setting."""

    def predict(self, observation: np.ndarray) -> int:
        return 1

    def explain(self, observation):
        return "Fixed setpoint → maintain"


class ThresholdPolicy:
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
        condition = [
            f"≤ {self.lower_temperature:g}",
            f"inside {self.lower_temperature:g}–{self.warning_temperature:g}",
            f"≥ {self.warning_temperature:g}",
        ][action]
        return f"{outlet:.2f} °C {condition} °C → {('decrease', 'maintain', 'increase')[action]}"


class RuleBasedPolicy(ThresholdPolicy):
    """Stateful hysteresis with a minimum dwell time and emergency override."""

    def __init__(self, lower=24.0, warning=28.0, release_band=0.5, min_hold_steps=5):
        super().__init__(warning, lower)
        self.release_band = release_band
        self.min_hold_steps = min_hold_steps
        self.reset()

    def reset(self):
        self.mode = 1
        self.held = self.min_hold_steps

    def predict(self, observation):
        outlet = float(observation[1])
        previous = self.mode
        if outlet >= self.warning_temperature + 0.75:
            self.mode = 2
        elif self.held >= self.min_hold_steps:
            if outlet >= self.warning_temperature:
                self.mode = 2
            elif outlet <= self.lower_temperature:
                self.mode = 0
            elif (
                self.mode == 0
                and outlet < self.lower_temperature + self.release_band
                or self.mode == 2
                and outlet > self.warning_temperature - self.release_band
            ):
                pass
            else:
                self.mode = 1
        self.held = self.held + 1 if previous == self.mode else 1
        return self.mode

    def explain(self, observation):
        mode = ("Decrease", "Maintain", "Increase")[self.mode]
        release = (
            self.lower_temperature + self.release_band,
            None,
            self.warning_temperature - self.release_band,
        )[self.mode]
        detail = "" if release is None else f", release at {release:g} °C"
        return f"{mode} mode{detail}, held {self.held}/{self.min_hold_steps} steps"
