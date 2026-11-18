"""Supervised transition model used by the experimental RL environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import STATE_COLUMNS, TARGET_COLUMNS

FEATURE_COLUMNS = [*STATE_COLUMNS, "configuration", "load_fraction"]


@dataclass
class TransitionModel:
    """Predict the next thermal/power outcomes for a proposed cooling setpoint."""

    random_state: int = 42
    n_estimators: int = 200

    def __post_init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.target_columns = [f"next_{column}" for column in TARGET_COLUMNS]

    def fit(self, transition_table: pd.DataFrame) -> TransitionModel:
        missing = set(FEATURE_COLUMNS + self.target_columns).difference(transition_table.columns)
        if missing:
            raise ValueError(f"Transition table lacks columns: {sorted(missing)}")
        preprocessor = ColumnTransformer(
            [
                ("numeric", "passthrough", [column for column in FEATURE_COLUMNS if column != "configuration"]),
                ("configuration", OneHotEncoder(handle_unknown="ignore"), ["configuration"]),
            ]
        )
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.pipeline = Pipeline([( "features", preprocessor), ("model", estimator)])
        self.pipeline.fit(transition_table[FEATURE_COLUMNS], transition_table[self.target_columns])
        return self

    def predict(self, state: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Fit the transition model before predicting")
        prediction = self.pipeline.predict(state[FEATURE_COLUMNS])
        return pd.DataFrame(prediction, columns=TARGET_COLUMNS, index=state.index)

    def score(self, transition_table: pd.DataFrame) -> dict[str, float]:
        prediction = self.predict(transition_table)
        return {
            column: float(mean_absolute_error(transition_table[f"next_{column}"], prediction[column]))
            for column in TARGET_COLUMNS
        }

    def score_by_scenario(self, transition_table: pd.DataFrame) -> pd.DataFrame:
        """Return MAE per target and scenario for diagnostics."""
        prediction = self.predict(transition_table)
        records: list[dict[str, float | str]] = []
        for scenario, indices in transition_table.groupby("scenario").groups.items():
            for column in TARGET_COLUMNS:
                records.append(
                    {
                        "scenario": scenario,
                        "target": column,
                        "mae": float(
                            mean_absolute_error(
                                transition_table.loc[indices, f"next_{column}"], prediction.loc[indices, column]
                            )
                        ),
                    }
                )
        return pd.DataFrame(records)

    def predicted_vs_actual(self, transition_table: pd.DataFrame) -> pd.DataFrame:
        """Return tidy prediction records for honest model-diagnostics charts."""
        prediction = self.predict(transition_table)
        records: list[pd.DataFrame] = []
        metadata = transition_table[["scenario", "configuration", "load_fraction"]].reset_index(drop=True)
        for column in TARGET_COLUMNS:
            target = metadata.copy()
            target["target"] = column
            target["actual"] = transition_table[f"next_{column}"].to_numpy()
            target["predicted"] = prediction[column].to_numpy()
            records.append(target)
        return pd.concat(records, ignore_index=True)

    def feature_importances(self) -> pd.DataFrame:
        """Expose random-forest feature importances with readable feature labels."""
        if self.pipeline is None:
            raise RuntimeError("Fit the transition model before reading importances")
        features = self.pipeline.named_steps["features"]
        estimator = self.pipeline.named_steps["model"]
        names = features.get_feature_names_out()
        return pd.DataFrame({"feature": names, "importance": estimator.feature_importances_}).sort_values(
            "importance", ascending=False, ignore_index=True
        )

    def predict_one(self, features: dict[str, float | str]) -> dict[str, float]:
        frame = pd.DataFrame([features])
        return self.predict(frame).iloc[0].to_dict()

    @staticmethod
    def observed_supply_bounds(data: pd.DataFrame) -> tuple[float, float]:
        lower = float(np.nanquantile(data["Cooling Supply Average"], 0.01))
        upper = float(np.nanquantile(data["Cooling Supply Average"], 0.99))
        if lower >= upper:
            raise ValueError("Cooling supply bounds are invalid")
        return lower, upper

