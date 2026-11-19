"""Supervised transition model used by the experimental RL environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from .data import STATE_COLUMNS, TARGET_COLUMNS

VARIANTS = ("Random forest (levels)", "Random forest (deltas)", "Monotone boosting (assumption)")
TEMPERATURES = ["Outlet Avg Temp", "Cooling Return Average"]
FEATURE_COLUMNS = [*STATE_COLUMNS, "configuration", "load_fraction"]


@dataclass
class TransitionModel:
    """Predict the next thermal/power outcomes for a proposed cooling setpoint."""

    random_state: int = 42
    n_estimators: int = 200
    variant: str = VARIANTS[0]

    def __post_init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.target_columns = [f"next_{column}" for column in TARGET_COLUMNS]

    def fit(self, transition_table: pd.DataFrame) -> TransitionModel:
        missing = set(FEATURE_COLUMNS + self.target_columns).difference(transition_table.columns)
        if missing:
            raise ValueError(f"Transition table lacks columns: {sorted(missing)}")
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    "passthrough",
                    [column for column in FEATURE_COLUMNS if column != "configuration"],
                ),
                (
                    "configuration",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["configuration"],
                ),
            ]
        )
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.pipeline = Pipeline([("features", preprocessor), ("model", estimator)])
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown simulator variant: {self.variant}")
        inputs = transition_table[FEATURE_COLUMNS]
        targets = transition_table[self.target_columns].copy()
        if self.variant == VARIANTS[1]:
            for column in TEMPERATURES:
                targets[f"next_{column}"] -= inputs[column]
        if self.variant == VARIANTS[2]:
            transformed = preprocessor.fit_transform(inputs)
            self.boosters = []
            for column in TARGET_COLUMNS:
                constraints = np.zeros(transformed.shape[1], dtype=int)
                constraints[STATE_COLUMNS.index("Cooling Supply Average")] = int(
                    column in TEMPERATURES
                )
                booster = HistGradientBoostingRegressor(
                    max_iter=self.n_estimators,
                    monotonic_cst=constraints,
                    random_state=self.random_state,
                    early_stopping=False,
                )
                with threadpool_limits(limits=1):
                    booster.fit(transformed, targets[f"next_{column}"])
                self.boosters.append(booster)
        else:
            self.pipeline.fit(inputs, targets)
        self._prepare_rollout_predictor()
        return self

    def predict(self, state: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Fit the transition model before predicting")
        if self.variant == VARIANTS[2]:
            encoded = self.pipeline.named_steps["features"].transform(state[FEATURE_COLUMNS])
            with threadpool_limits(limits=1):
                prediction = np.column_stack([model.predict(encoded) for model in self.boosters])
        else:
            prediction = self.pipeline.predict(state[FEATURE_COLUMNS])
        if self.variant == VARIANTS[1]:
            prediction[:, :2] += state[TEMPERATURES].to_numpy()
        return pd.DataFrame(prediction, columns=TARGET_COLUMNS, index=state.index)

    def score(self, transition_table: pd.DataFrame) -> dict[str, float]:
        prediction = self.predict(transition_table)
        return {
            column: float(
                mean_absolute_error(transition_table[f"next_{column}"], prediction[column])
            )
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
                                transition_table.loc[indices, f"next_{column}"],
                                prediction.loc[indices, column],
                            )
                        ),
                    }
                )
        return pd.DataFrame(records)

    def predicted_vs_actual(self, transition_table: pd.DataFrame) -> pd.DataFrame:
        """Return tidy prediction records for honest model-diagnostics charts."""
        prediction = self.predict(transition_table)
        records: list[pd.DataFrame] = []
        metadata = transition_table[["scenario", "configuration", "load_fraction"]].reset_index(
            drop=True
        )
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
        if self.variant == VARIANTS[2]:
            return pd.DataFrame(columns=["feature", "importance"])
        features = self.pipeline.named_steps["features"]
        estimator = self.pipeline.named_steps["model"]
        names = features.get_feature_names_out()
        return pd.DataFrame(
            {"feature": names, "importance": estimator.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)

    def _prepare_rollout_predictor(self):
        """Pack fitted numeric trees for single-state traversal across all trees at once."""
        self.categories = (
            self.pipeline.named_steps["features"]
            .named_transformers_["configuration"]
            .categories_[0]
        )
        branches = []
        leaves = []
        offset = 0
        roots = []
        if self.variant == VARIANTS[2]:
            trees = [
                iteration[0].nodes for model in self.boosters for iteration in model._predictors
            ]
        else:
            trees = [tree.tree_ for tree in self.pipeline.named_steps["model"].estimators_]
        for tree in trees:
            if self.variant == VARIANTS[2]:
                leaf = tree["is_leaf"].astype(bool)
                feature, threshold, left, right = (
                    tree[name] for name in ["feature_idx", "num_threshold", "left", "right"]
                )
                values = tree["value"]
            else:
                feature, threshold, left, right = (
                    tree.feature,
                    tree.threshold,
                    tree.children_left,
                    tree.children_right,
                )
                leaf = left == -1
                values = tree.value[:, :, 0]
            nodes = np.arange(len(feature))
            branches.append(
                (
                    np.maximum(feature, 0),
                    threshold,
                    np.where(leaf, nodes, left) + offset,
                    np.where(leaf, nodes, right) + offset,
                    leaf,
                )
            )
            leaves.append(values)
            roots.append(offset)
            offset += len(feature)
        self.features, self.thresholds, self.left, self.right, self.leaf = map(
            np.concatenate, zip(*branches, strict=True)
        )
        self.roots = np.array(roots)
        self.values = np.concatenate(leaves)
        if self.variant == VARIANTS[2]:
            self.base = np.array([model._baseline_prediction.item() for model in self.boosters])

    def predict_one(self, features):
        if not hasattr(self, "roots"):
            return self.predict(pd.DataFrame([features])).iloc[0].to_dict()
        row = np.array(
            [
                *[features[column] for column in STATE_COLUMNS],
                features["load_fraction"],
                *(self.categories == features["configuration"]),
            ],
            dtype=float if self.variant == VARIANTS[2] else np.float32,
        )
        nodes = self.roots
        while not self.leaf[nodes].all():
            nodes = np.where(
                row[self.features[nodes]] <= self.thresholds[nodes],
                self.left[nodes],
                self.right[nodes],
            )
        if self.variant == VARIANTS[2]:
            prediction = self.base + self.values[nodes].reshape(len(TARGET_COLUMNS), -1).sum(axis=1)
        else:
            prediction = self.values[nodes].mean(axis=0)
        if self.variant == VARIANTS[1]:
            prediction[:2] += [features[column] for column in TEMPERATURES]
        return dict(zip(TARGET_COLUMNS, prediction, strict=True))

    @staticmethod
    def observed_supply_bounds(data: pd.DataFrame) -> tuple[float, float]:
        lower = float(np.nanquantile(data["Cooling Supply Average"], 0.01))
        upper = float(np.nanquantile(data["Cooling Supply Average"], 0.99))
        if lower >= upper:
            raise ValueError("Cooling supply bounds are invalid")
        return lower, upper
