"""AAU data loading and leak-free transition-table construction."""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

import pandas as pd

STATE_COLUMNS = [
    "Inlet Temp Avg",
    "Outlet Avg Temp",
    "Cooling Supply Average",
    "Cooling Return Average",
    "P_IT",
]
TARGET_COLUMNS = ["Outlet Avg Temp", "Cooling Return Average", "P_DC", "PUE", "CoP"]
REQUIRED_COLUMNS = {"time", *STATE_COLUMNS, *TARGET_COLUMNS}
SCENARIO_PATTERN = re.compile(r"(?P<configuration>Baseline|Plate|PlateWall|Wall)Test_(?P<load>\d+)percent")


def parse_scenario_name(filename: str) -> tuple[str, float]:
    """Return configuration and fractional load encoded in an AAU CSV filename."""
    match = SCENARIO_PATTERN.search(Path(filename).stem)
    if not match:
        raise ValueError(f"Unrecognised AAU scenario name: {filename}")
    return match.group("configuration"), int(match.group("load")) / 100.0


def validate_frame(frame: pd.DataFrame, source: str) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{source} has no rows")
    if frame["time"].isna().any() or not frame["time"].is_monotonic_increasing:
        raise ValueError(f"{source} must have a non-null, increasing time column")
    if frame[list(REQUIRED_COLUMNS - {"time"})].isna().any().any():
        raise ValueError(f"{source} contains missing model values")


def attach_metadata(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Validate and label an AAU scenario without modifying the caller's data."""
    configuration, load_fraction = parse_scenario_name(source_name)
    labelled = frame.copy().sort_values("time").reset_index(drop=True)
    validate_frame(labelled, source_name)
    labelled["configuration"] = configuration
    labelled["load_fraction"] = load_fraction
    labelled["scenario"] = Path(source_name).stem
    return labelled


def load_csv_directory(data_dir: str | Path) -> pd.DataFrame:
    """Load all recognised AAU CSV files from an extracted archive directory."""
    paths = sorted(Path(data_dir).glob("*Test_*percent.csv"))
    if not paths:
        raise FileNotFoundError(f"No AAU CSV files found in {data_dir}")
    return pd.concat(
        [attach_metadata(pd.read_csv(path), path.name) for path in paths], ignore_index=True
    )


def load_aau_archive(archive_path: str | Path) -> pd.DataFrame:
    """Read recognised CSVs directly from the supplied .tar.xz archive."""
    frames: list[pd.DataFrame] = []
    with tarfile.open(archive_path, "r:xz") as archive:
        members = [member for member in archive.getmembers() if member.name.endswith(".csv")]
        for member in sorted(members, key=lambda item: item.name):
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            frames.append(attach_metadata(pd.read_csv(extracted), Path(member.name).name))
    if not frames:
        raise ValueError(f"No CSV files found in {archive_path}")
    return pd.concat(frames, ignore_index=True)


def make_transition_table(data: pd.DataFrame) -> pd.DataFrame:
    """Create one-step supervised targets, never crossing scenario boundaries."""
    ordered = data.sort_values(["scenario", "time"]).copy()
    grouped = ordered.groupby("scenario", group_keys=False)
    for column in TARGET_COLUMNS:
        ordered[f"next_{column}"] = grouped[column].shift(-1)
    transition_table = ordered.dropna(subset=[f"next_{column}" for column in TARGET_COLUMNS])
    return transition_table.reset_index(drop=True)


def scenario_split(data: pd.DataFrame, held_out_scenarios: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by whole scenarios to prevent time-series leakage."""
    test_mask = data["scenario"].isin(held_out_scenarios)
    if not test_mask.any() or test_mask.all():
        raise ValueError("Scenario split must leave at least one train and one test scenario")
    return data.loc[~test_mask].copy(), data.loc[test_mask].copy()

