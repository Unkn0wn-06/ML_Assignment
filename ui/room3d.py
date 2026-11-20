"""Assemble the offline replay viewer from matched evaluation records."""

import json
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components

from .theme import TEMPERATURE_RAMP

STATIC = Path(__file__).parent / "static"


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=8))


def render_room(replay, comparison, results, selected, raw, outcome, limit):
    source = raw.loc[raw.scenario == outcome.scenario].sort_values("time")
    initial = source.iloc[int(outcome.start_row)]
    start = {
        **json.loads(initial.to_json()),
        "step": 0,
        "outlet_temperature": initial["Outlet Avg Temp"],
        "proposed_supply": initial["Cooling Supply Average"],
        "decision_reason": "Initial state before the first action.",
    }
    frames = [start, *records(replay)]
    baseline = results.get("Rule-based")
    reference = {}
    if baseline is not None and not baseline.trajectories.empty:
        rows = baseline.trajectories
        matched = rows[
            (rows.scenario == outcome.scenario)
            & (rows.seed == outcome.seed)
            & (rows.episode == outcome.episode)
            & (rows.start_row == outcome.start_row)
        ]
        reference = {int(row.step): row.CoP for row in matched.itertuples()}
        reference[0] = start["CoP"]
    summaries = []
    traces = results[outcome.controller].trajectories
    for config, episodes in selected.groupby("configuration"):
        temps = traces.loc[traces.configuration == config, "outlet_temperature"]
        summaries.append(
            {
                "name": config,
                "pue": float(episodes.mean_pue.mean()),
                "n": len(episodes),
                "violated": int(episodes.any_violation.sum()),
                "low": temps.min(),
                "mid": temps.median(),
                "high": temps.max(),
            }
        )
    counts, edges = np.histogram(source["Outlet Avg Temp"].dropna(), bins=16)
    payload = {
        "frames": frames,
        "temperatureRamp": TEMPERATURE_RAMP,
        "comparison": []
        if comparison is None or comparison.empty
        else [start, *records(comparison)],
        "other": None if comparison is None or comparison.empty else comparison.controller.iloc[0],
        "baseline": reference,
        "configs": summaries,
        "histogram": counts.tolist(),
        "edges": edges.tolist(),
        "scenario": outcome.scenario,
        "config": initial.configuration,
        "load": 100 * initial.load_fraction,
        "seed": int(outcome.seed),
        "controller": outcome.controller,
        "limit": limit,
        "warning": limit - 1,
        "lower": min(24, limit - 3),
        "horizon": int(outcome.horizon),
        "reason": outcome.end_reason,
        "switches": int(replay.action.diff().dropna().ne(0).sum()),
        "clipped": int(replay.supply_clipped.sum()),
    }
    html = (STATIC / "viewer.html").read_text(encoding="utf-8")
    for name in ["viewer.css", "three.min.js", "room.js", "loop.js", "viewer.js"]:
        html = html.replace("{{" + name + "}}", (STATIC / name).read_text(encoding="utf-8"))
    html = html.replace("{{payload}}", json.dumps(payload, allow_nan=False).replace("<", r"\u003c"))
    components.html(html, height=860)
