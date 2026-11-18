import numpy as np
import pandas as pd
import pytest

from data_centre_rl.environment import DataCenterCoolingEnv
from data_centre_rl.experiments import action_sensitivity, prepare_experiment


def test_safe_starts_and_infeasible(sample_data):
    data = sample_data.copy()
    data.loc[data.configuration == "Wall", "Outlet Avg Temp"] = 31
    context = prepare_experiment(data, split_name="All scenarios", trees=3, seed=0)
    env = DataCenterCoolingEnv(data, context.model)
    assert env.infeasible == ["WallTest_100percent"]
    assert all(env.reset(seed=seed)[0][1] <= 29 for seed in range(8))
    with pytest.raises(ValueError, match="Infeasible"):
        env.reset(options={"scenario": env.infeasible[0]})
    unrestricted = DataCenterCoolingEnv(data, context.model, safe_starts=False)
    assert unrestricted.reset(options={"scenario": "WallTest_100percent"})[0][1] == 31


def test_sensitivity_uses_predictions(sample_data):
    context = prepare_experiment(sample_data, split_name="All scenarios", trees=3, seed=0)
    result = action_sensitivity(context)
    assert len(result) == 2 * sample_data.scenario.nunique()
    assert set(result["shift"]) == {-0.5, 0.5}
    assert np.isfinite(result[["Outlet Avg Temp", "P_DC"]]).all().all()
    assert (result.applied.abs() <= 0.5).all()


def test_missing_heatmap_and_replay_axes():
    from ui.charts import replay_chart, scenario_heatmap

    values = pd.DataFrame(
        {
            "configuration": ["Baseline", "Baseline"],
            "load_percent": [25, 25],
            "rate": [np.nan, np.nan],
        }
    )
    heat = scenario_heatmap(values, "rate", set())
    assert len(heat.layout.annotations) == 16
    assert all(cell.text == "n/a" for cell in heat.layout.annotations)
    replay = replay_chart(
        pd.DataFrame(
            {
                "step": [0, 1],
                "outlet_temperature": [28, 29],
                "proposed_supply": [22, 22],
                "action": [1, 1],
            }
        ),
        29,
    )
    assert [trace.yaxis for trace in replay.data] == ["y", "y2", "y3"]
    assert replay.layout.yaxis3.ticktext == ("Decrease", "Maintain", "Increase")
