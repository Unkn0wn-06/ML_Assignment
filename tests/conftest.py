import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rows = []
    scenarios = [("BaselineTest_25percent", "Baseline", 0.25), ("WallTest_100percent", "Wall", 1.0)]
    for scenario, configuration, load in scenarios:
        for time in range(8):
            rows.append({
                "time": time * 2,
                "Inlet Temp Avg": 20.0 + time * 0.1,
                "Outlet Avg Temp": 24.0 + time * 0.05,
                "Cooling Supply Average": 18.0 + time * 0.01,
                "Cooling Return Average": 23.0 + time * 0.05,
                "P_IT": 0.8 + load * 0.1,
                "P_DC": 1.5 + load,
                "PUE": 1.8,
                "CoP": 2.0,
                "configuration": configuration,
                "load_fraction": load,
                "scenario": scenario,
            })
    return pd.DataFrame(rows)

