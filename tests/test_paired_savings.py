import pandas as pd
import pytest

from ui.charts import paired_savings


def test_only_full_horizon_matched_pairs_contribute():
    rows = []
    for controller, power in [("Rule-based", 2.0), ("SARSA", 1.9)]:
        for episode in range(3):
            rows.append(
                {
                    "scenario": "test",
                    "seed": 42,
                    "start_row": episode,
                    "episode": episode,
                    "controller": controller,
                    "horizon": 5,
                    "steps_survived": 3 if episode == (1 if controller == "SARSA" else 2) else 5,
                    "mean_p_dc": power,
                    "any_violation": controller == "SARSA",
                }
            )
    result = paired_savings(pd.DataFrame(rows)).iloc[0]
    assert result["Pairs"] == 1
    assert result["Saving (kW)"] == pytest.approx(0.1)
    assert result["Saving (%)"] == pytest.approx(5)
    assert result["Any violation difference (pp)"] == 100
