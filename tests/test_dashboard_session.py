from pathlib import Path

import pytest


def test_real_widgets_with_synthetic_scenarios(sample_data, tmp_path):
    app_test = pytest.importorskip("streamlit.testing.v1").AppTest
    root = Path(__file__).resolve().parents[1]
    for scenario, frame in sample_data.groupby("scenario"):
        frame = frame.copy()
        if scenario.startswith("Wall"):
            frame["Outlet Avg Temp"] = 31
        frame.to_csv(tmp_path / f"{scenario}.csv", index=False)
    app = app_test.from_file(str(root / "app.py"), default_timeout=60).run()
    app.sidebar.text_input[0].set_value(str(tmp_path))
    app.sidebar.multiselect[0].set_value(["Rule-based"])
    app.sidebar.button[0].click().run()
    assert not app.exception and not app.error
    assert app.session_state["infeasible"] == ["WallTest_100percent"]
    assert len(app.tabs) == 4
    app.sidebar.button[1].click().run()
    assert "controller_results" not in app.session_state
