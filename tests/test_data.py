import pytest

from data_centre_rl.data import make_transition_table, parse_scenario_name, scenario_split


def test_parse_scenario_name():
    assert parse_scenario_name("PlateWallTest_75percent.csv") == ("PlateWall", 0.75)


def test_transition_table_keeps_scenarios_separate(sample_data):
    transitions = make_transition_table(sample_data)
    assert len(transitions) == len(sample_data) - 2
    assert transitions.groupby("scenario").size().eq(7).all()


def test_scenario_split_rejects_empty_partition(sample_data):
    with pytest.raises(ValueError):
        scenario_split(sample_data, set())

