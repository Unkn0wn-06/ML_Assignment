from data_centre_rl.experiments import make_split, prepare_experiment, run_controller


def test_evaluation_can_return_per_step_trajectories(sample_data):
    context = prepare_experiment(sample_data, split_name="All scenarios", trees=5, seed=7)
    result = run_controller(
        context,
        controller="Fixed setpoint",
        outlet_limit=27.0,
        horizon=3,
        episodes_per_scenario=1,
        tabular_episodes=1,
        deep_timesteps=1,
        seed=7,
    )
    assert {"step", "action", "P_DC", "violation", "reward"}.issubset(result.trajectories.columns)
    assert not result.scenario_metrics.empty


def test_load_generalisation_holds_out_only_full_load(sample_data):
    training, evaluation, held_out = make_split(sample_data, "Load generalisation: hold out 100%")
    assert set(evaluation["scenario"]) == {"WallTest_100percent"}
    assert "WallTest_100percent" in held_out
    assert set(training["scenario"]) == {"BaselineTest_25percent"}

