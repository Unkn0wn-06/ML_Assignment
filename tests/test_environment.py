from gymnasium.utils.env_checker import check_env

from data_centre_rl.data import make_transition_table
from data_centre_rl.environment import DataCenterCoolingEnv
from data_centre_rl.transition_model import TransitionModel


def test_environment_steps_with_valid_api(sample_data):
    model = TransitionModel(n_estimators=5).fit(make_transition_table(sample_data))
    environment = DataCenterCoolingEnv(sample_data, model, episode_horizon=3)
    observation, info = environment.reset(seed=7, options={"scenario": "BaselineTest_25percent"})
    assert observation.shape == (5,)
    assert info["scenario"] == "BaselineTest_25percent"
    observation, reward, terminated, truncated, info = environment.step(1)
    assert observation.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "violation" in info


def test_environment_passes_gymnasium_checker(sample_data):
    model = TransitionModel(n_estimators=5).fit(make_transition_table(sample_data))
    environment = DataCenterCoolingEnv(sample_data, model, episode_horizon=3)
    check_env(environment, skip_render_check=True)

