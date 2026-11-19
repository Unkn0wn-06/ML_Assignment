import numpy as np

from data_centre_rl.environment import DataCenterCoolingEnv
from data_centre_rl.tabular import SafetyStateEncoder
from data_centre_rl.training import train_tabular
from data_centre_rl.transition_model import TransitionModel
from data_centre_rl.data import make_transition_table


def test_safety_bins():
    encoder = SafetyStateEncoder(29, (15, 21))
    def obs(margin, supply):
        return np.array([0, 29-margin, supply, 0, 0])
    for index, margin in enumerate([0, .5, 1, 2, 4, 5]):
        assert encoder.transform(obs(margin, 15))[0] == index
    assert encoder.transform(obs(1, 19), obs(2, 18), 2) == (2, 2, 2, 2)
    assert encoder.transform(obs(2, 17), obs(1, 15)) == (3, 0, 1, 1)


def test_learns_best_action(sample_data):
    model = TransitionModel(n_estimators=2).fit(make_transition_table(sample_data))
    env = DataCenterCoolingEnv(sample_data, model, episode_horizon=1)
    original = env.step
    def step(action):
        obs, _, terminated, truncated, info = original(action)
        return obs, (5 if action == 2 else -5), terminated, truncated, info
    env.step = step
    for algorithm in ["q_learning", "sarsa"]:
        agent = train_tabular(env, sample_data, algorithm=algorithm, episodes=300, bins=5, seed=42)
        obs, _ = env.reset(seed=42)
        assert agent.predict(obs) == 2
