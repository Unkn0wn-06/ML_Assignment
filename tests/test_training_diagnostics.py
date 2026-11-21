import numpy as np

from data_centre_rl.data import make_transition_table
from data_centre_rl.environment import DataCenterCoolingEnv
from data_centre_rl.training import greedy_checkpoint, train_tabular, validation_starts
from data_centre_rl.transition_model import TransitionModel


def test_matched_starts_and_frozen_checkpoints(sample_data):
    model = TransitionModel(n_estimators=3).fit(make_transition_table(sample_data))
    env = DataCenterCoolingEnv(sample_data, model, episode_horizon=3)
    original = env.reset
    seen = []

    def track(**kwargs):
        obs, info = original(**kwargs)
        seen.append((info["scenario"], info["start_row"], tuple(obs), env.episode_horizon))
        return obs, info

    env.reset = track
    agent = train_tabular(env, sample_data, algorithm="sarsa", episodes=3, bins=5, seed=9)
    assert seen[:6:2] == seen[1:6:2]
    before = {k: v.copy() for k, v in agent.q_values.items()}
    visits = {k: v.copy() for k, v in agent.visits.items()}
    greedy_checkpoint(env, validation_starts(env), agent)
    assert before.keys() == agent.q_values.keys() and visits.keys() == agent.visits.keys()
    assert all(np.array_equal(v, agent.q_values[k]) for k, v in before.items())
    assert all(np.array_equal(v, agent.visits[k]) for k, v in visits.items())
