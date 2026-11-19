import numpy as np

from data_centre_rl.tabular import Discretizer, TabularAgent


def test_greedy_eligibility_and_margin():
    obs = np.zeros(5)
    agent = TabularAgent(Discretizer().fit(np.stack([obs, obs + 1])))
    state = agent.discretizer.transform(obs)
    assert agent.select_action(obs, False) == 1
    agent.visits[state][:] = [1, 1, 0]
    agent.q_values[state][:] = [-2, -2.03, 0]
    assert agent.select_action(obs, False) == 1
    assert "near tie" in agent.explain(obs)
    agent.q_values[state][0] = -1
    assert agent.select_action(obs, False) == 0
