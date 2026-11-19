import itertools

from data_centre_rl.baselines import RuleBasedPolicy, ThresholdPolicy


def test_threshold_chatter_and_override():
    temperatures = [23.99, 24.01] * 20
    for policy, expected in [(RuleBasedPolicy(), 0), (ThresholdPolicy(), 39)]:
        actions = [policy.predict([0, temp]) for temp in temperatures]
        assert sum(a != b for a, b in itertools.pairwise(actions)) == expected
    policy = RuleBasedPolicy()
    assert policy.predict([0, 23]) == 0
    assert policy.predict([0, 28.8]) == 2
    policy.reset()
    assert policy.predict([0, 26]) == 1
