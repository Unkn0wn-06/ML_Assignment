import numpy as np

from data_centre_rl.evaluation import evaluate_policy


class TinyEpisode:
    scenarios = ("stress", "short")
    episode_horizon = 5

    def reset(self, seed, options):
        self.scenario = options["scenario"]
        self.steps = 0
        return np.zeros(5), {"start_row": seed}

    def step(self, action):
        self.steps += 1
        violation = self.scenario == "stress" and self.steps == 2
        info = {
            "outlet_temperature": 29 + violation,
            "P_DC": 2,
            "PUE": 2,
            "CoP": 1,
            "violation": int(violation),
        }
        return np.zeros(5), -1, violation, self.scenario == "short", info


def test_episode_endings_and_matched_starts():
    first = evaluate_policy(TinyEpisode(), lambda obs: 1, episodes_per_scenario=1)
    second = evaluate_policy(TinyEpisode(), lambda obs: 0, episodes_per_scenario=1)
    assert list(first.end_reason) == ["violation", "data_end"]
    assert list(first.steps_survived) == [2, 1]
    assert list(first.violated) == [True, False]
    assert first.violated.mean() == 0.5
    assert first.start_row.equals(second.start_row)
