from data_centre_rl.environment import DataCenterCoolingEnv


class ConstantModel:
    def __init__(self, outlet):
        self.outlet = outlet

    def predict_one(self, features):
        return {"Outlet Avg Temp": self.outlet, "Cooling Return Average": 24, "P_DC": 2, "PUE": 2}


def test_termination_cost_and_legacy(sample_data):
    data = sample_data.assign(P_DC=2, PUE=2)

    def rollout(outlet, legacy=False):
        env = DataCenterCoolingEnv(data, ConstantModel(outlet), supply_bounds=(16, 23),
                                   episode_horizon=5, legacy_termination=legacy)
        env.reset(seed=42)
        rewards = []
        for _ in range(5):
            _, reward, done, truncated, info = env.step(1)
            rewards.append(reward)
            if done or truncated:
                break
        return rewards, info

    safe, _ = rollout(29)
    mild, info = rollout(29.1)
    assert len(mild) == 5 and info["temperature_penalty"] > 0
    early, info = rollout(30.6)
    assert len(early) == 1 and info["termination_penalty"] == 3 * 4
    assert sum(early) < sum(safe) == -15
    legacy, info = rollout(29.1, True)
    assert len(legacy) == 1 and info["termination_penalty"] == 0
