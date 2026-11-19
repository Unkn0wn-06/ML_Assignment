from data_centre_rl.environment import DataCenterCoolingEnv


class ConstantModel:
    def __init__(self, outlet):
        self.outlet = outlet

    def predict_one(self, features):
        return {"Outlet Avg Temp": self.outlet, "Cooling Return Average": 24, "P_DC": 2, "PUE": 2}


def test_termination_cost_and_legacy(sample_data):
    data = sample_data.assign(P_DC=2, PUE=2)

    def rollout(outlet, legacy=False):
        env = DataCenterCoolingEnv(
            data,
            ConstantModel(outlet),
            supply_bounds=(16, 23),
            episode_horizon=5,
            legacy_termination=legacy,
        )
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


def test_best_early_return_is_below_worst_safe_return(sample_data):
    from itertools import product

    class ControlledModel:
        def predict_one(self, row):
            unsafe = row["Cooling Supply Average"] >= 18.99
            return {
                "Outlet Avg Temp": 31 if unsafe else 28,
                "Cooling Return Average": 24,
                "P_DC": 1 if unsafe else 2,
                "PUE": 2,
            }

    data = sample_data.assign(P_DC=2, PUE=2)
    data["Cooling Supply Average"] = 18
    env = DataCenterCoolingEnv(data, ControlledModel(), episode_horizon=4, supply_bounds=(16, 22))
    early, full = [], []
    for actions in product(range(3), repeat=4):
        env.reset(seed=42)
        total = 0
        for action in actions:
            _, reward, done, truncated, _ = env.step(action)
            total += reward
            if done or truncated:
                if done and not truncated:
                    early.append(total)
                elif not done:
                    full.append(total)
                break
    assert max(early) < min(full)
