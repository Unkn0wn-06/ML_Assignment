# Energy-Efficient Data Centre Cooling Control Using Reinforcement Learning

## 1. Project aim

Build and evaluate reinforcement-learning (RL) controllers that recommend a cooling setting for a data-centre operating condition. The controller should reduce energy use while keeping outlet temperatures within a defined safe operating range. The study should compare tabular and deep RL methods with transparent non-RL baselines, and should test whether a policy learned from selected AAU scenarios generalises to unseen load levels and physical configurations.

This is a data-driven control study, not a claim that the learned policy can directly operate a real cooling system. The simulator is an approximation trained from historical/simulated AAU observations; recommendations must be bounded by safety constraints and evaluated in that simulator.

## 2. AAU dataset inventory

The supplied archive contains 16 CSV files:

| Physical configuration | Load levels |
| --- | --- |
| `BaselineTest` | 25%, 50%, 75%, 100% |
| `PlateTest` | 25%, 50%, 75%, 100% |
| `PlateWallTest` | 25%, 50%, 75%, 100% |
| `WallTest` | 25%, 50%, 75%, 100% |

Every file has a `time` column and these measurements:

```text
Inlet Temp Avg, Outlet Avg Temp, Cooling Supply Average,
Cooling Return Average, P_IT, P_DC, RCI_HI, RCI_LO, RTI,
RHI, RI, PUE, CoP
```

`P_IT` is useful as a load/state signal. `P_DC`, `PUE`, and `CoP` are primarily outcomes for reward and evaluation. The thermal indices (`RCI_HI`, `RCI_LO`, `RTI`, `RHI`, and `RI`) should be reported as safety/performance indicators rather than handed to the policy as shortcuts.

### Important data limitation

The files record operating measurements, but do not identify a control action, setpoint, fan speed, valve position, or the causal effect of a setting change. The observed `Cooling Supply Average` is therefore a **candidate proxy for the cooling setting**, not proof of the action that caused the next row. The report should state this explicitly.

Before calling the work an RL environment, inspect the time order, time interval, missing values, duplicate rows, ranges, and whether cooling supply values vary independently of load/configuration. If data provenance reveals an action column or simulator inputs, use those instead of the proxy.

## 3. Formulate the control problem

At time step *t*, observe a state `s_t`, choose a cooling action `a_t`, predict the next state `s_(t+1)`, and receive reward `r_t`.

### State

Start with the small, physically motivated state:

```python
state_columns = [
    "Inlet Temp Avg",
    "Outlet Avg Temp",
    "Cooling Supply Average",
    "Cooling Return Average",
    "P_IT",
]
```

For one policy that spans configurations, append encoded configuration and nominal load level:

```python
state = [thermal_measurements..., configuration_one_hot..., load_fraction]
```

Do not use `time` as a physical state unless a justified cyclic feature (such as time of day) exists. Standardise continuous inputs using scalers fitted only on the training split. Keep the configuration label in the scenario metadata too, even if it is encoded in the state.

### Action

Use a small, safe discrete set at first:

```python
DECREASE_COOLING = 0   # raise supply-temperature setpoint by one bounded step
MAINTAIN_COOLING = 1
INCREASE_COOLING = 2   # lower supply-temperature setpoint by one bounded step
```

Define both the sign and size precisely, for example a 0.5 deg C change in cooling-supply setpoint, clipped to an engineering-approved range. A discrete action space supports Q-Learning, SARSA, DQN, and discrete-action PPO consistently. It also makes comparison fair.

The environment action should update the controllable cooling-setting feature, not alter an outcome column such as `P_DC` or `PUE` directly.

### Reward and safety

Use a reward that makes unsafe behaviour unequivocally unattractive:

```python
energy_term = normalise(P_DC)          # lower is better
efficiency_term = normalise(PUE)       # lower is better
temperature_violation = max(0, outlet_temp - outlet_limit)

reward = -(w_energy * energy_term + w_pue * efficiency_term) \
         - w_temp * temperature_violation**2 \
         - w_switch * abs(action_change)
```

Choose `outlet_limit` from the assignment/domain requirement and record it in configuration. Use a high `w_temp`, plus either action masking or early episode termination after a serious violation. Evaluate several documented reward weights rather than selecting one only because it gives the best result.

Report reward components separately. A higher reward alone is not evidence of a safer or more efficient controller.

## 4. Build the data-driven transition model

The Gymnasium environment needs a transition function:

```text
(state, action, configuration, load) -> next thermal state and outcomes
```

Recommended first implementation:

1. Read all CSVs, attach `configuration`, `load_fraction`, `source_file`, and row order.
2. Sort each file by `time`; create next-row targets within the same file only.
3. Train supervised models to predict next thermal state and outcomes (`Outlet Avg Temp`, `Cooling Return Average`, `P_DC`, `PUE`, and optionally `CoP`) from the current state, proposed cooling setting/action, configuration, and load.
4. Use simple, auditable models first (regularised linear regression and random forest/gradient boosting), then select the model by validation error and physical plausibility.
5. Apply actions by converting them to a proposed bounded supply temperature before prediction. Never extrapolate freely far beyond the observed setting range.

Because actions are not logged explicitly, the transition model will have limited causal support. Strengthen it by: limiting actions to observed cooling-supply ranges; retaining configuration/load as inputs; assessing calibration and error by scenario; and labelling the result a **model-based, data-driven simulator**. If the data are insufficient to estimate counterfactual actions, use interpolation between nearby observed cooling-supply values or present the RL results as exploratory.

### Gymnasium environment contract

Implement `DataCenterCoolingEnv(gymnasium.Env)` with:

- `observation_space`: continuous scaled state, with documented bounds.
- `action_space`: `spaces.Discrete(3)` for the initial policy comparison.
- `reset(seed=None, options=None)`: selects a configuration/load episode, samples a valid initial row, and returns `(observation, info)`.
- `step(action)`: applies the bounded action, obtains a predicted transition, computes reward and safety flags, advances time, and returns `(observation, reward, terminated, truncated, info)`.
- `info`: includes unscaled outlet temperature, `P_DC`, `PUE`, `CoP`, violation amount, configuration, load, and each reward component.

Terminate for a hard safety violation if justified; truncate at the episode horizon. Seed NumPy, Python, the environment, and RL libraries. Test environment API compliance with `gymnasium.utils.env_checker.check_env` before training.

## 5. Controllers to compare

### Non-RL baselines

Implement both baselines before RL:

1. **Fixed-setting controller:** maintain the scenario-specific median observed cooling supply temperature, or one globally fixed documented setpoint.
2. **Rule-based controller:** if predicted/measured outlet temperature exceeds a warning threshold, increase cooling; if it is safely below a lower threshold, decrease cooling; otherwise maintain. Use hysteresis to prevent oscillation.

These controllers reveal whether RL adds value beyond a sensible control rule.

### Q-Learning

Use Q-Learning as the interpretable starting point. Discretise the five continuous state features into a small number of bins (for example 4–8 each); too many bins create a sparse Q-table. Update after each transition:

```python
q[state, action] += alpha * (
    reward + gamma * q[next_state].max() - q[state, action]
)
```

Use epsilon-greedy exploration with a documented decay schedule. Save bin edges, Q-table, hyperparameters, and seed.

### SARSA

SARSA uses the value of the next action actually selected under the exploratory policy:

```python
q[state, action] += alpha * (
    reward + gamma * q[next_state, next_action] - q[state, action]
)
```

Compare it to Q-Learning to discuss on-policy versus off-policy learning, especially around safety penalties.

### Deep Q-Network (DQN)

Use DQN once the environment and baselines are stable. Inputs are scaled continuous states; outputs are three action values. Include replay buffer, target network, batch training, epsilon schedule, gradient clipping, and separate training/evaluation environments. Start with a small MLP (two hidden layers) and use early stopping/model selection based on validation-scenario performance, not only training reward.

### Proximal Policy Optimization (PPO)

Use discrete-action PPO as the policy-gradient comparison. Train with vectorised seeded environments if practical; record rollout length, learning rate, batch size, clip range, entropy coefficient, and number of updates. Evaluate using deterministic actions. PPO is not automatically superior: compare it fairly under the same action bounds, episode horizon, reward, and scenario splits.

Use stable, maintained implementations such as Stable-Baselines3 where permitted, while keeping custom environment/reward code in the repository.

## 6. Experimental design and validation

### Avoid leakage

Do not randomly split individual rows across train and test: adjacent rows from the same time series are highly related. Keep sequences intact and fit scalers, transition models, discretisers, and any feature selection on the training portion only.

### Recommended evaluation matrix

Run multiple experiments, each with at least 5 random seeds where compute permits:

| Experiment | Training scenarios | Held-out scenarios | Purpose |
| --- | --- | --- | --- |
| Within-scenario | Earlier time blocks in one file | Later block in same file | Basic learning check |
| Load generalisation | 25%, 50%, 75% for all configurations | 100% for all configurations | High-load robustness |
| Configuration generalisation | Three configurations, all loads | The fourth configuration, all loads | Layout transfer |
| Leave-one-scenario-out | 15 configuration/load files | Remaining file | Fine-grained robustness |

The 16 AAU combinations are the evaluation units. Show results by each combination, then aggregate using mean and standard deviation (or confidence intervals) across seeds. Do not hide a poor high-load or unseen-layout result inside a global average.

### Metrics

For every controller and held-out scenario, report:

- mean, median, and total `P_DC` energy proxy;
- mean `PUE`, mean `CoP`, and their distributions;
- mean and maximum outlet temperature;
- safety-violation rate, count, maximum severity, and time outside limits;
- action distribution and switching rate (controller stability);
- episodic reward, shown alongside—not instead of—physical metrics;
- transition-model prediction error by target and scenario.

Use paired episode comparisons against the same initial states. Include plots of temperature and actions over time, energy/safety trade-off scatterplots, learning curves with seed bands, and a 16-scenario heatmap or table.

## 7. Suggested project layout

```text
ML_Assignment/
├── README.md
├── PROJECT_GUIDANCE.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   └── experiments/
├── data/                 # ignored; document how to obtain AAU data
│   ├── raw/
│   └── processed/
├── notebooks/             # exploratory only
├── src/
│   ├── data.py
│   ├── features.py
│   ├── transition_model.py
│   ├── environment.py
│   ├── baselines.py
│   ├── tabular_rl.py
│   ├── train_dqn.py
│   ├── train_ppo.py
│   ├── evaluate.py
│   └── plotting.py
├── tests/
│   ├── test_data.py
│   ├── test_environment.py
│   └── test_reward.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── results/              # ignored except selected small summaries
└── report/
```

Keep raw AAU data, trained model binaries, TensorBoard logs, and large figures out of Git unless using Git LFS. Commit code, configuration, data schema, small metadata files, and final reproducible result tables.

## 8. Implementation phases

1. **Data audit:** validate schema, units, time ordering, ranges, missingness, and scenario labels; produce descriptive plots.
2. **Preprocessing:** create leak-free splits, scaled features, scenario metadata, and next-step targets.
3. **Simulator:** train/evaluate transition model; add physical bounds and environment tests.
4. **Baselines:** implement fixed and rule-based policies; establish the reporting pipeline.
5. **Tabular RL:** implement Q-Learning and SARSA; tune only against validation scenarios.
6. **Deep RL:** add DQN and PPO after verifying environment behaviour and baselines.
7. **Robust evaluation:** run all scenario-split experiments with multiple seeds and produce tables/figures.
8. **Report:** explain modelling assumptions, causal limitations, safety rules, results, failure cases, and future work.

## 9. Reproducibility checklist

- Pin Python and dependency versions in `requirements.txt` or a lockfile.
- Store all parameters, scenario splits, reward weights, state bins, and action bounds in versioned YAML/JSON configurations.
- Use a master seed and log each run's seed, Git commit hash, data file checksums, timestamp, and hardware/library versions.
- Save transition-model metrics before RL training and policy checkpoints selected solely on validation data.
- Run `pytest` and an environment checker in continuous integration when the codebase is established.
- Keep a single command for training and another for evaluation, both driven by config files.

## 10. Recommended next actions

1. Add a short `README.md` that links here and records the course/assessment context.
2. Extract the AAU archive locally into an ignored `data/raw/AAU/` directory; do not alter the supplied project-reference copy.
3. Build the audit notebook/script and verify whether the source documentation identifies actual cooling-control inputs.
4. Agree the outlet-temperature limit and allowable cooling-supply range with the assignment specification or lecturer before fixing the reward.
5. Implement the deterministic fixed and rule-based controllers, then validate the data-driven environment before introducing DQN or PPO.

