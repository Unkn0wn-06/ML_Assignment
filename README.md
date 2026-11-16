# Data Centre Cooling RL

This project evaluates safe, energy-aware cooling policies in a data-driven simulator built from the AAU data-centre datasets. It is an academic simulation project: it must not be connected to a live cooling system.

The design decisions, assumptions, limitations, experiment matrix, and report checklist are in [PROJECT_GUIDANCE.md](PROJECT_GUIDANCE.md).

## Quick start

Create an environment and install the project with its development tools:

```powershell
python -m pip install -e ".[dev]"
```

For DQN and PPO, install the optional deep-RL stack as well:

```powershell
python -m pip install -e ".[deep-rl]"
```

Prepare a local copy of the supplied AAU archive (the raw data stays out of Git):

```powershell
python scripts/prepare_data.py --archive "..\sources\AAU.tar.xz"
```

Train the transition model and a rule-based baseline:

```powershell
python scripts/train.py --algorithm baseline --data-dir data/raw/AAU
```

Train a tabular agent after checking the environment and transition-model diagnostics:

```powershell
python scripts/train.py --algorithm q_learning --data-dir data/raw/AAU --episodes 250
```

`dqn` and `ppo` use Stable-Baselines3 and are available through the same script after dependencies are installed. All outputs are written under the ignored `artifacts/` directory.

Use `--evaluation-episodes 5` (or more) for report results; the default of one episode per scenario is intended only as a quick smoke test. `--episode-horizon` can shorten an integration check, but retain the configured full horizon for final experiments.

## Dashboard

Launch the interactive results dashboard with:

```powershell
python -m pip install -e ".[ui]"
streamlit run app.py
```

In the sidebar, select the extracted AAU data folder, controller, episode horizon, and evaluation repetitions, then select **Run experiment**. The dashboard displays scenario-level energy, efficiency, temperature, safety, action-stability, and transition-model diagnostics. It runs only the learned simulator; it must not be used to operate physical cooling equipment.

## Safety and scientific limits

The AAU CSVs contain observations, not explicit cooling-control actions. The environment is therefore a learned counterfactual approximation. Interpret its policies as experimental recommendations only. Keep cooling setpoint and outlet-temperature constraints enabled, report violations separately from reward, and validate every result on held-out configuration/load scenarios.

