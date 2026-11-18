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

Launch the interactive control-room dashboard with:

```powershell
python -m pip install -e ".[ui]"
streamlit run app.py
```

In the sidebar, select one or more controllers, a scenario split, outlet limit, episode settings, and a seed, then select **Run experiment**. Results stay in the current dashboard session, so changing a chart filter does not retrain controllers.

- **Overview:** safety and efficiency KPIs, 16-scenario heatmap, episode replay, controller trade-off scatter, transition-model PUE surface, and a schematic AAU-average server-room replay.
- **Compare controllers:** physical metrics alongside reward, action distributions, learning curves, paired savings against the rule-based baseline, and CSV downloads.
- **Model diagnostics:** predicted-versus-actual plots, in-sample/evaluation MAE, scenario error heatmap, feature importances, and proposed-supply clipping count.
- **Source data:** dataset-quality KPIs, filtered AAU data export, supply-versus-load inspection, and time-series preview.

The **Load generalisation: hold out 100%** split fits the transition model and tabular discretiser only on the 25%, 50%, and 75% scenarios, then evaluates on all 100% scenarios. This dashboard runs only the learned simulator; AAU files have no logged control actions and it must not be used to operate physical cooling equipment.

## Safety and scientific limits

The AAU CSVs contain observations, not explicit cooling-control actions. The environment is therefore a learned counterfactual approximation. Interpret its policies as experimental recommendations only. Keep cooling setpoint and outlet-temperature constraints enabled, report violations separately from reward, and validate every result on held-out configuration/load scenarios.

