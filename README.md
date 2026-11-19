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
python -m streamlit run app.py
```

In the sidebar, select one or more controllers, a scenario split, outlet limit, episode settings, and a seed, then select **Run experiment**. Results stay in the current dashboard session, so changing a chart filter does not retrain controllers.

- **Overview:** safety and efficiency KPIs, 16-scenario heatmap, episode replay, controller trade-off scatter, transition-model PUE surface, and a schematic AAU-average server-room replay.
- **Compare controllers:** physical metrics alongside reward, action distributions, learning curves, paired savings against the rule-based baseline, and CSV downloads.
- **Model diagnostics:** predicted-versus-actual plots, in-sample/evaluation MAE, scenario error heatmap, feature importances, and proposed-supply clipping count.
- **Source data:** dataset-quality KPIs, filtered AAU data export, supply-versus-load inspection, and time-series preview.

The **Load generalisation: hold out 100%** split fits the transition model and tabular discretiser only on the 25%, 50%, and 75% scenarios, then evaluates on all 100% scenarios. This dashboard runs only the learned simulator; AAU files have no logged control actions and it must not be used to operate physical cooling equipment.

### Reading a run

The default limit is 29 °C with a 28 °C warning. This is an assignment assumption informed by the AAU outlet distribution. Eligible starts have at least one future row and an outlet at or below the limit. The simulator prefers starts with a full horizon. Infeasible scenarios are listed separately. The first subsequent violation still ends an episode. Compare violation rates alongside peak temperature and episode length.

For generalisation results, hold out the 100% loads or one configuration. In **All scenarios**, both error series use the fitting rows. The sensitivity bars perturb each evaluated scenario's median state by ±0.5 °C, bounded to the training range. Small responses indicate limited opportunity for simulated optimisation. Paired energy differences integrate power over common two-minute steps; intervals require repeated episodes. Increase consecutive seeds to display training variability.

[Overview](docs/ui/overview-2800.png) · [Controller comparison](docs/ui/compare-controllers-2800.png) · [Model diagnostics](docs/ui/model-diagnostics-2800.png) · [Source data](docs/ui/source-data-2800.png). Each view also has a 1600 × 1000 capture in the same folder. To repeat the captures on Windows with Edge installed, install Playwright and run `python scripts/capture_dashboard.py` from the project folder.

### Episode and simulator comparison

The headline violation rate counts **episodes ending at their first violation divided by evaluated episodes**. Steps survived counts executed transitions including the terminal one. Survival fraction divides that count by the requested horizon. The descriptive violating-step share uses a different denominator (executed steps). Replays distinguish violation, horizon and source-data exhaustion, and retain the pre-action decision reason and matched starting row.

The level forest is the default reference. The delta forest learns next-minus-current outlet and return temperatures and converts them back to levels. Monotone boosting constrains those two predictions to be non-decreasing with supply temperature, an explicit physical assumption. All three leave power and efficiency unconstrained. Select a variant in the sidebar and inspect held-out error, sensitivity and power-sign flags in Diagnostics.

Tabular training defaults to a constant initial Q equal to the negative average training-row energy, PUE and thermal penalties with zero switching penalty. Advanced settings expose legacy zero initialisation. Greedy evaluation considers visited actions only. An unseen state chooses maintain. When eligible values are tied or within the default 0.05 reward-unit margin, maintain takes precedence if eligible and within that margin. These rules affect evaluation independently of initialisation.

The benchmark uses 250 training episodes per tabular policy and seeds 42–44. Reproduce the [held-out comparison](docs/ui/REPORT.md) with `python scripts/verify_variants.py`.

## Safety and scientific limits

The AAU CSVs contain observations, not explicit cooling-control actions. The environment is therefore a learned counterfactual approximation. Interpret its policies as experimental recommendations only. Keep cooling setpoint and outlet-temperature constraints enabled, report violations separately from reward, and validate every result on held-out configuration/load scenarios.

