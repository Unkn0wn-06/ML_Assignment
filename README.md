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

- **Viewer:** select the controller, seed, scenario, episode and comparison partner above the room. Expand **Run metrics and analysis charts** for the KPIs, scenario heatmap, three-row replay, controller trade-offs and PUE surface.
- **Compare controllers:** physical metrics alongside reward, action distributions, learning curves, paired savings against the rule-based baseline, and CSV downloads.
- **Model diagnostics:** predicted-versus-actual plots, in-sample/evaluation MAE, scenario error heatmap, feature importances, and proposed-supply clipping count.
- **Source data:** dataset-quality KPIs, filtered AAU data export, supply-versus-load inspection, and time-series preview.

The **Load generalisation: hold out 100%** split fits the transition model and tabular discretiser only on the 25%, 50%, and 75% scenarios, then evaluates on all 100% scenarios. This dashboard runs only the learned simulator; AAU files have no logged control actions and it must not be used to operate physical cooling equipment.

### Reading a run

The default limit is 29 °C with a 28 °C warning. This is an assignment assumption informed by the AAU outlet distribution. Eligible starts have at least one future row and an outlet at or below the limit. The simulator prefers starts with a full horizon. Infeasible scenarios are listed separately. Mild violations continue to accrue penalties. Compare violation rates alongside peak temperature and episode length.

For generalisation results, hold out the 100% loads or one configuration. In **All scenarios**, both error series use the fitting rows. The sensitivity bars perturb each evaluated scenario's median state by ±0.5 °C, bounded to the training range. Small responses indicate limited opportunity for simulated optimisation. Paired savings compare mean power on matched full-horizon episodes and include the difference in violation probability. Increase consecutive seeds to display training variability.

### Using the viewer

Choose **3D** to orbit the room by dragging and zoom with the wheel. **Plan** looks straight down on the same equipment. Select a zone marker to pin its schematic readings. The left rail resets the camera, hides the legends or opens the inspector. On narrow screens the inspector opens as a drawer.

**Loop** follows the blue supply and red return circuits. Temperature labels track the selected step. The histogram uses observed AAU rows from the selected scenario. Configuration summaries pool the selected controller's evaluated episodes across loads and seeds. A missing configuration is labelled explicitly.

Scrub from step zero (the recorded initial state) to the last executed step. The end chip gives the requested horizon. Play animates the readings at the selected speed. Switch on **Compare** to inspect the other controller at the same scenario, initial row, seed and step. The inspector marks where a shorter comparison episode ends.

Rack placement, gradients and zone markers are schematic. Every zone displays the same underlying readings. AAU provides IT and total facility power; it does not separately meter cooling power.

**1600 × 900 views** · [3D viewer](docs/ui/viewer-3d.png) · [Plan](docs/ui/viewer-plan.png) · [Cooling loop](docs/ui/viewer-loop.png) · [Compare](docs/ui/compare-900.png) · [Diagnostics](docs/ui/diagnostics-900.png) · [Source data](docs/ui/source-data-900.png). To repeat the captures on Windows with Edge and Playwright installed, run `python scripts/capture_dashboard.py`. The three viewer captures show the component at full viewport width. The other captures include the Streamlit experiment controls.

The design uses the same light room and dark loop as the previews. Plan shares the room geometry and camera controls. The inspector uses real scenario-wide readings, and the configuration list counts evaluated episodes. Streamlit keeps its navigation above the component so Compare, Diagnostics and Source data retain their filters and downloads. The six captures use seed 42.

### Episode and simulator comparison

The headline violation rate counts **episodes with any positive outlet excess divided by evaluated episodes**. Steps survived counts executed transitions including the terminal one. Survival fraction divides that count by the requested horizon. The descriptive violating-step share uses a different denominator (executed steps). Replays distinguish violation, horizon and source-data exhaustion, and retain the pre-action decision reason and matched starting row.

The level forest is the default reference. The delta forest learns next-minus-current outlet and return temperatures and converts them back to levels. Monotone boosting constrains those two predictions to be non-decreasing with supply temperature, an explicit physical assumption. All three leave power and efficiency unconstrained. Select a variant in the sidebar and inspect held-out error, sensitivity and power-sign flags in Diagnostics.

With `legacy_termination=false`, outlet excess incurs a quadratic penalty every step. Only excess strictly above `termination_excess` (1.5 °C) terminates early. The final reward includes `termination_penalty_per_step × remaining steps`. With the default null setting, this rate is the ceiling of the sum of weighted training maxima for power and PUE. The run records the rate. Legacy termination stops at the first positive excess and omits the remaining-step charge. Return per step includes every penalty. Degree-steps sum positive temperature excess across observed steps.

The 162-state safety encoder combines six outlet safety margins, three outlet trends, three supply positions within training bounds, and the previous action. Episodes start with flat trend and maintain. Advanced settings expose the legacy five-bin discretiser, learning rate (0.1), discount (0.9), and exploration schedule (1 to 0.05 over the first 60% of 300 episodes). Evaluation requires ten visits per state-action pair. Without a trusted action the policy maintains. Trusted maintain wins within the greater of 0.05 and 2% of the best trusted Q-value magnitude.

The rule-based controller enters decrease at the lower threshold or increase at the warning threshold. It releases 0.5 °C inside each threshold and holds changes for five steps. At 0.25 °C below the limit or hotter, increasing cooling takes priority. Advanced options adjust hold and release. Select **Rule-based (no hysteresis)** for the original threshold controller.

Reproduce the [held-out comparison](docs/ui/REPORT.md) with `python scripts/verify_variants.py`. It compares both termination rules and both encoders with matched seeds 42–44.

## Safety and scientific limits

The AAU CSVs contain observations, not explicit cooling-control actions. The environment is therefore a learned counterfactual approximation. Interpret its policies as experimental recommendations only. Keep cooling setpoint and outlet-temperature constraints enabled, report violations separately from reward, and validate every result on held-out configuration/load scenarios.

