"""Reproduce the held-out three-seed comparison in docs/ui/REPORT.md."""

from pathlib import Path

import pandas as pd

from data_centre_rl.data import load_csv_directory
from data_centre_rl.experiments import action_sensitivity, prepare_experiment, run_controller
from data_centre_rl.transition_model import VARIANTS


def table(frame):
    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main():
    data = load_csv_directory("data/raw/AAU")
    diagnostics = []
    results = []
    for variant in VARIANTS:
        context = prepare_experiment(
            data,
            split_name="Load generalisation: hold out 100%",
            variant=variant,
            trees=50,
            seed=42,
        )
        sensitivity = action_sensitivity(context)
        flags = sensitivity.loc[(sensitivity.applied < 0) & (sensitivity.P_DC < 0), "scenario"]
        diagnostics.append(
            {
                "Simulator": variant,
                **{key: f"{value:.4f}" for key, value in context.diagnostics.held_out_mae.items()},
                "Mean abs Δ outlet (°C)": f"{sensitivity['Outlet Avg Temp'].abs().mean():.4f}",
                "Mean abs Δ power (kW)": f"{sensitivity.P_DC.abs().mean():.4f}",
                "Power sign flags": ", ".join(flags) or "None",
            }
        )
        for controller in ("Rule-based", "Q-Learning", "SARSA"):
            seeds = []
            for seed in (42, 43, 44):
                run = run_controller(
                    context,
                    controller=controller,
                    outlet_limit=29,
                    horizon=50,
                    episodes_per_scenario=3,
                    tabular_episodes=250,
                    deep_timesteps=0,
                    seed=seed,
                )
                metrics = run.scenario_metrics
                seeds.append(
                    {
                        "Episodes violated (%)": 100 * metrics.violated.mean(),
                        "Mean steps": metrics.steps_survived.mean(),
                        "Switching (%)": 100 * metrics.switching_rate.mean(),
                        "Power (kW)": metrics.mean_p_dc.mean(),
                    }
                )
            spread = pd.DataFrame(seeds)
            results.append(
                {
                    "Simulator": variant,
                    "Controller": controller,
                    **{
                        key: f"{spread[key].mean():.3f} ± {spread[key].std():.3f}" for key in spread
                    },
                }
            )
            print(variant, controller, results[-1], flush=True)
    report = [
        "# Held-out cooling experiments",
        "Fit each simulator once (50 trees or boosting iterations, fit seed 42) on all 25%, 50% and 75% load files. Evaluate on the four 100% files. Each controller used the same seeded starts, limit 29 °C, eligible starts, horizon 50 and three episodes per scenario. Tabular policies trained for 250 episodes. Results give mean ± sample standard deviation of three seed-level episode means (42–44).",
        "## Surrogate validation",
        "First five numeric columns are held-out next-step MAE in source units. Sensitivity averages absolute responses to both ±0.5 °C supply perturbations around each held-out scenario median. Sign flags identify a predicted power decrease under stronger cooling.",
        table(pd.DataFrame(diagnostics)),
        "## Simulated control",
        "A violation contributes one failed episode regardless of its duration. Duration includes the terminal transition. Switching is the fraction of consecutive action pairs that differ. Power is the episode mean over its observed duration.",
        table(pd.DataFrame(results)),
        "The level forest remains the reference. Temperature monotonicity is an imposed physical assumption. Differences between controllers depend on the surrogate, episode duration and seed. Logged AAU data do not identify causal savings.",
    ]

    Path("docs/ui/REPORT.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
