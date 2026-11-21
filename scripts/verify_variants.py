"""Reproduce the held-out three-seed comparison in docs/ui/REPORT.md."""

from math import ceil
from pathlib import Path

import pandas as pd

from data_centre_rl.data import load_csv_directory
from data_centre_rl.experiments import (
    BOUNDS,
    action_sensitivity,
    prepare_experiment,
    run_controller,
)
from data_centre_rl.transition_model import VARIANTS


def table(frame):
    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def benchmark(variant):
    data = load_csv_directory("data/raw/AAU")
    context = prepare_experiment(
        data,
        split_name="Load generalisation: hold out 100%",
        variant=variant,
        trees=50,
        seed=42,
    )
    sensitivity = action_sensitivity(context)
    flags = sensitivity.loc[(sensitivity.applied < 0) & (sensitivity.P_DC < 0), "scenario"]
    diagnostics = {
        "Simulator": variant,
        **{key: f"{value:.4f}" for key, value in context.diagnostics.held_out_mae.items()},
        "Mean abs Δ outlet (°C)": f"{sensitivity['Outlet Avg Temp'].abs().mean():.4f}",
        "Mean abs Δ power (kW)": f"{sensitivity.P_DC.abs().mean():.4f}",
        "Terminal cost per remaining step": ceil(
            context.training_data.P_DC.max() + 0.5 * context.training_data.PUE.max()
        ),
        "Power sign flags": ", ".join(flags) or "None",
    }
    violation_sets = {}
    rows = []
    base_cache = {}
    corrected = {}
    legacy_baseline = None
    for legacy in [True, False]:
        for legacy_state in [True, False]:
            for controller in [
                "Rule-based (no hysteresis)",
                "Rule-based",
                "Q-Learning",
                "SARSA",
                *BOUNDS,
            ]:
                by_seed = []
                for seed in [42, 43, 44]:
                    key = (controller, seed, legacy)
                    if key in base_cache:
                        metrics = base_cache[key]
                    else:
                        run = run_controller(
                            context,
                            controller=controller,
                            outlet_limit=29,
                            horizon=50,
                            episodes_per_scenario=3,
                            tabular_episodes=300,
                            deep_timesteps=0,
                            legacy_termination=legacy,
                            legacy_state=legacy_state,
                            seed=seed,
                        )
                        metrics = run.scenario_metrics
                        if controller.startswith("Rule-based") or controller in BOUNDS:
                            base_cache[key] = metrics
                    if not legacy and not legacy_state:
                        hot = metrics.loc[
                            metrics.any_violation, ["seed", "scenario", "episode", "start_row"]
                        ]
                        violation_sets.setdefault(controller, set()).update(
                            hot.itertuples(index=False, name=None)
                        )
                    by_seed.append(
                        {
                            "Any violation (%)": 100 * metrics.any_violation.mean(),
                            "Violating steps (%)": 100
                            * metrics.violation_count.sum()
                            / metrics.steps.sum(),
                            "Steps": metrics.steps.mean(),
                            "Return / step": metrics.return_per_step.mean(),
                            "PUE": metrics.mean_pue.mean(),
                            "Power (kW)": metrics.mean_p_dc.mean(),
                            "Switching (%)": 100 * metrics.switching_rate.mean(),
                        }
                    )
                frame = pd.DataFrame(by_seed)
                if legacy and not legacy_state and controller == "Rule-based":
                    legacy_baseline = frame
                if not legacy and not legacy_state:
                    corrected[controller] = frame
                rows.append(
                    {
                        "Termination": "Legacy" if legacy else "Serious",
                        "State": "5-bin" if legacy_state else "Safety",
                        "Controller": controller,
                        **{k: f"{frame[k].mean():.3f} ± {frame[k].std():.3f}" for k in frame},
                    }
                )
            print(variant, legacy, legacy_state, flush=True)
    comparisons = []
    baseline = corrected["Rule-based"]
    for metric in ["Steps", "Any violation (%)", "Violating steps (%)"]:
        before, after = legacy_baseline[metric], baseline[metric]
        comparisons.append(
            f"Hysteresis {metric}, legacy to corrected termination: {before.mean():.3f} ± {before.std():.3f} to {after.mean():.3f} ± {after.std():.3f}."
        )

    for name in ["Q-Learning", "SARSA"]:
        diff = corrected[name] - corrected["Rule-based"]
        comparisons.append(
            f"{name} minus hysteresis, paired by seed: any violation {diff['Any violation (%)'].mean():+.2f} ± {diff['Any violation (%)'].std():.2f} percentage points; mean power {diff['Power (kW)'].mean():+.4f} ± {diff['Power (kW)'].std():.4f} kW."
        )
        saving_pct = -100 * diff["Power (kW)"].mean() / baseline["Power (kW)"].mean()
        hot_steps = diff["Violating steps (%)"].mean()
        qualifies = saving_pct >= 1 and diff["Any violation (%)"].mean() <= 0 and hot_steps <= 0
        comparisons.append(
            f"{name} {'met' if qualifies else 'did not meet'} the joint benchmark of at least 1% lower mean power and no increase in either safety measure. Mean power saving was {saving_pct:+.2f}%, and the violating-step share changed by {hot_steps:+.2f} percentage points. {'This is a power–safety trade-off.' if saving_pct > 0 and hot_steps > 0 else ''}"
        )
    bounds = []
    for name in BOUNDS:
        bounds.append(
            {
                "Simulator": variant,
                "Controller": name,
                **{
                    k: f"{corrected[name][k].mean():.3f} ± {corrected[name][k].std():.3f}"
                    for k in ["Any violation (%)", "Violating steps (%)", "Power (kW)", "PUE"]
                },
            }
        )
    maximum = violation_sets["Bound: max cooling"]
    minimum = violation_sets["Bound: min cooling"]
    overlap = ", ".join(
        f"{name} {len(starts & maximum)}/{len(starts)}"
        for name, starts in violation_sets.items()
        if name in {"Rule-based", "Q-Learning", "SARSA"}
    )
    change = (
        corrected["Bound: max cooling"]["Any violation (%)"].mean()
        - corrected["Bound: min cooling"]["Any violation (%)"].mean()
    )
    floor = f"{variant}: maximum cooling {'lowered' if change < 0 else 'raised' if change > 0 else 'did not change'} episode violation rate versus minimum cooling ({change:+.2f} percentage points). {len(maximum & minimum)} matched starts violated under both extremes. Violating starts also hot under maximum cooling: {overlap}."
    return diagnostics, table(pd.DataFrame(rows)), comparisons, bounds, floor


def main():
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(benchmark, VARIANTS))
    report = [
        "# Held-out cooling experiments",
        "Each model was fitted once on 25%, 50% and 75% loads (50 estimators, seed 42). Evaluation used the four held-out 100% scenarios, three matched starts per scenario for each seed (42–44), limit 29 °C and horizon 50. Each tabular run trained for 300 episodes. Values are seed means ± sample standard deviation. The violating-step percentage pools steps within each seed. Other metrics average episodes within each seed. The two rule controllers do not use a state encoder.",
        "## Reward design correction",
        "The original pipeline analysis supplied with the assignment measured a typical step cost of about 3.25, a mean excess of 0.16 °C, a mean violation cost of about 0.43 and a return–duration correlation of −0.96. Ending 10, 30 or 49 steps early avoided about 33, 98 or 159 cost units. Corrected runs continue through mild excess, charging quadratic cost each step. Excess above 1.5 °C ends the run and incurs the training-derived cost for each remaining step. The tables compare both rules at matched starts and report return per observed step.",
        "## Simulator validation",
        table(pd.DataFrame([item[0] for item in results])),
        "Sensitivity is the mean absolute response to a ±0.5 °C supply change around the median held-out state. The small predicted outlet responses give the agent little leverage in these scenarios.",
    ]
    report.extend(
        [
            "## Unavoidable violation floor",
            "These constant-action bounds test the sensitivity of the fitted simulators. A shared violation is consistent with load or initial conditions dominating control. Observational data cannot identify a causal load fraction or prove that every intermediate policy must violate.",
            table(pd.DataFrame([row for item in results for row in item[3]])),
            *[item[4] for item in results],
        ]
    )
    for variant, (_, rows, comparisons, _, _) in zip(VARIANTS, results, strict=True):
        report.extend(["## " + variant, rows, *comparisons])
    Path("docs/ui/REPORT.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
