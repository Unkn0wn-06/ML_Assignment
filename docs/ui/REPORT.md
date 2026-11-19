# Held-out cooling experiments

Fit each simulator once (50 trees or boosting iterations, fit seed 42) on all 25%, 50% and 75% load files. Evaluate on the four 100% files. Each controller used the same seeded starts, limit 29 °C, eligible starts, horizon 50 and three episodes per scenario. Tabular policies trained for 250 episodes. Results give mean ± sample standard deviation of three seed-level episode means (42–44).

## Surrogate validation

First five numeric columns are held-out next-step MAE in source units. Sensitivity averages absolute responses to both ±0.5 °C supply perturbations around each held-out scenario median. Sign flags identify a predicted power decrease under stronger cooling.

| Simulator | Outlet Avg Temp | Cooling Return Average | P_DC | PUE | CoP | Mean abs Δ outlet (°C) | Mean abs Δ power (kW) | Power sign flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random forest (levels) | 0.2975 | 0.6854 | 0.1520 | 0.2889 | 0.1830 | 0.0143 | 0.0037 | BaselineTest_100percent |
| Random forest (deltas) | 0.2270 | 0.6557 | 0.1555 | 0.2674 | 0.1712 | 0.0188 | 0.0042 | None |
| Monotone boosting (assumption) | 0.3017 | 0.7094 | 0.1513 | 0.2641 | 0.1775 | 0.0312 | 0.0011 | None |

## Simulated control

A violation contributes one failed episode regardless of its duration. Duration includes the terminal transition. Switching is the fraction of consecutive action pairs that differ. Power is the episode mean over its observed duration.

| Simulator | Controller | Episodes violated (%) | Mean steps | Switching (%) | Power (kW) |
| --- | --- | --- | --- | --- | --- |
| Random forest (levels) | Rule-based | 13.889 ± 4.811 | 43.194 ± 2.358 | 1.077 ± 0.708 | 2.090 ± 0.011 |
| Random forest (levels) | Q-Learning | 22.222 ± 4.811 | 39.472 ± 2.045 | 44.227 ± 8.962 | 2.080 ± 0.001 |
| Random forest (levels) | SARSA | 22.222 ± 4.811 | 39.472 ± 2.045 | 48.621 ± 4.802 | 2.085 ± 0.007 |
| Random forest (deltas) | Rule-based | 13.889 ± 4.811 | 43.194 ± 2.358 | 3.741 ± 0.295 | 2.138 ± 0.009 |
| Random forest (deltas) | Q-Learning | 38.889 ± 12.729 | 37.583 ± 2.554 | 55.671 ± 3.378 | 2.106 ± 0.014 |
| Random forest (deltas) | SARSA | 30.556 ± 4.811 | 39.972 ± 0.809 | 56.317 ± 4.990 | 2.112 ± 0.006 |
| Monotone boosting (assumption) | Rule-based | 8.333 ± 0.000 | 45.917 ± 0.000 | 2.778 ± 0.393 | 2.204 ± 0.006 |
| Monotone boosting (assumption) | Q-Learning | 19.444 ± 4.811 | 43.028 ± 0.268 | 51.077 ± 1.659 | 2.206 ± 0.008 |
| Monotone boosting (assumption) | SARSA | 30.556 ± 4.811 | 37.917 ± 1.665 | 49.620 ± 1.371 | 2.206 ± 0.012 |

The level forest remains the reference. Temperature monotonicity is an imposed physical assumption. Differences between controllers depend on the surrogate, episode duration and seed. Logged AAU data do not identify causal savings.

On the held-out 100% loads, both tabular controllers switched more frequently and had higher mean violation rates than the rule-based controller in every simulator. The two forests gave the tabular policies lower mean power over their shorter observed episodes. Those averages alone do not establish energy savings at equal safety or duration. The levels model also predicted lower power for increased cooling in Baseline at full load. The delta model had the lowest outlet MAE; monotone boosting had the largest average outlet response but a slightly higher outlet MAE than levels.
