# Held-out cooling experiments

Each model was fitted once on 25%, 50% and 75% loads (50 estimators, seed 42). Evaluation used the four held-out 100% scenarios, three matched starts per scenario for each seed (42–44), limit 29 °C and horizon 50. Each tabular run trained for 300 episodes. Values are seed means ± sample standard deviation. The violating-step percentage pools steps within each seed. Other metrics average episodes within each seed. The two rule controllers do not use a state encoder.

## Reward design correction

The original pipeline analysis supplied with the assignment measured a typical step cost of about 3.25, a mean excess of 0.16 °C, a mean violation cost of about 0.43 and a return–duration correlation of −0.96. Ending 10, 30 or 49 steps early avoided about 33, 98 or 159 cost units. Corrected runs continue through mild excess, charging quadratic cost each step. Excess above 1.5 °C ends the run and incurs the training-derived cost for each remaining step. The tables compare both rules at matched starts and report return per observed step.

## Simulator validation

| Simulator | Outlet Avg Temp | Cooling Return Average | P_DC | PUE | CoP | Mean abs Δ outlet (°C) | Mean abs Δ power (kW) | Terminal cost per remaining step | Power sign flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random forest (levels) | 0.2975 | 0.6854 | 0.1520 | 0.2889 | 0.1830 | 0.0143 | 0.0037 | 5 | BaselineTest_100percent |
| Random forest (deltas) | 0.2270 | 0.6557 | 0.1555 | 0.2674 | 0.1712 | 0.0188 | 0.0042 | 5 | None |
| Monotone boosting (assumption) | 0.3017 | 0.7094 | 0.1513 | 0.2641 | 0.1775 | 0.0312 | 0.0011 | 5 | None |

Sensitivity is the mean absolute response to a ±0.5 °C supply change around the median held-out state. The small predicted outlet responses give the agent little leverage in these scenarios.

## Random forest (levels)

| Termination | State | Controller | Any violation (%) | Violating steps (%) | Steps | Return / step | Power (kW) | Switching (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Legacy | 5-bin | Rule-based (no hysteresis) | 13.889 ± 4.811 | 0.326 ± 0.125 | 43.194 ± 2.358 | -3.352 ± 0.011 | 2.090 ± 0.011 | 1.077 ± 0.708 |
| Legacy | 5-bin | Rule-based | 13.889 ± 4.811 | 0.326 ± 0.125 | 43.194 ± 2.358 | -3.358 ± 0.017 | 2.094 ± 0.015 | 0.340 ± 0.340 |
| Legacy | 5-bin | Q-Learning | 22.222 ± 4.811 | 0.568 ± 0.147 | 39.472 ± 2.045 | -3.335 ± 0.011 | 2.084 ± 0.007 | 26.276 ± 8.102 |
| Legacy | 5-bin | SARSA | 22.222 ± 4.811 | 0.568 ± 0.147 | 39.472 ± 2.045 | -3.331 ± 0.010 | 2.082 ± 0.006 | 25.340 ± 2.595 |
| Legacy | Safety | Rule-based (no hysteresis) | 13.889 ± 4.811 | 0.326 ± 0.125 | 43.194 ± 2.358 | -3.352 ± 0.011 | 2.090 ± 0.011 | 1.077 ± 0.708 |
| Legacy | Safety | Rule-based | 13.889 ± 4.811 | 0.326 ± 0.125 | 43.194 ± 2.358 | -3.358 ± 0.017 | 2.094 ± 0.015 | 0.340 ± 0.340 |
| Legacy | Safety | Q-Learning | 25.000 ± 0.000 | 0.654 ± 0.001 | 38.250 ± 0.083 | -3.328 ± 0.010 | 2.082 ± 0.008 | 16.249 ± 7.088 |
| Legacy | Safety | SARSA | 25.000 ± 0.000 | 0.645 ± 0.013 | 38.750 ± 0.795 | -3.345 ± 0.011 | 2.085 ± 0.008 | 36.536 ± 11.144 |
| Serious | 5-bin | Rule-based (no hysteresis) | 13.889 ± 4.811 | 0.556 ± 0.192 | 50.000 ± 0.000 | -3.330 ± 0.002 | 2.081 ± 0.007 | 1.077 ± 0.708 |
| Serious | 5-bin | Rule-based | 13.889 ± 4.811 | 0.556 ± 0.192 | 50.000 ± 0.000 | -3.336 ± 0.006 | 2.085 ± 0.010 | 0.340 ± 0.340 |
| Serious | 5-bin | Q-Learning | 16.667 ± 8.333 | 3.889 ± 1.602 | 50.000 ± 0.000 | -3.328 ± 0.001 | 2.079 ± 0.006 | 23.810 ± 2.784 |
| Serious | 5-bin | SARSA | 19.444 ± 4.811 | 4.333 ± 0.833 | 50.000 ± 0.000 | -3.331 ± 0.009 | 2.079 ± 0.005 | 30.669 ± 12.822 |
| Serious | Safety | Rule-based (no hysteresis) | 13.889 ± 4.811 | 0.556 ± 0.192 | 50.000 ± 0.000 | -3.330 ± 0.002 | 2.081 ± 0.007 | 1.077 ± 0.708 |
| Serious | Safety | Rule-based | 13.889 ± 4.811 | 0.556 ± 0.192 | 50.000 ± 0.000 | -3.336 ± 0.006 | 2.085 ± 0.010 | 0.340 ± 0.340 |
| Serious | Safety | Q-Learning | 25.000 ± 0.000 | 5.111 ± 1.182 | 50.000 ± 0.000 | -3.347 ± 0.017 | 2.086 ± 0.003 | 41.837 ± 23.418 |
| Serious | Safety | SARSA | 22.222 ± 4.811 | 4.722 ± 0.977 | 50.000 ± 0.000 | -3.336 ± 0.015 | 2.082 ± 0.004 | 40.873 ± 21.854 |

Hysteresis Steps, legacy to corrected termination: 43.194 ± 2.358 to 50.000 ± 0.000.

Hysteresis Any violation (%), legacy to corrected termination: 13.889 ± 4.811 to 13.889 ± 4.811.

Hysteresis Violating steps (%), legacy to corrected termination: 0.326 ± 0.125 to 0.556 ± 0.192.

Q-Learning minus hysteresis, paired by seed: any violation +11.11 ± 4.81 percentage points; mean power +0.0015 ± 0.0081 kW.

Q-Learning did not meet the joint benchmark of at least 1% lower mean power and no increase in either safety measure. Mean power saving was -0.07%, and the violating-step share changed by +4.56 percentage points.

SARSA minus hysteresis, paired by seed: any violation +8.33 ± 8.33 percentage points; mean power -0.0031 ± 0.0098 kW.

SARSA did not meet the joint benchmark of at least 1% lower mean power and no increase in either safety measure. Mean power saving was +0.15%, and the violating-step share changed by +4.17 percentage points.

## Random forest (deltas)

| Termination | State | Controller | Any violation (%) | Violating steps (%) | Steps | Return / step | Power (kW) | Switching (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Legacy | 5-bin | Rule-based (no hysteresis) | 13.889 ± 4.811 | 0.326 ± 0.125 | 43.194 ± 2.358 | -3.369 ± 0.013 | 2.138 ± 0.009 | 3.741 ± 0.295 |
| Legacy | 5-bin | Rule-based | 13.889 ± 4.811 | 0.326 ± 0.125 | 43.194 ± 2.358 | -3.383 ± 0.013 | 2.150 ± 0.010 | 0.737 ± 0.393 |
| Legacy | 5-bin | Q-Learning | 27.778 ± 9.623 | 0.701 ± 0.262 | 40.028 ± 1.684 | -3.339 ± 0.002 | 2.107 ± 0.007 | 24.588 ± 9.235 |
| Legacy | 5-bin | SARSA | 33.333 ± 8.333 | 0.868 ± 0.220 | 38.417 ± 0.441 | -3.336 ± 0.021 | 2.106 ± 0.017 | 23.719 ± 3.431 |
| Legacy | Safety | Rule-based (no hysteresis) | 13.889 ± 4.811 | 0.326 ± 0.125 | 43.194 ± 2.358 | -3.369 ± 0.013 | 2.138 ± 0.009 | 3.741 ± 0.295 |
| Legacy | Safety | Rule-based | 13.889 ± 4.811 | 0.326 ± 0.125 | 43.194 ± 2.358 | -3.383 ± 0.013 | 2.150 ± 0.010 | 0.737 ± 0.393 |
| Legacy | Safety | Q-Learning | 44.444 ± 9.623 | 1.226 ± 0.308 | 36.528 ± 1.577 | -3.333 ± 0.013 | 2.100 ± 0.007 | 27.535 ± 13.552 |
| Legacy | Safety | SARSA | 41.667 ± 0.000 | 1.177 ± 0.022 | 35.417 ± 0.667 | -3.347 ± 0.015 | 2.101 ± 0.016 | 28.097 ± 16.267 |
| Serious | 5-bin | Rule-based (no hysteresis) | 13.889 ± 4.811 | 0.889 ± 0.192 | 50.000 ± 0.000 | -3.348 ± 0.001 | 2.142 ± 0.008 | 4.535 ± 0.098 |
| Serious | 5-bin | Rule-based | 13.889 ± 4.811 | 0.889 ± 0.192 | 50.000 ± 0.000 | -3.361 ± 0.004 | 2.154 ± 0.009 | 0.737 ± 0.393 |
| Serious | 5-bin | Q-Learning | 19.444 ± 4.811 | 4.889 ± 1.337 | 50.000 ± 0.000 | -3.337 ± 0.012 | 2.103 ± 0.017 | 33.107 ± 9.969 |
| Serious | 5-bin | SARSA | 22.222 ± 17.347 | 4.944 ± 1.584 | 50.000 ± 0.000 | -3.335 ± 0.021 | 2.105 ± 0.026 | 30.499 ± 8.024 |
| Serious | Safety | Rule-based (no hysteresis) | 13.889 ± 4.811 | 0.889 ± 0.192 | 50.000 ± 0.000 | -3.348 ± 0.001 | 2.142 ± 0.008 | 4.535 ± 0.098 |
| Serious | Safety | Rule-based | 13.889 ± 4.811 | 0.889 ± 0.192 | 50.000 ± 0.000 | -3.361 ± 0.004 | 2.154 ± 0.009 | 0.737 ± 0.393 |
| Serious | Safety | Q-Learning | 30.556 ± 4.811 | 5.611 ± 0.347 | 50.000 ± 0.000 | -3.329 ± 0.012 | 2.097 ± 0.014 | 25.964 ± 7.491 |
| Serious | Safety | SARSA | 30.556 ± 12.729 | 4.944 ± 2.637 | 50.000 ± 0.000 | -3.330 ± 0.017 | 2.098 ± 0.009 | 39.229 ± 27.133 |

Hysteresis Steps, legacy to corrected termination: 43.194 ± 2.358 to 50.000 ± 0.000.

Hysteresis Any violation (%), legacy to corrected termination: 13.889 ± 4.811 to 13.889 ± 4.811.

Hysteresis Violating steps (%), legacy to corrected termination: 0.326 ± 0.125 to 0.889 ± 0.192.

Q-Learning minus hysteresis, paired by seed: any violation +16.67 ± 0.00 percentage points; mean power -0.0569 ± 0.0041 kW.

Q-Learning did not meet the joint benchmark of at least 1% lower mean power and no increase in either safety measure. Mean power saving was +2.64%, and the violating-step share changed by +4.72 percentage points.

SARSA minus hysteresis, paired by seed: any violation +16.67 ± 14.43 percentage points; mean power -0.0560 ± 0.0097 kW.

SARSA did not meet the joint benchmark of at least 1% lower mean power and no increase in either safety measure. Mean power saving was +2.60%, and the violating-step share changed by +4.06 percentage points.

## Monotone boosting (assumption)

| Termination | State | Controller | Any violation (%) | Violating steps (%) | Steps | Return / step | Power (kW) | Switching (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Legacy | 5-bin | Rule-based (no hysteresis) | 8.333 ± 0.000 | 0.181 ± 0.000 | 45.917 ± 0.000 | -3.425 ± 0.005 | 2.204 ± 0.006 | 2.778 ± 0.393 |
| Legacy | 5-bin | Rule-based | 8.333 ± 0.000 | 0.181 ± 0.000 | 45.917 ± 0.000 | -3.431 ± 0.006 | 2.208 ± 0.007 | 1.304 ± 0.098 |
| Legacy | 5-bin | Q-Learning | 33.333 ± 8.333 | 0.894 ± 0.291 | 37.861 ± 3.002 | -3.427 ± 0.007 | 2.205 ± 0.009 | 18.447 ± 3.193 |
| Legacy | 5-bin | SARSA | 30.556 ± 9.623 | 0.837 ± 0.366 | 37.667 ± 4.032 | -3.436 ± 0.006 | 2.204 ± 0.009 | 35.319 ± 10.021 |
| Legacy | Safety | Rule-based (no hysteresis) | 8.333 ± 0.000 | 0.181 ± 0.000 | 45.917 ± 0.000 | -3.425 ± 0.005 | 2.204 ± 0.006 | 2.778 ± 0.393 |
| Legacy | Safety | Rule-based | 8.333 ± 0.000 | 0.181 ± 0.000 | 45.917 ± 0.000 | -3.431 ± 0.006 | 2.208 ± 0.007 | 1.304 ± 0.098 |
| Legacy | Safety | Q-Learning | 25.000 ± 0.000 | 0.627 ± 0.011 | 39.861 ± 0.728 | -3.425 ± 0.011 | 2.203 ± 0.008 | 17.726 ± 8.834 |
| Legacy | Safety | SARSA | 19.444 ± 19.245 | 0.544 ± 0.628 | 41.556 ± 7.554 | -3.443 ± 0.002 | 2.203 ± 0.008 | 51.304 ± 13.132 |
| Serious | 5-bin | Rule-based (no hysteresis) | 8.333 ± 0.000 | 0.333 ± 0.000 | 50.000 ± 0.000 | -3.419 ± 0.005 | 2.203 ± 0.006 | 2.778 ± 0.393 |
| Serious | 5-bin | Rule-based | 8.333 ± 0.000 | 0.333 ± 0.000 | 50.000 ± 0.000 | -3.425 ± 0.006 | 2.207 ± 0.007 | 1.304 ± 0.098 |
| Serious | 5-bin | Q-Learning | 33.333 ± 8.333 | 5.667 ± 1.014 | 50.000 ± 0.000 | -3.428 ± 0.009 | 2.206 ± 0.008 | 18.821 ± 11.293 |
| Serious | 5-bin | SARSA | 25.000 ± 0.000 | 5.222 ± 2.175 | 50.000 ± 0.000 | -3.432 ± 0.007 | 2.205 ± 0.007 | 26.361 ± 12.296 |
| Serious | Safety | Rule-based (no hysteresis) | 8.333 ± 0.000 | 0.333 ± 0.000 | 50.000 ± 0.000 | -3.419 ± 0.005 | 2.203 ± 0.006 | 2.778 ± 0.393 |
| Serious | Safety | Rule-based | 8.333 ± 0.000 | 0.333 ± 0.000 | 50.000 ± 0.000 | -3.425 ± 0.006 | 2.207 ± 0.007 | 1.304 ± 0.098 |
| Serious | Safety | Q-Learning | 22.222 ± 12.729 | 3.222 ± 0.694 | 50.000 ± 0.000 | -3.431 ± 0.005 | 2.204 ± 0.005 | 21.259 ± 3.539 |
| Serious | Safety | SARSA | 8.333 ± 0.000 | 1.167 ± 0.577 | 50.000 ± 0.000 | -3.432 ± 0.011 | 2.202 ± 0.003 | 31.463 ± 22.802 |

Hysteresis Steps, legacy to corrected termination: 45.917 ± 0.000 to 50.000 ± 0.000.

Hysteresis Any violation (%), legacy to corrected termination: 8.333 ± 0.000 to 8.333 ± 0.000.

Hysteresis Violating steps (%), legacy to corrected termination: 0.181 ± 0.000 to 0.333 ± 0.000.

Q-Learning minus hysteresis, paired by seed: any violation +13.89 ± 12.73 percentage points; mean power -0.0022 ± 0.0039 kW.

Q-Learning did not meet the joint benchmark of at least 1% lower mean power and no increase in either safety measure. Mean power saving was +0.10%, and the violating-step share changed by +2.89 percentage points.

SARSA minus hysteresis, paired by seed: any violation +0.00 ± 0.00 percentage points; mean power -0.0050 ± 0.0045 kW.

SARSA did not meet the joint benchmark of at least 1% lower mean power and no increase in either safety measure. Mean power saving was +0.23%, and the violating-step share changed by +0.83 percentage points.
