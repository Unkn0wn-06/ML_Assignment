## 2026-11-21

- **Training diagnostics.** Paired each tabular training episode with fixed setpoint at the same initial row. Added 40-episode reward differences, seed-level confidence intervals, Q-table coverage and periodic greedy checks on 12 held-in starts.
- **Controller bounds.** Added maximum and minimum cooling to evaluation and controller comparisons. The held-out report compares both extremes on matched starts and quantifies power–safety trade-offs.
- **Dashboard.** The viewer prioritises high-violation full-load replays and opens with pausable autoplay. Resized the timeline and legends for a 1000 px window, simplified sensitivity diagnostics and rerouted the schematic cooling loop.
