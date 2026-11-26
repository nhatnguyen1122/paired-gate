# Simulation results

Run with: `python3 sim.py --which single,budget,paired`. Raw results are in `single.json`, `budget.json` and `paired.json`.

## Model

- **Tasks:** base logit $a_i\sim\mathcal N(0.6, 2.5^2)$, giving base accuracy 58%. Each agent run is an independent Bernoulli draw, so 26% of outcomes flip between two runs of the same skill.
- **Edits:** an edit touches 30% of tasks; on those tasks it shifts the logit by its mean effect $\mu$ plus $\mathcal N(0, 0.3^2)$.
- **Proposals:** 50% harmful ($\mu\approx-0.45$), 35% neutral, 15% good ($\mu\approx+0.45$, about +1.7 accuracy points).
- **True accuracy:** computed exactly on 3,000 test tasks.

## Experiment 1: one decision

One candidate against the base skill; 400 runs per cell. The "cost" columns include the incumbent's rollouts.

| n | true gain (pts) | strict | sign test α=0.05 | seq. freq. | seq. Bayes | cost strict | cost seq. Bayes |
|---|---|---|---|---|---|---|---|
| 140 | −2.34 | 0.217 | 0.013 | 0.003 | 0.060 | 280 | 156 |
| 140 | −1.16 | 0.347 | 0.025 | 0.005 | 0.092 | 280 | 167 |
| 140 | 0.00 | 0.450 | 0.037 | 0.013 | 0.122 | 280 | 177 |
| 140 | +1.13 | 0.550 | 0.055 | 0.022 | 0.188 | 280 | 185 |
| 140 | +2.23 | 0.655 | 0.083 | 0.040 | 0.250 | 280 | 193 |
| 140 | +4.30 | 0.792 | 0.193 | 0.115 | 0.398 | 280 | 190 |
| 400 | −2.35 | 0.180 | 0.007 | 0.003 | 0.033 | 800 | 274 |
| 400 | +2.23 | 0.800 | 0.163 | 0.098 | 0.415 | 800 | 400 |

`single.json` also has n = 40, where every gate is poor.

## Experiment 2: budget-matched optimization

$K=4$ candidates per step; each gate runs until the rollout budget is spent. Mean true gain in points ± s.e., 100 runs.

| n = 140, budget | strict | sign test α=0.2 | seq. freq. α=0.2 | seq. Bayes |
|---|---|---|---|---|
| 2.8k | 0.35 ± 0.21 | 0.35 ± 0.14 | 0.02 ± 0.10 | **0.56 ± 0.18** |
| 5.6k | 0.62 ± 0.25 | 0.67 ± 0.19 | −0.04 ± 0.13 | **0.88 ± 0.25** |
| 11.2k | 1.03 ± 0.31 | 1.18 ± 0.23 | 0.03 ± 0.16 | **1.34 ± 0.33** |
| 22.4k | 1.69 ± 0.36 | 1.87 ± 0.30 | 0.29 ± 0.21 | **2.24 ± 0.38** |

**Accepted edits at 22.4k (harmful / neutral / good):**
- strict: 0.83 / 1.34 / 1.75
- seq. Bayes: 2.39 / 4.33 / 3.48

**Other cases:**
- With n = 40, every gate stays within ±0.4 points, and strict is slightly negative at every budget.
- Accepting every edit loses 58 points after 400 steps.

## Paired re-run

200 runs; for each run index, every gate starts from the same initial world. The table shows the difference to strict, with a 95% bootstrap CI.

| budget | sign test α=0.2 | seq. Bayes η=0.05 | seq. Bayes η=0.2 |
|---|---|---|---|
| 5.6k | +0.12 [−0.24, 0.47] | +0.38 [−0.09, 0.85] | −0.05 [−0.68, 0.53] |
| 11.2k | +0.08 [−0.36, 0.51] | +0.31 [−0.31, 0.88] | −0.22 [−1.04, 0.56] |

**Notes**
- With η = 0.05, the harm cap is the binding constraint, so `q_acc` in {0.7, 0.8, 0.9} gives identical results.
- With η = 0.2, the gate accepts about 3× more edits (15.5 vs. 5.7 at 11.2k), and the final gain is lower.
