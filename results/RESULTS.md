# Paired-gate study: results

Results of the experiments described in the [README](../README.md). All numbers come from the files in this folder; `cost_by_experiment.csv` is the full ledger summary.

## Models, date and settings

- Run date: **2026-11-26**, via the Mistral API.
- **Target:** `ministral-8b-latest` (Ministral 3 8B, $0.15 / $0.15 per 1M input / output tokens).
  - It scored 62.5% on the 40-item probe, inside the required 40–85% band, so `mistral-small` was not needed.
  - Temperature **0.7**, max 1024 tokens.
- **Optimizer:** `mistral-large-latest` (Mistral Large 4, $0.70 / $2.10 per 1M).
  - Temperature 0.7.
  - This model always reasons and rejects `reasoning_effort: none`, so we send `low`.
  - Its `content` comes back as thinking + text chunks; only the text is kept.
  - `mistral-medium-latest` was skipped because its model card publishes no price.
- Prices are from the docs.mistral.ai model cards, read 2026-11-26.
- Splits (seed 0):
  - pool = SkillOpt val (200 items);
  - test = 300 items sampled from SkillOpt test;
  - train batches of 20 from train.
- Environment fix: on this machine IPv6 to `api.mistral.ai` stalls every new connection for 10–150 s. `llm_cache` therefore resolves IPv4 only (turn this off with `SKILLOPT_FORCE_IPV4=0`).

## Tokens and cost (real API calls only; cache hits are free)

| Experiment | API calls | Prompt tok | Completion tok | USD |
|---|---:|---:|---:|---:|
| Setup (probe, smoke tests, rate test) | 393 | 0.51 M | 0.025 M | 0.17 |
| E0 (2 runs × (200 pool + 300 test); 40 items were already cached) | 960 | 1.18 M | 0.010 M | 0.18 |
| E1 generate (6 batches × 4 samples) | 235 | 0.85 M | 0.189 M | 0.92 |
| E1 score (24 candidates × 500 items) | 12 000 | 19.61 M | 0.120 M | 2.96 |
| E2 strict (training + final test) | 3 220 | 7.70 M | 0.211 M | 1.97 |
| E2 paired_bayes (training; final skill = initial, so the test eval is cached) | 2 803 | 4.96 M | 0.159 M | 1.36 |
| E2 diagnostic (14 rejected paired candidates on test) | 4 200 | 6.85 M | 0.042 M | 1.03 |
| E2 strict, seeds 43 + 44 (training + final test) | 6 450 | 17.61 M | 0.421 M | 4.36 |
| E2 paired_bayes, seeds 43 + 44 (training + final test) | 6 101 | 13.57 M | 0.370 M | 3.55 |
| Check of the refactored trainer (one paired step; stopped by the USD cap) | 208 | 0.37 M | 0.015 M | 0.12 |
| **Total** | **36 570** | **73.20 M** | **1.562 M** | **16.60** |

The extra E2 seeds were added after the first $8.58 had been spent. For them the cap was raised to `SKILLOPT_MAX_USD=18`, with the user's agreement to go over $10. They ran as 4 processes at once with 4 in-flight requests each: 16 in total, the same load as the first E2 pair (2 × 8). That is above the 8 in-flight requests the spec asks for, and it took 32 minutes.

A target rollout averages **1 238 tokens**: 1 228 prompt + 11 completion, measured in E0. That is about $0.00019 per rollout. The full ledger is in `results/cost_by_experiment.csv`.

## Results

### E0: base accuracy and run-to-run noise (initial skill)

| Split | Run 0 acc | Run 1 acc | Flip rate |
|---|---:|---:|---:|
| Pool (val, 200) | 60.5% | 62.5% | **8.0%** |
| Test (300) | 70.7% | 70.0% | **4.0%** |

A third, fresh run of the same skill on the pool (the E2 incumbent run) scored 63.5%. Noise is therefore far lower than the 26% flip rate assumed in the simulation. Even so, single pool scores of the same skill still spread over 3 pp (60.5% to 63.5%).

### E1: offline gate replay on 24 real SkillOpt candidates

- **Candidates:** 24, made by SkillOpt's own reflect → merge → rank → apply-patch stages, one step each from the initial skill.
- **Ground truth:** labels come from the 90% paired bootstrap CI of Δtest against incumbent run 0. That gives 2 good, 3 harmful and 19 unclear.
- **Replay:** 1 000 random pool orders for `paired_bayes`. "Rollouts" counts candidate rollouts per decision. Paired gates need the same number of incumbent rollouts again, while strict needs none (it reuses the recorded score).

Main analysis (as specified: paired gates against incumbent run 0, strict against the recorded score of run 1):

| Gate | P(acc \| good) | P(acc \| harmful) | P(acc \| unclear) | Cand. rollouts / decision | Inc. rollouts / decision | Expected accepts / 24 | Mean Δtest of accepted (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict | 1.00 | 0.33 | 0.89 | 200 | 0 | 20.0 | +0.23 |
| signtest α=0.05 | 1.00 | 0.00 | 0.58 | 200 | 200 | 13.0 | +1.10 |
| signtest α=0.2 | 1.00 | 0.33 | 0.89 | 200 | 200 | 20.0 | +0.23 |
| **paired_bayes η=0.05** | 0.97 | **0.19** | 0.80 | **118** | 118 | 17.7 | +0.44 |
| paired_bayes η=0.2 | 0.97 | 0.42 | 0.84 | 103 | 103 | 19.1 | +0.13 |

Sensitivity check (`results/e1_replay_swapped_runs.csv`). This swaps the roles of the two incumbent runs: paired gates use run 1, strict uses run 0's recorded score. No parameter was chosen from it.

| Gate | P(acc \| good) | P(acc \| harmful) | P(acc \| unclear) | Cand. rollouts | Mean Δtest of accepted (pp) |
|---|---:|---:|---:|---:|---:|
| strict | 1.00 | 0.67 | 1.00 | 200 | −0.20 |
| signtest α=0.05 | 1.00 | 0.00 | 0.21 | 200 | +2.61 |
| signtest α=0.2 | 1.00 | 0.00 | 0.58 | 200 | +1.10 |
| **paired_bayes η=0.05** | 0.94 | **0.03** | 0.58 | 122 | +1.01 |
| paired_bayes η=0.2 | 0.94 | 0.14 | 0.71 | 105 | +0.64 |

What E1 shows:
- `paired_bayes` with η=0.05 accepts harmful edits less often than `strict` (0.19 vs 0.33 in the main analysis; 0.03 vs 0.67 in the sensitivity check).
- It keeps almost all of the good edits (0.94–0.97) and uses about 40% fewer candidate rollouts per decision.
- Decided one at a time, it needs the same number of fresh incumbent rollouts again (about 236 per decision vs strict's 200). When the candidates share one incumbent, as here, the incumbent cost is shared: about 126 per decision vs 208 for strict (see the amortized cost below).
- η=0.2 is clearly worse than η=0.05, as in the simulation.
- `signtest α=0.05` is the most selective gate, but it always spends the full pool.
- **Pool/test shift:** candidates average +4.8 pp over incumbent run 0 on the pool, yet −0.5 pp on test. So every pool-based gate over-accepts "unclear" edits. The incumbent run used for pairing also matters a lot: run 0 (60.5%) is the lowest of three runs, and it inflates every candidate's fix count.

#### E1 amortized cost (shared incumbent)

All 24 candidates are decided against the same incumbent, so incumbent rollouts can be shared.
- For each of the 1 000 replay orders, one shared pool order is used for all 24 candidates, as in a real race.
- Total = sum of candidate rollouts + number of distinct pool items on which the incumbent was needed (the union over candidates). With one shared order that union is the longest prefix any candidate consumed.
- Strict and the full-pool sign test need one full incumbent run (200 items): strict for its recorded score, the sign test for pairing.
- Paired gates are against incumbent run 0, as in the main analysis. Results are in `results/e1_amortized_cost.csv`, and the amortized numbers are annotated on `e1_accept_by_label.png`.

| Gate | Candidate rollouts (24 decisions) | Distinct incumbent rollouts (min–max) | Total (min–max) | **Amortized / decision** | Unshared / decision |
|---|---:|---:|---:|---:|---:|
| strict | 4 800 | 200 | 5 000 | **208** | 200 |
| signtest α=0.05 | 4 800 | 200 | 5 000 | **208** | 400 |
| signtest α=0.2 | 4 800 | 200 | 5 000 | **208** | 400 |
| paired_bayes η=0.05 | 2 837 | 195 (20–200) | 3 032 (500–4 300) | **126** | 236 |
| paired_bayes η=0.2 | 2 481 | 182 (20–200) | 2 663 (500–3 760) | **111** | 207 |

"Unshared" is the per-decision cost when each decision pays for its own incumbent rollouts (strict reuses its recorded score, so it pays nothing extra). With a shared incumbent, `paired_bayes` η=0.05 costs about 40% fewer target rollouts than strict. In almost every order at least one candidate runs to the end of the pool, so the incumbent is still needed on nearly all 200 items. The saving comes from the candidates.

#### E1 with a third incumbent run

The E2 `paired_bayes` arm ran the initial skill freshly on the pool (run_id `incumbent-v0`, 63.5%, all 200 items). Its per-item outcomes were extracted to `results/e0/pool_run2_e2.json`, and the E1 replay was rerun with it as the incumbent: strict compares with that run's pool score, and the paired gates pair against it. Same orders, same parameters, no tuning. The test labels are unchanged (they come from the test split). Results: `results/e1_replay_run3.csv`.

| Gate | P(acc \| good) | P(acc \| harmful) | P(acc \| unclear) | Cand. rollouts / decision | Expected accepts / 24 | Mean Δtest of accepted (pp) |
|---|---:|---:|---:|---:|---:|---:|
| strict | 1.00 | 0.00 | 0.79 | 200 | 17.0 | +0.84 |
| signtest α=0.05 | 1.00 | 0.00 | 0.16 | 200 | 5.0 | +2.87 |
| signtest α=0.2 | 1.00 | 0.00 | 0.42 | 200 | 10.0 | +1.50 |
| **paired_bayes η=0.05** | 0.94 | **0.01** | 0.41 | 122 | 9.7 | +1.46 |
| paired_bayes η=0.2 | 0.94 | 0.06 | 0.60 | 104 | 13.4 | +0.89 |

Across the three incumbent runs of the initial skill (E0 run 0 at 60.5%, E0 run 1 at 62.5%, E2 run at 63.5%), each run is used both as strict's recorded score and as the paired incumbent. The table gives the mean, with the range in brackets. Per-run rows are in `results/e1_replay_by_incumbent_run.csv` and the summary in `results/e1_replay_across_incumbent_runs.csv`.

| Gate | P(acc \| good) | P(acc \| harmful) | P(acc \| unclear) |
|---|---|---|---|
| strict | 1.00 [1.00–1.00] | 0.33 [0.00–0.67] | 0.89 [0.79–1.00] |
| signtest α=0.05 | 1.00 [1.00–1.00] | 0.00 [0.00–0.00] | 0.32 [0.16–0.58] |
| signtest α=0.2 | 1.00 [1.00–1.00] | 0.11 [0.00–0.33] | 0.63 [0.42–0.89] |
| **paired_bayes η=0.05** | 0.95 [0.94–0.97] | **0.08 [0.01–0.19]** | 0.60 [0.41–0.80] |
| paired_bayes η=0.2 | 0.95 [0.94–0.97] | 0.21 [0.06–0.42] | 0.72 [0.60–0.84] |

What the three runs show:
- Every gate depends on which incumbent run it is compared with. A higher incumbent pool score makes all gates stricter.
- Strict is the most sensitive: it accepts between 0 and 2 of the 3 harmful edits.
- `paired_bayes` η=0.05 accepts harmful edits less often than strict for every incumbent run. It always keeps the good edits with probability ≥ 0.94.
- `signtest` α=0.05 never accepted a harmful edit, but it always uses the full pool and rejects more unclear edits.
- With only 3 harmful and 2 good candidates, these rates are coarse.

#### Test accuracy of the initial skill and every E1 candidate

Δtest is against initial-skill test run 0 (70.7%), with a 90% paired bootstrap CI. The initial skill is 104 characters long. Data: `results/e1_test_accuracy.csv`.

| Skill | Length (chars) | Pool acc | Test acc | Δtest (pp) | 90% CI (pp) | Label |
|---|---:|---:|---:|---:|---|---|
| initial skill, test run 0 | 104 | 60.5% | 70.7% | +0.0 | – | – |
| initial skill, test run 1 | 104 | 62.5% | 70.0% | – | – | – |
| b0_s0 | 1724 | 63.5% | 68.0% | -2.7 | -6.0 to +0.7 | unclear |
| b0_s1 | 1468 | 63.5% | 66.0% | -4.7 | -8.3 to -1.0 | harmful |
| b2_s0 | 1608 | 64.5% | 70.7% | +0.0 | -3.3 to +3.0 | unclear |
| b4_s0 | 2570 | 65.0% | 71.7% | +1.0 | -2.0 to +4.0 | unclear |
| b0_s2 | 1755 | 68.5% | 72.0% | +1.3 | -2.0 to +4.7 | unclear |
| b4_s1 | 2519 | 61.0% | 68.3% | -2.3 | -6.0 to +1.3 | unclear |
| b2_s1 | 2267 | 68.5% | 73.0% | +2.3 | -1.0 to +5.7 | unclear |
| b2_s2 | 1635 | 65.5% | 70.0% | -0.7 | -3.7 to +2.3 | unclear |
| b0_s3 | 1616 | 60.5% | 64.0% | -6.7 | -10.0 to -3.0 | harmful |
| b4_s2 | 2098 | 68.5% | 73.7% | +3.0 | -0.7 to +6.7 | unclear |
| b2_s3 | 1771 | 67.5% | 72.0% | +1.3 | -1.7 to +4.7 | unclear |
| b1_s0 | 2051 | 61.5% | 66.0% | -4.7 | -7.7 to -1.7 | harmful |
| b3_s0 | 1853 | 63.5% | 68.3% | -2.3 | -5.3 to +0.7 | unclear |
| b4_s3 | 1974 | 66.5% | 69.3% | -1.3 | -4.7 to +2.0 | unclear |
| b3_s1 | 2128 | 61.5% | 68.3% | -2.3 | -5.7 to +1.0 | unclear |
| b5_s0 | 2239 | 64.5% | 68.0% | -2.7 | -6.0 to +0.7 | unclear |
| b1_s1 | 2153 | 65.5% | 70.7% | +0.0 | -3.0 to +3.0 | unclear |
| b3_s2 | 1862 | 66.5% | 72.7% | +2.0 | -1.0 to +5.0 | unclear |
| b5_s1 | 1350 | 66.0% | 71.3% | +0.7 | -2.3 to +3.7 | unclear |
| b1_s2 | 2018 | 70.0% | 74.7% | +4.0 | +0.7 to +7.3 | good |
| b3_s3 | 2301 | 64.5% | 72.3% | +1.7 | -1.3 to +4.7 | unclear |
| b5_s2 | 1811 | 66.5% | 68.7% | -2.0 | -5.7 to +1.7 | unclear |
| b1_s3 | 2442 | 68.0% | 74.3% | +3.7 | +0.3 to +7.0 | good |
| b5_s3 | 1695 | 65.5% | 70.7% | +0.0 | -3.3 to +3.3 | unclear |

The pool accuracy of each initial-skill row is the pool run with the same index (E0 run 0 and run 1).

### E2: closed loop (cap of 3 000 target rollouts; seed 42 first, then seeds 43 and 44)

| Gate | Steps | Accepted edits | Target rollouts (train / sel-cand / sel-inc / baseline) | Tokens (prompt + compl.) | USD | Initial → final test | Test gain | Final skill length | Recorded sel. − test |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| strict | 13 | 2 (steps 1, 4) | 3 060 (260 / 2 600 / 0 / 200) | 7.30 M | 1.87 (+0.09 test) | 70.7% → 73.3% | **+2.7 pp** (90% CI −1.0 to +6.3) | 3 679 chars | 68.0 − 73.3 = −5.3 pp |
| paired_bayes η=0.05 | 14 | 0 | 3 080 (280 / 2 400 / 200 / 200) | 5.11 M | 1.36 | 70.7% → 70.7% | 0.0 | 104 chars | – |

Notes on E2:
- Both arms ran at the same time, so their first steps did not share cached optimizer samples. The candidates therefore differ between arms from step 1 on.
- The negative "selection − test" gap for strict comes from the pool being harder than test (see E0). It does not mean the score was underestimated.
- **Diagnostic** (`results/e2_candidates_test_paired_bayes.csv`): the 14 candidates that `paired_bayes` rejected were scored on test.
  - 12 of 14 had a negative test effect (mean −1.9 pp).
  - One (step 14, +4.3 pp, CI 1.0 to 7.7) was a real improvement that the gate missed. On the pool it showed fix 12 / break 11.
  - Strict's final skill gained +2.7 pp, but its CI includes 0.
- With seed 42 alone, E2 does not show that either gate is better in closed loop. Strict happened to accept two edits that helped on test. `paired_bayes` was conservative and protected the incumbent, but it missed the one clearly good candidate.

#### E2 with three seeds per gate (42, 43, 44)

Same config and cap; only `train.seed` changes (it sets the train batches and the paired pool order). All six final skills were scored on the same 300 test items with run_id 0, against initial-skill test run 0 (70.7%). The CI is a 90% paired bootstrap. USD covers training only. Data: `results/e2_runs.csv`, `results/e2_runs_ci.csv`, `results/e2_seeds_summary.csv`.

| Gate | Seed | Steps | Accepted edits | Target rollouts (sel-cand / sel-inc) | Final test | Test gain (90% CI) | Final skill length | USD |
|---|---:|---:|---:|---|---:|---|---:|---:|
| strict | 42 | 13 | 2 | 3 060 (2 600 / 0) | 73.3% | +2.7 pp (−1.0 to +6.3) | 3 679 | 1.87 |
| strict | 43 | 13 | 4 | 3 060 (2 600 / 0) | 70.7% | 0.0 pp (−3.3 to +3.3) | 5 174 | 2.04 |
| strict | 44 | 13 | 5 | 3 060 (2 600 / 0) | 73.3% | +2.7 pp (−1.0 to +6.3) | 7 647 | 2.08 |
| paired_bayes η=0.05 | 42 | 14 | 0 | 3 080 (2 400 / 200) | 70.7% | 0.0 pp (unchanged skill) | 104 | 1.36 |
| paired_bayes η=0.05 | 43 | 12 | 4 | 3 000 (1 670 / 890) | 75.0% | **+4.3 pp (+0.7 to +8.0)** | 5 629 | 1.67 |
| paired_bayes η=0.05 | 44 | 16 | 1 | 3 190 (2 270 / 400) | 70.7% | 0.0 pp (−3.3 to +3.3) | 2 193 | 1.68 |

| Gate | Mean test gain [min–max] | Mean accepted edits | Mean target rollouts | Mean final skill length | Training USD (3 seeds) |
|---|---|---:|---:|---:|---:|
| strict | +1.8 pp [0.0 – +2.7] | 3.7 | 3 060 | 5 500 | 5.99 |
| paired_bayes η=0.05 | +1.4 pp [0.0 – +4.3] | 1.7 | 3 090 | 2 642 | 4.72 |

What three seeds show:
- No final skill of either gate is worse than the initial skill on test. Both gates average a gain of about +1.5 pp, and the difference between them (0.3 pp) is far smaller than the seed-to-seed spread.
- Only one run has a 90% CI above zero: `paired_bayes` seed 43 (+4.3 pp, 4 accepted edits). No strict run does.
- Strict accepts about twice as many edits (3.7 vs 1.7 per run) for the same gain, so its skills grow about twice as long (5 500 vs 2 600 characters on average).
- `paired_bayes` spends 200–890 of its ~3 000 target rollouts per run on fresh incumbent rollouts, and its early rejections free the rest for more steps (12–16 vs 13). Its training runs cost about 20% less in USD ($1.57 vs $2.00 per run), mostly because they used fewer prompt tokens.
- With 3 seeds and 300 test items, E2 still cannot separate the two gates in closed loop.

Figures are in `results/figures/`:
- `e1_accept_by_label.png`: P(accept) by label for each gate, annotated with candidate rollouts per decision and amortized total target rollouts per decision.
- `e1_scatter_gain_vs_accept.png`: Δtest against P(accept), for strict and paired_bayes.
- `e2_gain_vs_rollouts.png`: test gain with 90% CI for each E2 run, grouped by seed and annotated with accepted edits and target rollouts.

## Limitations

- One benchmark (SearchQA), one target model (`ministral-8b-latest`) and one optimizer.
- E1 has only 24 candidates. Most are "unclear" on a 300-item test set, so the per-label rates (n = 2 good, 3 harmful) are very coarse.
- E2 used three seeds per gate (42, 43, 44). That is enough to see the spread, but not to separate the gates.
- The pool is small (200 items). Every gate's E1 replay depends on which incumbent run is used (see the three-run table). Strict varies most.
- Run-to-run noise at temperature 0.7 is small here (4–8% flips), while the val (pool) and test splits differ systematically. Under these conditions the pool signal itself is weak, whatever the gate.
- Gate parameters are the specification's defaults. None were tuned on test.

