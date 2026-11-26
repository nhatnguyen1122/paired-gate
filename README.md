# SkillOpt with a paired acceptance gate


SkillOpt keeps an edited skill only if its score on the selection split is **strictly greater** than the recorded score of the current skill. With 0/1 outcomes from a stochastic agent, that comparison is noisy. This branch adds paired gates that compare candidate and incumbent **item by item** on the same selection items, plus a Mistral API setup with a cost cap to study them.

Results: [`results/RESULTS.md`](results/RESULTS.md).

## Gates

Set `evaluation.gate_type`:

| `gate_type` | Rule |
|---|---|
| `strict` (default) | Upstream behaviour, unchanged. |
| `signtest` | Full pool; accept if the one-sided exact sign test on fix vs break has p ≤ `signtest_alpha`. |
| `paired_bayes` | Sequential race against a fresh incumbent run, in batches of `paired_b` items. Reject early when P(gain > `paired_d_useful`) < `paired_q_fut`. Accept only after `paired_acc_min_frac` of the pool, if P(gain > 0) ≥ `paired_q_acc` and P(harm) ≤ `paired_eta`. |

- *fix* = only the candidate solves the item, *break* = only the incumbent does. The posterior is Dirichlet(1,1,1) over (fix, break, same).
- Incumbent outcomes come from a fresh run under their own `run_id`, filled lazily on the items the race uses, and reset after every acceptance.
- Defaults: `paired_b=10`, `paired_q_acc=0.8`, `paired_eta=0.05`, `paired_eps_harm=0.01`, `paired_d_useful=0.005`, `paired_q_fut=0.2`, `paired_acc_min_frac=0.5`, `signtest_alpha=0.05`.
- `evaluation.max_target_rollouts` stops training once that many target rollouts were used (0 = no cap).
- The rules are ported verbatim from the reference simulation [`sim/sim.py`](sim/sim.py). [`tests/test_paired_gate.py`](tests/test_paired_gate.py) checks that the port gives the same decisions and rollout counts on 300 random cases.
- Each step logs a `gate` record (decision, fix/break/same, posterior, rollouts) in `step_record.json` and writes `paired_gate.json`.

## Running on the Mistral API

```bash
conda create -n skillopt-gate python=3.11 -y && conda activate skillopt-gate
pip install -e '.[searchqa]' scipy matplotlib pandas pytest
python scripts/materialize_searchqa.py        # data/searchqa_split
export MISTRAL_API_KEY=...
source scripts/mistral_env.sh                 # both roles -> api.mistral.ai, cache + USD ledger on
python scripts/train.py --config configs/searchqa/e2_paired_bayes.yaml
```

`scripts/mistral_env.sh` sets the models (target `ministral-8b-latest`, optimizer `mistral-large-latest`) and turns on [`skillopt/model/llm_cache.py`](skillopt/model/llm_cache.py). That module is active only when `SKILLOPT_LLM_DIR` is set; otherwise the backend behaves as upstream. It provides:
- a disk cache keyed by `(model, temperature, max_tokens, messages, run_id)`, so runs with different `run_id`s are never merged;
- a USD ledger (`.llm_cache/ledger.jsonl`) shared by all processes, with a cap `SKILLOPT_MAX_USD` (default 10);
- at most `SKILLOPT_MAX_CONCURRENCY` (default 8) requests in flight per process;
- IPv4-only DNS, because IPv6 to the API stalled on the test machine (disable with `SKILLOPT_FORCE_IPV4=0`).

## Reproducing the study

All scripts read and write `results/`. Scripts marked offline make no API calls.

```bash
python -m pytest -q tests/test_paired_gate.py
python scripts/gate/probe_models.py --models ministral-8b-latest
python scripts/gate/e0_flip_rate.py
python scripts/gate/e1_generate_candidates.py --batches 0,1,2,3,4,5
python scripts/gate/e1_score_candidates.py
python scripts/gate/e1_replay.py
python scripts/gate/e1_extra.py
for s in 42 43 44; do for g in strict paired_bayes; do
  python scripts/gate/e2_closed_loop.py --gates $g --seed $s; done; done
python scripts/gate/e2_candidates_on_test.py --gate paired_bayes
python scripts/gate/e2_seeds_summary.py
python scripts/gate/cost_report.py && python scripts/make_figures.py
```

## Code layout

| What | Where |
|---|---|
| Gate rules | `skillopt/evaluation/paired_gate.py` |
| Trainer integration | `ReflACTTrainer._paired_evaluate` and step ⑥ EVALUATE in `skillopt/engine/trainer.py` |
| Config keys | `evaluation.*` in `skillopt/config.py` |
| Cache, ledger, Mistral backend | `skillopt/model/llm_cache.py`, `skillopt/model/openai_compatible_backend.py` |
| Configs | `configs/searchqa/mistral_small.yaml`, `e2_strict.yaml`, `e2_paired_bayes.yaml` |
| Experiment scripts | `skillopt/experiments/common.py`, `scripts/gate/`, `scripts/make_figures.py` |
| Reference simulation | `sim/` |

The upstream test suite passes except 2 tests in `tests/test_sleep_adopt_skill_subset.py`, which fail the same way on the unmodified upstream commit on macOS.


