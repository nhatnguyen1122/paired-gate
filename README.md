# SkillOpt with a paired acceptance gate (branch `paired-gate`)

> This section describes the fork. The upstream SkillOpt README follows below the line.

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
export MISTRAL_API_KEY=...                    # keep it out of the repo
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
python scripts/gate/probe_models.py --models ministral-8b-latest   # pick the target
python scripts/gate/e0_flip_rate.py                                # E0: accuracy and run-to-run noise
python scripts/gate/e1_generate_candidates.py --batches 0,1,2,3,4,5  # E1: 24 SkillOpt candidates
python scripts/gate/e1_score_candidates.py                         # E1: score on pool and test
python scripts/gate/e1_replay.py                                   # E1: gate replay (offline)
python scripts/gate/e1_extra.py                                    # E1: amortized cost, 3 incumbents (offline)
for s in 42 43 44; do for g in strict paired_bayes; do             # E2: closed loop
  python scripts/gate/e2_closed_loop.py --gates $g --seed $s; done; done
python scripts/gate/e2_candidates_on_test.py --gate paired_bayes   # E2: rejected candidates on test
python scripts/gate/e2_seeds_summary.py                            # E2: CIs and summary (offline)
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

---

# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

*Train agent skills like you train neural networks — with epochs, (mini-)batchsize, learning rates, and validation gates — but without touching model weights.*

[![Project Page](https://img.shields.io/badge/Project%20Page-SkillOpt-8dbb3c)](https://microsoft.github.io/SkillOpt/) [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2605.23904) [![Project Video](https://img.shields.io/badge/Project%20Video-Watch%20Demo-ff0000)](https://youtu.be/JUBMDTCiM0M) [![PyPI](https://img.shields.io/badge/PyPI-skillopt-green.svg)](https://pypi.org/project/skillopt/) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <a href="https://trendshift.io/repositories/38498?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-38498" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/repositories/38498" alt="microsoft%2FSkillOpt | Trendshift" width="250" height="55"/></a>
  <a href="https://trendshift.io/repositories/38498?utm_source=trendshift-badge&utm_medium=badge&utm_campaign=badge-trendshift-38498" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/38498/weekly?language=Python" alt="microsoft%2FSkillOpt | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://www.credly.com/badges/d9fd6cd8-e310-4e9b-9f63-ed7a3383fc85/public_url" target="_blank" rel="noopener noreferrer"><img src="skillopt-assets/hackathon-2026-executive-1st.png" alt="Microsoft Global Hackathon 2026 – 1st Place Executive Challenge Winner" width="110" height="110"/></a>
</p>

> 📖 **For installation, data preparation, training/eval commands, configuration, and framework internals, start with the versioned [SkillOpt documentation](https://github.com/microsoft/SkillOpt/blob/main/docs/index.md). A concise rendered overview is available in the [Documentation & Reproduction Guide](https://microsoft.github.io/SkillOpt/docs/guideline.html), and longer-form engineering analysis appears on the [Technical Blog](https://microsoft.github.io/SkillOpt/blog/). We also maintain a [Changelog](CHANGELOG.md) for released and unreleased changes.**

---

## News 🔥🔥🔥
- **[2026-11]** 🎤 **SkillOpt is accepted to NeurIPS 2026 as a Spotlight!** See you in Sydney!
- **[2026-11]** 🏆 **SkillOpt won 1st Place in the Executive Challenge at the Microsoft Global Hackathon 2026** ([badge](https://www.credly.com/badges/d9fd6cd8-e310-4e9b-9f63-ed7a3383fc85/public_url), issued by The Microsoft Garage).
- **[2026-07-24]** 📰 **SkillOpt in the news.** Read the official [Microsoft Research feature](https://www.microsoft.com/en-us/research/blog/skillopt-agent-skills-as-trainable-parameters/), along with recent coverage from [VentureBeat](https://venturebeat.com/orchestration/microsofts-open-source-skillopt-automatically-upgrades-ai-agent-skills-without-touching-model-weights), [Synced (机器之心)](https://mp.weixin.qq.com/s/pMlyj3a3KOh8L7cIHClRXA), [Flowtivity](https://flowtivity.ai/blog/microsoft-skillopt-train-ai-agent-skills/), and [The Decoder](https://the-decoder.com/microsofts-skillopt-boosts-gpt-5-5-by-using-nothing-but-a-trained-markdown-file/).
- **[2026-07-02]** 🚀 **SkillOpt [v0.2.0](https://github.com/microsoft/SkillOpt/releases/tag/v0.2.0) is out on [PyPI](https://pypi.org/project/skillopt/)!** Headline feature: **SkillOpt-Sleep**, a nightly offline self-evolution engine (harvest → mine → replay → consolidate behind a held-out validation gate), now shipped as the `skillopt-sleep` CLI. It also includes experimental multi-objective, replay, and dream-rollout controls; the main CLI keeps conservative defaults and does not expose every experiment-harness control as a flag. The release source adds integration shells for **Claude Code, Codex, Copilot, and Devin**, plus an **OpenClaw reference adaptation**; these plugin/MCP files live in the repository rather than the PyPI wheel. It also adds SearchQA split materialization, Windows robustness, and hardened JSON parsing. See the [release notes](https://github.com/microsoft/SkillOpt/releases/tag/v0.2.0) for full release details and contributor acknowledgements.
- **[2026-06-15]** 😴 **SkillOpt-Sleep (preview)** — a nightly offline self-evolution companion for local coding agents (Claude Code / Codex / Copilot): review past sessions, replay recurring tasks, and consolidate validated skills behind a held-out gate. See **[`docs/sleep/README.md`](docs/sleep/README.md)** for what it is, how to use it, and results.
- **[2026-06-03]** 🎉 **[gbrain](https://github.com/garrytan/gbrain), [gbrain-evals](https://github.com/garrytan/gbrain-evals/blob/main/docs/benchmarks/2026-06-03-skillopt.md), and [darwin-skill](https://github.com/alchaincyf/darwin-skill) have all integrated SkillOpt.**
- **[2026-06-02]** 🎉 **SkillOpt [v0.1.0](https://github.com/microsoft/SkillOpt/releases/tag/v0.1.0) is now available on [PyPI](https://pypi.org/project/skillopt/)!** Install with `pip install skillopt`. This initial release includes the full training loop (rollout → reflect → aggregate → select → update → evaluate), multi-backend support (OpenAI / Azure / Claude / Qwen / MiniMax), six built-in benchmarks, and WebUI dashboard.

---

## Overview

Modern agent skills are usually hand-crafted, generated one-shot by a strong
LLM, or evolved through loosely controlled self-revision — none of which
behaves like a deep-learning optimizer for the skill itself, and none of
which reliably improves over its starting point under feedback.

**SkillOpt treats the skill document as the trainable state of a frozen
agent**, and trains it with the discipline that makes weight-space
optimization reproducible. A separate optimizer model turns scored rollouts
into bounded add / delete / replace edits on a single skill document; in the
default paper-style path, a candidate edit is accepted only when it strictly
improves a held-out validation score. A textual learning-rate budget, a rejected-edit buffer,
and an epoch-wise slow / meta update make skill training stable while
adding **zero inference-time model calls** at deployment.

The deployed artifact is a compact `best_skill.md` (typically 300–2,000
tokens) that runs against the unchanged target model. Across **six
benchmarks, seven target models, and three execution harnesses** (direct
chat, Codex CLI, Claude Code CLI), SkillOpt is best or tied-best on **all
52 evaluated (model, benchmark, harness) cells** and on GPT-5.5 lifts the
average no-skill accuracy by **+23.5 points in direct chat, +24.8 inside
the Codex agentic loop, and +19.1 inside Claude Code**. Optimized skill
artifacts transfer across model scales, between Codex and Claude Code
harnesses, and to nearby benchmarks without further optimization.

For the full method, ablations, and per-cell results see the [paper](https://arxiv.org/abs/2605.23904); for a visual walkthrough of the loop see the [project page](https://microsoft.github.io/SkillOpt/); for deeper API / backend / benchmark docs see [`docs/`](docs/).

## 🎬 Demo Video

https://github.com/user-attachments/assets/eb12d3bc-371c-467f-904d-91b61f339ed7

<p align="center">
  <a href="https://youtu.be/JUBMDTCiM0M"><b>▶ Watch the full demo on YouTube</b></a>
</p>

---

## Extensibility & WebUI

### Adding a new backend

A backend = a chat / exec target (e.g. `openai_chat`, `claude_chat`,
`qwen_chat`, `minimax_chat`, `copilot_chat`, `openai_compatible`, `codex_exec`,
`claude_code_exec`, `cursor_exec`, `copilot_exec`). If a provider implements the OpenAI Chat Completions
protocol, try the built-in `openai_compatible` backend before adding code. See
[`docs/guide/new-backend.md`](docs/guide/new-backend.md) for the full
contract. Chat backends add a `skillopt/model/<name>_backend.py` module;
target-only exec backends use the shared harness in `codex_harness.py`.
Both register through `common.py`, `backend_config.py`, and
`skillopt/model/__init__.py`.

### Adding a new benchmark

A benchmark = a `skillopt/envs/<name>/` package with an adapter, a data loader,
a scored rollout helper, a YAML config, and optionally an initial seed skill.
See
[`docs/guide/new-benchmark.md`](docs/guide/new-benchmark.md) for the full
contract; the simplest reference is `skillopt/envs/searchqa/`.

### WebUI

Launch the monitoring dashboard (optional):

```bash
pip install -e ".[webui]"
python -m skillopt_webui.app
```

| Flag | Default | Description |
|---|---|---|
| `--port` | 7860 | Server port |
| `--host` | `127.0.0.1` | Bind address |
| `--share` | off | Create a public Gradio share link |

The default host is loopback-only. Exposing the dashboard (`--host 0.0.0.0` or `--share`) requires
`--auth-user`/`--auth-pass` (or `SKILLOPT_WEBUI_USER`/`SKILLOPT_WEBUI_PASS`).

---

## Citation

```bibtex
@article{yang2026skillopt,
  title={Skillopt: Executive strategy for self-evolving agent skills},
  author={Yang, Yifan and Gong, Ziyang and Huang, Weiquan and Yang, Qihao and Zhou, Ziwei and Huang, Zisu and Li, Yan and Gao, Xuemei and Dai, Qi and Liu, Bei and others},
  journal={arXiv preprint arXiv:2605.23904},
  year={2026}
}
```
