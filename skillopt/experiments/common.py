"""Shared helpers for the paired-gate experiments on SearchQA via Mistral."""
from __future__ import annotations

import json
import os
import random

from skillopt.model import (
    get_token_summary,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_target_backend,
    set_target_deployment,
)
from skillopt.model import llm_cache
from skillopt.evaluation.paired_gate import outcomes_by_id

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_DIR = os.path.join(ROOT, "data", "searchqa_split")
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(ROOT, "runs")
INITIAL_SKILL = os.path.join(ROOT, "skillopt", "envs", "searchqa", "skills", "initial.md")


def setup_backends() -> None:
    set_optimizer_backend("openai_compatible")
    set_target_backend("openai_compatible")
    set_optimizer_deployment(os.environ.get("OPTIMIZER_OPENAI_COMPATIBLE_MODEL", "mistral-large-latest"))
    set_target_deployment(os.environ.get("TARGET_OPENAI_COMPATIBLE_MODEL", "ministral-8b-latest"))


def load_split(name: str) -> list[dict]:
    with open(os.path.join(SPLIT_DIR, name, "items.json")) as f:
        return json.load(f)


def pool_items(n: int = 200) -> list[dict]:
    """Selection pool = SkillOpt val split (first n items, file order)."""
    return load_split("val")[:n]


def test_items(n: int = 300, seed: int = 0) -> list[dict]:
    """Test = n items sampled with a fixed seed from SkillOpt's test split."""
    items = load_split("test")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in idx]


def initial_skill() -> str:
    with open(INITIAL_SKILL) as f:
        return f.read()


def _rollout_tokens() -> tuple[int, int, int]:
    s = get_token_summary().get("rollout", {})
    return s.get("calls", 0), s.get("prompt_tokens", 0), s.get("completion_tokens", 0)


def rollout(adapter, items: list[dict], skill: str, out_dir: str, run_id) -> tuple[dict, list[dict], dict]:
    """Roll ``skill`` out on ``items`` under ``run_id``.

    Returns ({id: hard}, raw results, token stats). ``out_dir`` must be unique per
    (skill, run_id) because SkillOpt's run_batch resumes from ``results.jsonl``.
    """
    c0, p0, q0 = _rollout_tokens()
    with llm_cache.run_id_scope(run_id):
        res = adapter.rollout(items, skill, out_dir)
    c1, p1, q1 = _rollout_tokens()
    n_err = sum(1 for r in res if not r.get("agent_ok"))
    stats = {"calls": c1 - c0, "prompt_tokens": p1 - p0, "completion_tokens": q1 - q0,
             "n": len(res), "n_agent_error": n_err}
    if n_err:
        print(f"  [warn] {n_err}/{len(res)} rollouts failed before an agent response in {out_dir}")
    return outcomes_by_id(res), res, stats


def make_adapter(workers: int = 8, max_completion_tokens: int = 1024):
    from skillopt.envs.searchqa.adapter import SearchQAAdapter

    return SearchQAAdapter(split_dir=SPLIT_DIR, split_mode="split_dir", workers=workers,
                           max_completion_tokens=max_completion_tokens, exec_timeout=120)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
