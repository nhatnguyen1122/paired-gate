"""E2 across seeds (no API calls): per-run 90% paired bootstrap CI of the test gain
(final vs initial skill, same 300 test items, both run_id 0) and a per-gate summary."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skillopt.evaluation import paired_gate as pg  # noqa: E402
from skillopt.experiments import common as C  # noqa: E402


def load(d):
    return pg.outcomes_by_id([json.loads(line) for line in open(os.path.join(d, "results.jsonl")) if line.strip()])


def main():
    R = C.RESULTS
    runs = list(csv.DictReader(open(os.path.join(R, "e2_runs.csv"))))
    init = load(os.path.join(C.RUNS, "e0", "test", "run0"))
    ids = list(init)
    rng = np.random.default_rng(1)
    rows = []
    for r in sorted(runs, key=lambda r: (r["gate"] != "strict", int(r["seed"]))):
        sfx = "" if r["seed"] == "42" else f"_s{r['seed']}"
        fin = load(os.path.join(C.RUNS, f"e2_test_{r['gate']}{sfx}"))
        d = np.array([fin[i] - init[i] for i in ids], float)
        boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
        lo, hi = np.quantile(boots, [0.05, 0.95])
        rows.append({"gate": r["gate"], "seed": int(r["seed"]), "accepted_edits": int(r["accepted_edits"]),
                     "steps": int(r["steps"]), "target_rollouts": int(r["target_rollouts"]),
                     "rollouts_sel_candidate": int(r["rollouts_sel_candidate"]),
                     "rollouts_sel_incumbent": int(r["rollouts_sel_incumbent"]),
                     "final_test_acc": float(r["final_test_acc"]), "test_gain": float(d.mean()),
                     "ci90_lo": float(lo), "ci90_hi": float(hi), "final_skill_len": int(r["final_skill_len"]),
                     "usd": float(r["usd"])})
    with open(os.path.join(R, "e2_runs_ci.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = []
    for g in ("strict", "paired_bayes"):
        xs = [r for r in rows if r["gate"] == g]
        gains = np.array([r["test_gain"] for r in xs])
        summ.append({"gate": g, "n_seeds": len(xs), "mean_test_gain": gains.mean(), "min_test_gain": gains.min(),
                     "max_test_gain": gains.max(), "mean_accepted_edits": np.mean([r["accepted_edits"] for r in xs]),
                     "mean_target_rollouts": np.mean([r["target_rollouts"] for r in xs]),
                     "mean_final_skill_len": np.mean([r["final_skill_len"] for r in xs]),
                     "total_usd": sum(r["usd"] for r in xs)})
    with open(os.path.join(R, "e2_seeds_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    for r in rows + summ:
        print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
