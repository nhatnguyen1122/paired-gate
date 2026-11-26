"""E1 offline extras (no API calls).

1. Amortized cost: all 24 candidates are decided against the SAME incumbent, so
   incumbent rollouts are shared. For each of the 1,000 replay orders, ONE pool
   order is used for all 24 candidates (as in a real race). Total target rollouts
   to decide all 24 = sum of candidate rollouts + number of DISTINCT pool items on
   which the incumbent was needed (union over candidates). Strict needs one full
   incumbent run (200) for its recorded score; the full-pool sign test also needs
   the incumbent on all 200 items.
2. Acceptance over the three incumbent runs of the initial skill on the pool
   (E0 run 0, E0 run 1, E2 paired_bayes arm), each used both as strict's recorded
   score and as the paired incumbent.
3. Test accuracy of the initial skill and of every E1 candidate.

Uses the same orders (np.random.default_rng(0) permutations of the pool) and
gate parameters as e1_replay.py; nothing is tuned.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skillopt.evaluation import paired_gate as pg  # noqa: E402
from skillopt.experiments import common as C  # noqa: E402

N_ORDERS = 1000
R = C.RESULTS
GATES = ["strict", "signtest_alpha0.05", "signtest_alpha0.2", "paired_bayes_eta0.05", "paired_bayes_eta0.2"]
LABELS = ["good", "harmful", "unclear"]


def write_csv(name, rows):
    with open(os.path.join(R, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv(name):
    with open(os.path.join(R, name)) as f:
        return list(csv.DictReader(f))


def amortized(pair_run=0):
    inc = json.load(open(os.path.join(R, "e0", f"pool_run{pair_run}.json")))
    scores = json.load(open(os.path.join(R, "e1_scores.json")))
    pool_ids = list(json.load(open(os.path.join(R, "e0", "pool_run0.json"))).keys())
    n = len(pool_ids)
    rng = np.random.default_rng(0)
    orders = [list(rng.permutation(pool_ids)) for _ in range(N_ORDERS)]
    cids = list(scores.keys())
    K = len(cids)
    rows = []
    for g in GATES:
        if g.startswith("paired_bayes"):
            cfg = pg.PairedGateConfig(eta=float(g.split("eta")[1]))
            cand_tot, inc_tot = [], []
            for o in orders:
                used = [pg.replay_sequential(scores[c]["pool"], inc, o, cfg)[1] for c in cids]
                cand_tot.append(sum(used))
                # same order for every candidate -> the union of the incumbent
                # prefixes is the longest prefix
                inc_tot.append(len(set().union(*[set(o[:u]) for u in used])))
            cand_tot, inc_tot = np.array(cand_tot, float), np.array(inc_tot, float)
        else:  # strict and full-pool sign test: every candidate uses the full pool
            cand_tot = np.full(N_ORDERS, K * n, float)
            inc_tot = np.full(N_ORDERS, n, float)
        tot = cand_tot + inc_tot
        rows.append({"gate": g, "n_candidates": K, "cand_rollouts_total": float(cand_tot.mean()),
                     "inc_rollouts_distinct": inc_tot.mean(), "inc_rollouts_distinct_min": inc_tot.min(),
                     "inc_rollouts_distinct_max": inc_tot.max(), "total_rollouts": tot.mean(),
                     "total_rollouts_min": tot.min(), "total_rollouts_max": tot.max(),
                     "amortized_per_decision": tot.mean() / K,
                     # per decision without sharing: strict reuses its recorded score (0 incumbent);
                     # paired gates rerun the incumbent on every item they consume
                     "unshared_per_decision": float(cand_tot.mean() / K * (1 if g == "strict" else 2))})
    write_csv("e1_amortized_cost.csv", rows)
    return rows


def across_incumbent_runs():
    """Per incumbent run r: paired gates paired against r, strict compared with r's score."""
    files = {"main": "e1_replay.csv", "swapped": "e1_replay_swapped_runs.csv", "run3": "e1_replay_run3.csv"}
    tabs = {k: {r["gate"]: r for r in read_csv(v)} for k, v in files.items()}
    # main: pair 0 / strict 1; swapped: pair 1 / strict 0; run3: pair 2 / strict 2
    src = {0: {"strict": "swapped", "paired": "main"},
           1: {"strict": "main", "paired": "swapped"},
           2: {"strict": "run3", "paired": "run3"}}
    names = {0: "E0 run 0 (60.5%)", 1: "E0 run 1 (62.5%)", 2: "E2 run (63.5%)"}
    per_run, summ = [], []
    for g in GATES:
        vals = {lab: [] for lab in LABELS}
        for r in (0, 1, 2):
            t = tabs[src[r]["strict" if g == "strict" else "paired"]][g]
            row = {"gate": g, "incumbent_run": names[r]}
            for lab in LABELS:
                v = float(t[f"p_accept_{lab}"])
                vals[lab].append(v)
                row[f"p_accept_{lab}"] = v
            row["mean_cand_rollouts_per_decision"] = float(t["mean_cand_rollouts_per_decision"])
            per_run.append(row)
        s = {"gate": g}
        for lab in LABELS:
            s[f"{lab}_mean"] = float(np.mean(vals[lab]))
            s[f"{lab}_min"] = float(np.min(vals[lab]))
            s[f"{lab}_max"] = float(np.max(vals[lab]))
        summ.append(s)
    write_csv("e1_replay_by_incumbent_run.csv", per_run)
    write_csv("e1_replay_across_incumbent_runs.csv", summ)
    return per_run, summ


def test_table():
    meta = json.load(open(os.path.join(R, "e1_replay_meta.json")))
    t1 = json.load(open(os.path.join(R, "e0", "test_run1.json")))
    rows = [{"skill": "initial (E0 test run 0, the label reference)", "skill_len": "",
             "pool_acc": meta["incumbent_pool_acc_run0"], "test_acc": meta["incumbent_test_acc_run0"],
             "test_gain_pp": 0.0, "ci90_lo_pp": "", "ci90_hi_pp": "", "label": ""},
            {"skill": "initial (E0 test run 1)", "skill_len": "",
             "pool_acc": meta["incumbent_pool_acc_run1"], "test_acc": sum(t1.values()) / len(t1),
             "test_gain_pp": "", "ci90_lo_pp": "", "ci90_hi_pp": "", "label": ""}]
    for r in read_csv("e1_replay_per_candidate.csv"):
        rows.append({"skill": r["cand_id"], "skill_len": int(r["skill_len"]), "pool_acc": float(r["pool_acc"]),
                     "test_acc": float(r["test_acc"]), "test_gain_pp": 100 * float(r["test_gain"]),
                     "ci90_lo_pp": 100 * float(r["ci_lo"]), "ci90_hi_pp": 100 * float(r["ci_hi"]),
                     "label": r["label"]})
    write_csv("e1_test_accuracy.csv", rows)
    return rows


if __name__ == "__main__":
    a = amortized()
    for r in a:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
    _, s = across_incumbent_runs()
    for r in s:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    t = test_table()
    print(len(t), "rows in e1_test_accuracy.csv")
