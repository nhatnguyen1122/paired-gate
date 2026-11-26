"""E1 step 3: offline replay of the gates on the real candidates (no API calls)."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skillopt.evaluation import paired_gate as pg  # noqa: E402
from skillopt.experiments import common as C  # noqa: E402

N_ORDERS = 1000
N_BOOT = 10000


def paired_bootstrap_ci(cand, inc, ids, rng, level=0.90):
    d = np.array([cand[i] - inc[i] for i in ids], float)
    boots = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)


def main(pair_run=0, strict_run=1, suffix=""):
    R = C.RESULTS
    inc = {s: [json.load(open(os.path.join(R, "e0", f"{s}_run{r}.json"))) for r in (0, 1)] for s in ("pool", "test")}
    # third pool run of the initial skill: the E2 paired_bayes arm's fresh incumbent
    # (run_id incumbent-v0), extracted from runs/e2_paired_bayes/incumbent_runs/v000
    inc["pool"].append(json.load(open(os.path.join(R, "e0", "pool_run2_e2.json"))))
    scores = json.load(open(os.path.join(R, "e1_scores.json")))
    with open(os.path.join(R, "e1_candidates.jsonl")) as f:
        cands = [json.loads(line) for line in f if line.strip()]
    cands = [c for c in cands if c["cand_id"] in scores]
    pool_ids = list(inc["pool"][0].keys())
    test_ids = list(inc["test"][0].keys())
    n = len(pool_ids)
    inc1_pool_score = sum(inc["pool"][strict_run][i] for i in pool_ids) / n
    inc_pair_pool_score = sum(inc["pool"][pair_run][i] for i in pool_ids) / n
    rng = np.random.default_rng(0)
    orders = [list(rng.permutation(pool_ids)) for _ in range(N_ORDERS)]

    gates = [("strict", {}), ("signtest", {"alpha": 0.05}), ("signtest", {"alpha": 0.2}),
             ("paired_bayes", {"eta": 0.05}), ("paired_bayes", {"eta": 0.2})]
    per_cand = []
    brng = np.random.default_rng(1)
    for c in cands:
        s = scores[c["cand_id"]]
        cp, ct = s["pool"], s["test"]
        gain, lo, hi = paired_bootstrap_ci(ct, inc["test"][0], test_ids, brng)
        label = "good" if lo > 0 else ("harmful" if hi < 0 else "unclear")
        fix, brk, same = pg.paired_counts(cp, inc["pool"][pair_run], pool_ids)
        row = {"cand_id": c["cand_id"], "skill_len": c["skill_len"], "pool_acc": s["pool_acc"],
               "test_acc": s["test_acc"], "test_gain": gain, "ci_lo": lo, "ci_hi": hi, "label": label,
               "pool_fix": fix, "pool_brk": brk, "pool_same": same}
        for g, kw in gates:
            name = g + "".join(f"_{k}{v}" for k, v in kw.items())
            if g == "strict":
                acc = float(pg.strict_decision(s["pool_acc"], inc1_pool_score) == "accept")
                used = n
            elif g == "signtest":
                acc = float(pg.signtest_decision((fix, brk, same), kw["alpha"]) == "accept")
                used = n
            else:
                cfg = pg.PairedGateConfig(eta=kw["eta"])
                res = [pg.replay_sequential(cp, inc["pool"][pair_run], o, cfg) for o in orders]
                acc = float(np.mean([d == "accept" for d, _, _ in res]))
                used = float(np.mean([u for _, u, _ in res]))
            row[f"pacc_{name}"] = acc
            row[f"rollouts_{name}"] = used
        per_cand.append(row)

    with open(os.path.join(R, f"e1_replay_per_candidate{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cand[0].keys()))
        w.writeheader()
        w.writerows(per_cand)

    summary = []
    labels = ["good", "harmful", "unclear"]
    for g, kw in gates:
        name = g + "".join(f"_{k}{v}" for k, v in kw.items())
        row = {"gate": name}
        for lab in labels:
            xs = [r[f"pacc_{name}"] for r in per_cand if r["label"] == lab]
            row[f"n_{lab}"] = len(xs)
            row[f"p_accept_{lab}"] = float(np.mean(xs)) if xs else float("nan")
        pa = np.array([r[f"pacc_{name}"] for r in per_cand])
        gains = np.array([r["test_gain"] for r in per_cand])
        row["p_accept_all"] = float(pa.mean())
        row["mean_cand_rollouts_per_decision"] = float(np.mean([r[f"rollouts_{name}"] for r in per_cand]))
        # incumbent rollouts per decision: strict reuses its recorded score (0);
        # paired gates need a fresh incumbent run on the items they consume.
        row["mean_inc_rollouts_per_decision"] = 0.0 if g == "strict" else row["mean_cand_rollouts_per_decision"]
        row["expected_accepts"] = float(pa.sum())
        row["mean_test_gain_of_accepted"] = float((pa * gains).sum() / pa.sum()) if pa.sum() > 0 else float("nan")
        row["expected_total_test_gain"] = float((pa * gains).sum())
        summary.append(row)
    with open(os.path.join(R, f"e1_replay{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    meta = {"pair_run": pair_run, "strict_run": strict_run, "n_candidates": len(per_cand), "n_pool": n, "n_test": len(test_ids),
            "incumbent_pool_acc_run0": sum(inc["pool"][0].values()) / n,
            "incumbent_pool_acc_run1": sum(inc["pool"][1].values()) / n,
            "incumbent_pool_acc_run2_e2": sum(inc["pool"][2].values()) / n,
            "strict_recorded_score": inc1_pool_score,
            "paired_incumbent_pool_acc": inc_pair_pool_score,
            "incumbent_test_acc_run0": sum(inc["test"][0].values()) / len(test_ids),
            "labels": {lab: sum(r["label"] == lab for r in per_cand) for lab in labels},
            "n_orders": N_ORDERS, "n_boot": N_BOOT}
    C.write_json(os.path.join(R, f"e1_replay_meta{suffix}.json"), meta)
    if suffix:
        return
    # merge scores into the candidates file (the E1 deliverable)
    by = {r["cand_id"]: r for r in per_cand}
    with open(os.path.join(R, "e1_candidates_scored.jsonl"), "w") as f:
        for c in cands:
            f.write(json.dumps({**c, **{k: v for k, v in by[c["cand_id"]].items() if k != "cand_id"}},
                               ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=1))
    for r in summary:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()  # main analysis: paired gates vs incumbent run 0, strict vs recorded run 1
    # sensitivity (not used for any choice): swap the roles of the two incumbent runs
    main(pair_run=1, strict_run=0, suffix="_swapped_runs")
    # third incumbent run (E2 paired_bayes arm, 63.5%): strict and paired gates both use it
    main(pair_run=2, strict_run=2, suffix="_run3")
