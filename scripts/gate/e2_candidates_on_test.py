"""Diagnostic: test accuracy (run_id 0) of every candidate an E2 arm evaluated."""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skillopt.experiments import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gate", default="paired_bayes")
args = ap.parse_args()
os.environ["SKILLOPT_EXPERIMENT"] = f"E2-cands-test-{args.gate}"
C.setup_backends()
ad = C.make_adapter()
test = C.test_items(300)
inc = json.load(open(os.path.join(C.RESULTS, "e0", "test_run0.json")))
ids = list(inc)
hist = {h["step"]: h for h in json.load(open(os.path.join(C.ROOT, "runs", f"e2_{args.gate}", "history.json")))}
rows = []
rng = np.random.default_rng(1)
for p in sorted(glob.glob(os.path.join(C.ROOT, "runs", f"e2_{args.gate}", "steps", "step_*", "candidate_skill.md"))):
    step = int(p.split("step_")[1][:4])
    skill = open(p).read()
    o, _, _ = C.rollout(ad, test, skill, os.path.join(C.RUNS, f"e2_cands_test_{args.gate}", f"step{step:04d}"), run_id=0)
    d = np.array([o[i] - inc[i] for i in ids])
    bs = rng.choice(d, (10000, len(d))).mean(1)
    lo, hi = np.quantile(bs, [0.05, 0.95])
    h = hist.get(step, {})
    g = h.get("gate", {})
    rows.append({"gate": args.gate, "step": step, "action": h.get("action"), "items_used": g.get("items_used"),
                 "fix": g.get("fix"), "brk": g.get("brk"), "p_pos": g.get("p_pos"), "p_harm": g.get("p_harm"),
                 "selection_hard": h.get("selection_hard"),
                 "test_acc": float(np.mean([o[i] for i in ids])), "test_gain_vs_initial": float(d.mean()),
                 "ci_lo": float(lo), "ci_hi": float(hi)})
    print(rows[-1], flush=True)
with open(os.path.join(C.RESULTS, f"e2_candidates_test_{args.gate}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
