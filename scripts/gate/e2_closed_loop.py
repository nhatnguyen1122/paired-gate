"""E2: closed-loop SkillOpt with gate_type=strict vs paired_bayes, then test eval.

Each arm: ``scripts/train.py --config configs/searchqa/e2_<gate>.yaml`` (1 epoch,
same seed, cap of 3000 target rollouts), then initial and final skills on the
300 test items with run_id 0 (the initial skill's run 0 is the E0 run).
"""
import argparse
import csv
import fcntl
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from skillopt.experiments import common as C  # noqa: E402
from skillopt.model import llm_cache  # noqa: E402


def ledger_usage(tag):
    out = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    p = os.path.join(llm_cache.llm_dir(), "ledger.jsonl")
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r["exp"] == tag:
                out["calls"] += 1
                out["prompt_tokens"] += r["prompt_tokens"]
                out["completion_tokens"] += r["completion_tokens"]
                out["usd"] += r["usd"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="strict,paired_bayes")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--test_n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = []
    for g in args.gates.split(","):
        sfx = "" if args.seed == 42 else f"_s{args.seed}"  # seed 42 keeps its original names
        out_root = os.path.join(ROOT, "runs", f"e2_{g}{sfx}")
        tag = f"E2-{g}" + ("" if args.seed == 42 else f"-s{args.seed}")
        if not args.skip_train:
            env = dict(os.environ, SKILLOPT_EXPERIMENT=tag)
            with open(os.path.join(ROOT, "logs", f"e2_{g}{sfx}.log"), "a") as log:
                subprocess.run([sys.executable, "-u", os.path.join(ROOT, "scripts", "train.py"),
                                "--config", os.path.join(ROOT, "configs", "searchqa", f"e2_{g}.yaml"),
                                "--out_root", out_root, "--cfg-options", f"train.seed={args.seed}"], cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=False)
        hist = json.load(open(os.path.join(out_root, "history.json")))
        skills = sorted(glob.glob(os.path.join(out_root, "skills", "skill_v*.md")))
        final_skill = open(skills[-1]).read() if skills else C.initial_skill()
        os.environ["SKILLOPT_EXPERIMENT"] = f"E2-test-{g}{sfx}"
        C.setup_backends()
        ad = C.make_adapter()
        test = C.test_items(args.test_n)
        init_o, _, _ = C.rollout(ad, test, C.initial_skill(), os.path.join(C.RUNS, "e0", "test", "run0"), run_id=0)
        fin_o, _, _ = C.rollout(ad, test, final_skill, os.path.join(C.RUNS, f"e2_test_{g}{sfx}"), run_id=0)
        ids = list(init_o)
        init_acc = sum(init_o.values()) / len(ids)
        fin_acc = sum(fin_o[i] for i in ids) / len(ids)
        last = hist[-1] if hist else {}
        counter = last.get("target_rollouts_cum", {})
        n_acc = sum(1 for h in hist if "accept" in h.get("action", ""))
        sel_hard_recorded = last.get("current_score")
        usage = ledger_usage(tag)
        row = {"gate": g, "seed": args.seed, "steps": len(hist), "accepted_edits": n_acc,
               "target_rollouts": counter.get("total"), "rollouts_train": counter.get("train"),
               "rollouts_sel_candidate": counter.get("selection_candidate"),
               "rollouts_sel_incumbent": counter.get("selection_incumbent"),
               "rollouts_baseline": counter.get("baseline"),
               "initial_test_acc": init_acc, "final_test_acc": fin_acc, "test_gain": fin_acc - init_acc,
               "final_skill_len": len(final_skill),
               "recorded_selection_score": sel_hard_recorded,
               "selection_minus_test": (sel_hard_recorded - fin_acc) if g == "strict" and sel_hard_recorded is not None else None,
               "api_calls": usage["calls"], "prompt_tokens": usage["prompt_tokens"],
               "completion_tokens": usage["completion_tokens"], "usd": usage["usd"]}
        rows.append(row)
        print(row, flush=True)
    path = os.path.join(C.RESULTS, "e2_runs.csv")
    done = {(r["gate"], str(r["seed"])) for r in rows}
    with open(path + ".lock", "w") as lk:  # several seeds/gates may finish at once
        fcntl.flock(lk, fcntl.LOCK_EX)
        old = []
        if os.path.exists(path):
            with open(path) as f:
                old = [r for r in csv.DictReader(f) if (r["gate"], str(r["seed"])) not in done]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in old + rows:
                w.writerow(r)


if __name__ == "__main__":
    main()
