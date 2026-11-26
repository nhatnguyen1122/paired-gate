"""E1 step 1: generate one-step candidate skills with SkillOpt's own stages.

For each of ``--n_batches`` train batches the initial skill is rolled out once
(run_id 0); then for each of ``--k`` optimizer samples (temperature 0.7,
distinct optimizer run_id => distinct cached samples) we run SkillOpt's
REFLECT -> AGGREGATE (merge) -> SELECT (rank, edit budget 4) -> UPDATE (apply
patch) pipeline, exactly as in step 1 of ``ReflACTTrainer.train`` with
patch mode and no meta skill / step buffer.
"""
import argparse
import functools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skillopt.engine.trainer import _normalise_patches  # noqa: E402
from skillopt.experiments import common as C  # noqa: E402
print = functools.partial(print, flush=True)  # noqa: A001
from skillopt.gradient.aggregate import merge_patches  # noqa: E402
from skillopt.model import llm_cache  # noqa: E402
from skillopt.optimizer.clip import rank_and_select  # noqa: E402
from skillopt.optimizer.skill import apply_patch_with_report  # noqa: E402
from skillopt.optimizer.update_modes import get_payload_items  # noqa: E402
from skillopt.utils import skill_hash  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n_batches", type=int, default=6)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--batch_size", type=int, default=20)
ap.add_argument("--edit_budget", type=int, default=4)
ap.add_argument("--max_usd", type=float, default=None)
ap.add_argument("--batches", default="", help="comma list of batch indices (default: all)")
args = ap.parse_args()
if args.max_usd is not None:
    os.environ["SKILLOPT_MAX_USD"] = str(args.max_usd)
os.environ["SKILLOPT_EXPERIMENT"] = "E1-generate"
C.setup_backends()
ad = C.make_adapter()
ad.analyst_workers = 4
ad.minibatch_size = 8
ad.edit_budget = args.edit_budget
skill0 = C.initial_skill()
train = C.load_split("train")
rng = random.Random(0)
order = list(range(len(train)))
rng.shuffle(order)
out_path = os.path.join(C.RESULTS, "e1_candidates.jsonl")
done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        done = {json.loads(line)["cand_id"] for line in f if line.strip()}
base = os.path.join(C.RUNS, "e1_gen")
sel_batches = [int(x) for x in args.batches.split(",")] if args.batches else range(args.n_batches)
for bi in sel_batches:
    items = [train[i] for i in order[bi * args.batch_size:(bi + 1) * args.batch_size]]
    bdir = os.path.join(base, f"batch{bi}")
    o, res, st = C.rollout(ad, items, skill0, os.path.join(bdir, "rollout"), run_id=0)
    train_acc = sum(o.values()) / len(o)
    for k in range(args.k):
        cid = f"b{bi}_s{k}"
        if cid in done:
            continue
        sdir = os.path.join(bdir, f"sample{k}")
        os.makedirs(sdir, exist_ok=True)
        with llm_cache.run_id_scope(f"e1-opt-{cid}"):
            raw = ad.reflect(res, skill0, sdir,
                             prediction_dir=os.path.join(bdir, "rollout", "predictions"),
                             patches_dir=os.path.join(sdir, "patches"),
                             random_seed=1000 * bi + k)
            fail, succ = _normalise_patches(raw, update_mode="patch")
            if not fail and not succ:
                print(f"[{cid}] no patches; skipping")
                continue
            merged = merge_patches(skill0, fail, succ, batch_size=8, verbose=False, workers=4,
                                   update_mode="patch", meta_skill_context="")
            ranked = rank_and_select(skill0, merged, max_edits=args.edit_budget, update_mode="patch",
                                     meta_skill_context="")
        cand, report = apply_patch_with_report(skill0, ranked)
        if skill_hash(cand) == skill_hash(skill0):
            print(f"[{cid}] patch left skill unchanged; skipping")
            continue
        rec = {"cand_id": cid, "batch": bi, "sample": k, "train_acc_initial": train_acc,
               "skill": cand, "skill_hash": skill_hash(cand), "skill_len": len(cand),
               "patch": ranked, "edits": get_payload_items(ranked, "patch"),
               "n_failure_patches": len(fail), "n_success_patches": len(succ),
               "apply_report": report}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{cid}] skill_len={len(cand)} edits={len(rec['edits'])} usd_total={llm_cache.total_usd():.3f}",
              flush=True)
