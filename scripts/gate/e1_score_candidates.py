"""E1 step 2: roll every candidate out once on the pool and once on test (run_id 0)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skillopt.experiments import common as C  # noqa: E402
from skillopt.model import llm_cache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool_n", type=int, default=200)
ap.add_argument("--test_n", type=int, default=300)
ap.add_argument("--max_usd", type=float, default=None)
args = ap.parse_args()
if args.max_usd is not None:
    os.environ["SKILLOPT_MAX_USD"] = str(args.max_usd)
os.environ["SKILLOPT_EXPERIMENT"] = "E1-score"
C.setup_backends()
ad = C.make_adapter()
pool, test = C.pool_items(args.pool_n), C.test_items(args.test_n)
cpath = os.path.join(C.RESULTS, "e1_candidates.jsonl")
with open(cpath) as f:
    cands = [json.loads(line) for line in f if line.strip()]
spath = os.path.join(C.RESULTS, "e1_scores.json")
scores = json.load(open(spath)) if os.path.exists(spath) else {}
for c in cands:
    cid = c["cand_id"]
    if cid in scores:
        continue
    rec = {}
    for split, items in [("pool", pool), ("test", test)]:
        o, res, st = C.rollout(ad, items, c["skill"], os.path.join(C.RUNS, "e1_score", cid, split), run_id=0)
        rec[split] = o
        rec[f"{split}_acc"] = sum(o.values()) / len(o)
        rec[f"{split}_tokens"] = st
    scores[cid] = rec
    C.write_json(spath, scores)
    print(f"[{cid}] pool={rec['pool_acc']:.3f} test={rec['test_acc']:.3f} usd_total={llm_cache.total_usd():.3f}",
          flush=True)
