"""E0: base accuracy and run-to-run flip rate of the initial skill (run_id 0 and 1)."""
import argparse
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
os.environ["SKILLOPT_EXPERIMENT"] = "E0"
C.setup_backends()
ad = C.make_adapter()
skill = C.initial_skill()
usd0 = llm_cache.total_usd()
out = {"target_model": os.environ.get("TARGET_OPENAI_COMPATIBLE_MODEL"),
       "temperature": float(os.environ.get("TARGET_OPENAI_COMPATIBLE_TEMPERATURE", "nan"))}
tok = {"n": 0, "prompt": 0, "completion": 0}
for split, items in [("pool", C.pool_items(args.pool_n)), ("test", C.test_items(args.test_n))]:
    runs = []
    for r in (0, 1):
        o, res, st = C.rollout(ad, items, skill, os.path.join(C.RUNS, "e0", split, f"run{r}"), run_id=r)
        C.write_json(os.path.join(C.RESULTS, "e0", f"{split}_run{r}.json"), o)
        runs.append(o)
        tok["n"] += st["n"]; tok["prompt"] += st["prompt_tokens"]; tok["completion"] += st["completion_tokens"]
        out[f"{split}_acc_run{r}"] = sum(o.values()) / len(o)
        out[f"{split}_agent_errors_run{r}"] = st["n_agent_error"]
    ids = list(runs[0])
    out[f"{split}_flip_rate"] = sum(runs[0][i] != runs[1][i] for i in ids) / len(ids)
    out[f"{split}_n"] = len(ids)
out["mean_prompt_tokens_per_rollout"] = tok["prompt"] / tok["n"]
out["mean_completion_tokens_per_rollout"] = tok["completion"] / tok["n"]
out["mean_tokens_per_rollout"] = (tok["prompt"] + tok["completion"]) / tok["n"]
out["usd_this_run"] = llm_cache.total_usd() - usd0
C.write_json(os.path.join(C.RESULTS, "e0_summary.json"), out)
print(out)
