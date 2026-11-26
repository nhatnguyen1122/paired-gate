"""Pick the target model: SearchQA accuracy of the initial skill on 40 val items."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skillopt.experiments import common as C  # noqa: E402
from skillopt.model import llm_cache, set_target_deployment  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="ministral-8b-latest,mistral-small-latest")
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()
os.environ["SKILLOPT_EXPERIMENT"] = "probe"
C.setup_backends()
ad = C.make_adapter()
items = C.load_split("val")[-args.n:]   # last 40 of val
rows = []
for m in args.models.split(","):
    os.environ["TARGET_OPENAI_COMPATIBLE_MODEL"] = m
    set_target_deployment(m)
    out, res, st = C.rollout(ad, items, C.initial_skill(), os.path.join(C.RUNS, "probe", m), run_id="probe")
    acc = sum(out.values()) / len(out)
    rows.append({"model": m, "acc": acc, **st, "tokens_per_rollout": (st["prompt_tokens"] + st["completion_tokens"]) / max(1, st["n"])})
    print(rows[-1], "spent_total_usd=%.4f" % llm_cache.total_usd(), flush=True)
C.write_json(os.path.join(C.RESULTS, "probe_models.json"), rows)
