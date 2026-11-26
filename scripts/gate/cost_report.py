"""Tokens and USD per experiment tag, from the LLM ledger (real API calls only)."""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skillopt.experiments import common as C  # noqa: E402
from skillopt.model import llm_cache  # noqa: E402

agg = collections.defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0})
with open(os.path.join(llm_cache.llm_dir(), "ledger.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        for key in ((r["exp"], r["model"]), (r["exp"], "ALL")):
            a = agg[key]
            a["calls"] += 1
            a["prompt_tokens"] += r["prompt_tokens"]
            a["completion_tokens"] += r["completion_tokens"]
            a["usd"] += r["usd"]
rows = [{"experiment": e, "model": m, **v} for (e, m), v in sorted(agg.items())]
tot = {"experiment": "TOTAL", "model": "ALL",
       **{k: sum(r[k] for r in rows if r["model"] == "ALL") for k in ("calls", "prompt_tokens", "completion_tokens", "usd")}}
rows.append(tot)
with open(os.path.join(C.RESULTS, "cost_by_experiment.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
