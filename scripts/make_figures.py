"""Figures for the paired-gate study (E1 replay, E2 closed loop)."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
F = os.path.join(R, "figures")
os.makedirs(F, exist_ok=True)


def read(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def fig_e1_bars(rows, amort=None):
    labels = ["good", "harmful", "unclear"]
    colors = {"good": "#2a9d8f", "harmful": "#e76f51", "unclear": "#8d99ae"}
    gates = [r["gate"] for r in rows]
    x = np.arange(len(gates))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for j, lab in enumerate(labels):
        vals = [float(r[f"p_accept_{lab}"]) if r[f"p_accept_{lab}"] != "nan" else 0 for r in rows]
        n = rows[0][f"n_{lab}"]
        ax.bar(x + (j - 1) * w, vals, w, label=f"{lab} (n={n})", color=colors[lab])
    am = {a["gate"]: a for a in (amort or [])}
    for i, r in enumerate(rows):
        txt = f"cand {float(r['mean_cand_rollouts_per_decision']):.0f}"
        if r["gate"] in am:
            txt += f"\namortized {float(am[r['gate']]['amortized_per_decision']):.0f}"
        ax.text(i, 1.03, txt, ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(gates, rotation=15, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("P(accept)")
    ax.set_title("E1: acceptance by true test effect\n(annotated: candidate rollouts / decision; amortized total target rollouts / decision, 24 candidates vs one incumbent)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(F, "e1_accept_by_label.png"), dpi=150)
    plt.close(fig)


def fig_e1_scatter(rows):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    gains = np.array([float(r["test_gain"]) for r in rows]) * 100
    for key, lab, mk in [("pacc_strict", "strict", "o"), ("pacc_paired_bayes_eta0.05", "paired_bayes η=0.05", "s")]:
        ax.scatter(gains, [float(r[key]) for r in rows], label=lab, marker=mk, alpha=0.75)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Δ test accuracy vs incumbent (pp)")
    ax.set_ylabel("P(accept)")
    ax.set_title("E1: per-candidate acceptance vs true test gain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "e1_scatter_gain_vs_accept.png"), dpi=150)
    plt.close(fig)


def fig_e2(rows):
    colors = {"strict": "#e76f51", "paired_bayes": "#2a9d8f"}
    off = {"strict": -0.12, "paired_bayes": 0.12}
    seeds = sorted({int(r["seed"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    seen = set()
    for r in rows:
        g = r["gate"]
        x = seeds.index(int(r["seed"])) + off.get(g, 0)
        y = 100 * float(r["test_gain"])
        lo, hi = 100 * float(r.get("ci90_lo", r["test_gain"])), 100 * float(r.get("ci90_hi", r["test_gain"]))
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="o", color=colors.get(g, "k"), capsize=3,
                    label=None if g in seen else g)
        seen.add(g)
        ax.annotate(f"{r['accepted_edits']} acc.\n{int(float(r['target_rollouts']))} roll.", (x, y),
                    textcoords="offset points", xytext=(6, -4), fontsize=7)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_xlim(-0.5, len(seeds) - 0.3)
    ax.set_ylabel("test gain vs initial skill (pp, 90% CI)")
    ax.set_title("E2: closed loop (annotated: accepted edits, target rollouts)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "e2_gain_vs_rollouts.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    s = read("e1_replay.csv")
    if s:
        fig_e1_bars(s, read("e1_amortized_cost.csv"))
    pc = read("e1_replay_per_candidate.csv")
    if pc:
        fig_e1_scatter(pc)
    e2 = read("e2_runs_ci.csv") or read("e2_runs.csv")
    if e2:
        fig_e2(e2)
    print("figures in", F)
