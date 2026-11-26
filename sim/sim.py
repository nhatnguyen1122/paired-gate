"""Simulation of acceptance gates for feedback-based skill optimization.

A skill is a set of accepted edits. Each task i has a base logit a_i; an edit e changes the
logit of the tasks it touches by an amount drawn around its mean effect mu_e. One agent run on
task i with skill s succeeds with probability sigmoid(a_i + sum_e effect_e,i), independently
across runs, so tasks near p = 0.5 flip between runs (run-to-run noise).

Gates compared (all see only sampled 0/1 outcomes on a finite selection pool):
  strict   : SkillOpt-style. Each candidate scored once on the full pool; accepted if its score
             is strictly greater than the incumbent's recorded score (the score it had when it
             was accepted).
  signtest : full pool, paired against a fresh incumbent run, exact one-sided sign test at alpha.
  seqfreq  : paired group-sequential test with 4 looks (quarters of the pool); accept when a
             Bonferroni-over-looks sign test rejects H0 (no gain); stop early when an upper
             confidence bound on the gain falls below a minimum useful effect.
  seqbayes : paired sequential racing with a Dirichlet(1,1,1) posterior over (fix, break, same);
             looks every b tasks. Reject early (futility) when P(gain > d_useful) < q_fut; accept only
             after at least acc_min_frac of the pool, when P(gain > 0) >= q_acc and
             P(gain < -eps_harm) <= eta ("reject early, accept late").
  nogate   : accept a random candidate every step (no evaluation).
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.stats import beta as beta_dist, binom, norm


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class World:
    """Tasks and the distribution of edits proposed by the optimizer."""

    def __init__(self, rng, n_pool, n_test=3000, a_mean=0.6, a_sd=2.5,
                 touch=0.3, het=0.3, mix=(0.5, 0.35, 0.15), mu_bad=-0.45, mu_good=0.45):
        self.rng = rng
        self.n_pool, self.n_test = n_pool, n_test
        self.a_pool = rng.normal(a_mean, a_sd, n_pool)
        self.a_test = rng.normal(a_mean, a_sd, n_test)
        self.touch, self.het = touch, het
        self.mix, self.mu_bad, self.mu_good = mix, mu_bad, mu_good

    def edit_effects(self, mu):
        """Per-task logit effects of one edit on pool and test tasks."""
        r = self.rng
        def eff(n):
            m = r.random(n) < self.touch
            return m * (mu + self.het * r.normal(size=n))
        return eff(self.n_pool), eff(self.n_test)

    def propose(self):
        """Draw an edit from the proposal distribution: mostly harmful or neutral."""
        u = self.rng.random()
        if u < self.mix[0]:
            mu = self.rng.normal(self.mu_bad, 0.15)
        elif u < self.mix[0] + self.mix[1]:
            mu = 0.0
        else:
            mu = self.rng.normal(self.mu_good, 0.15)
        return self.edit_effects(mu)


class Skill:
    def __init__(self, world, u_pool=None, u_test=None):
        self.w = world
        self.u_pool = np.zeros(world.n_pool) if u_pool is None else u_pool
        self.u_test = np.zeros(world.n_test) if u_test is None else u_test

    def add(self, edit):
        return Skill(self.w, self.u_pool + edit[0], self.u_test + edit[1])

    def p_pool(self):
        return sigmoid(self.w.a_pool + self.u_pool)

    def true_acc(self):
        return float(sigmoid(self.w.a_test + self.u_test).mean())

    def run(self, idx):
        """One stochastic agent run on pool tasks idx."""
        return (self.w.rng.random(len(idx)) < self.p_pool()[idx]).astype(np.int8)


# ---------------------------------------------------------------- gate statistics
def sign_test_p(fix, brk):
    """One-sided exact sign test: P(X >= fix), X ~ Bin(fix + brk, 1/2)."""
    m = fix + brk
    if m == 0:
        return 1.0
    return float(binom.sf(fix - 1, m, 0.5))


def cp_upper_gain(fix, brk, same, level):
    """Conservative upper bound on gain = p_fix - p_break (normal approximation on the paired mean)."""
    n = fix + brk + same
    if n == 0:
        return 1.0
    d = (fix - brk) / n
    var = ((fix + brk) / n - d * d) / n
    return d + norm.ppf(1 - level) * math.sqrt(max(var, 1e-9)) + 1.0 / n


def dirichlet_gain(fix, brk, same, eps_harm, d_useful):
    """Posterior summaries of gain = p_fix - p_break under Dirichlet(1,1,1) prior."""
    p_pos = float(beta_dist.sf(0.5, 1 + fix, 1 + brk))  # P(p_fix > p_break)
    a = np.array([1 + fix, 1 + brk, 1 + same], float)
    a0 = a.sum()
    mean = (a[0] - a[1]) / a0
    var = (a[0] * (a0 - a[0]) + a[1] * (a0 - a[1]) + 2 * a[0] * a[1]) / (a0 * a0 * (a0 + 1))
    sd = math.sqrt(var)
    p_harm = float(norm.cdf((-eps_harm - mean) / sd))
    p_useful = float(norm.sf((d_useful - mean) / sd))
    return p_pos, p_harm, p_useful, mean


# ---------------------------------------------------------------- one optimization step per gate
class Optimizer:
    def __init__(self, world, gate, K=4, b=10, alpha=0.05, q_acc=0.8, eta=0.05, eps_harm=0.01,
                 d_useful=0.005, q_fut=0.2, alpha_fut=0.05, acc_min_frac=0.5, freq_looks=4, rng=None):
        self.w, self.gate, self.K, self.b = world, gate, K, b
        self.alpha, self.q_acc, self.eta, self.eps_harm = alpha, q_acc, eta, eps_harm
        self.d_useful, self.q_fut, self.alpha_fut = d_useful, q_fut, alpha_fut
        self.acc_min_frac, self.freq_looks = acc_min_frac, freq_looks
        self.rng = rng
        self.inc = Skill(world)
        self.cost = 0
        self.inc_cache = {}  # lazy per-task outcomes of a fresh incumbent run
        self.inc_score = None  # strict gate: recorded selection score
        self.log = []

    # incumbent outcomes on demand (fresh run, never reused from a race)
    def inc_outcomes(self, idx):
        need = [i for i in idx if i not in self.inc_cache]
        if need:
            out = self.inc.run(np.array(need))
            self.cost += len(need)
            for i, o in zip(need, out):
                self.inc_cache[i] = int(o)
        return np.array([self.inc_cache[i] for i in idx], dtype=np.int8)

    def accept(self, edit, true_gain):
        self.inc = self.inc.add(edit)
        self.inc_cache = {}
        self.log.append(true_gain)

    def step(self):
        n = self.w.n_pool
        cands = [self.w.propose() for _ in range(self.K)]
        skills = [self.inc.add(e) for e in cands]
        base_true = self.inc.true_acc()
        gains = [s.true_acc() - base_true for s in skills]
        allidx = np.arange(n)

        if self.gate == "nogate":
            k = int(self.rng.integers(self.K))
            self.accept(cands[k], gains[k])
            return

        if self.gate == "strict":
            if self.inc_score is None:
                self.inc_score = self.inc.run(allidx).mean()
                self.cost += n
            scores = []
            for s in skills:
                scores.append(s.run(allidx).mean())
                self.cost += n
            k = int(np.argmax(scores))
            if scores[k] > self.inc_score:
                self.inc_score = scores[k]
                self.accept(cands[k], gains[k])
            return

        if self.gate == "signtest":
            inc = self.inc_outcomes(allidx)
            best, best_p = None, 1.0
            for k, s in enumerate(skills):
                out = s.run(allidx)
                self.cost += n
                fix = int(((out == 1) & (inc == 0)).sum())
                brk = int(((out == 0) & (inc == 1)).sum())
                p = sign_test_p(fix, brk)
                if p < best_p:
                    best, best_p = k, p
            if best is not None and best_p <= self.alpha:
                self.accept(cands[best], gains[best])
            return

        # sequential racing gates
        order = self.rng.permutation(n)
        b = self.b if self.gate == "seqbayes" else math.ceil(n / self.freq_looks)
        looks = math.ceil(n / b)
        acc_min = self.acc_min_frac * n
        alive = list(range(self.K))
        stats = {k: [0, 0, 0] for k in alive}  # fix, brk, same
        for l in range(looks):
            idx = order[l * b:(l + 1) * b]
            inc = self.inc_outcomes(idx)
            winners = []
            for k in list(alive):
                out = skills[k].run(idx)
                self.cost += len(idx)
                f = int(((out == 1) & (inc == 0)).sum()); br = int(((out == 0) & (inc == 1)).sum())
                st = stats[k]; st[0] += f; st[1] += br; st[2] += len(idx) - f - br
                fix, brk, same = st
                if self.gate == "seqfreq":
                    if sign_test_p(fix, brk) <= self.alpha / looks:
                        winners.append((fix - brk, k))
                    elif cp_upper_gain(fix, brk, same, self.alpha_fut / looks) < self.d_useful:
                        alive.remove(k)
                else:  # seqbayes
                    p_pos, p_harm, p_useful, mean = dirichlet_gain(fix, brk, same, self.eps_harm, self.d_useful)
                    m = fix + brk + same
                    if m >= acc_min and p_pos >= self.q_acc and p_harm <= self.eta:
                        winners.append((mean, k))
                    elif m >= 2 * self.b and p_useful < self.q_fut:
                        alive.remove(k)
            if winners:
                k = max(winners)[1]
                self.accept(cands[k], gains[k])
                return
            if not alive:
                return
        # pool exhausted: Bayesian gate decides on the final posterior, frequentist keeps incumbent
        if self.gate == "seqbayes":
            best = None
            for k in alive:
                p_pos, p_harm, _, mean = dirichlet_gain(*stats[k], self.eps_harm, self.d_useful)
                if p_pos >= self.q_acc and p_harm <= self.eta and (best is None or mean > best[0]):
                    best = (mean, k)
            if best:
                self.accept(cands[best[1]], gains[best[1]])


# ---------------------------------------------------------------- experiments
def exp_single(args):
    """Acceptance probability and tasks used for one candidate of known true gain."""
    rows = []
    mus = [-0.6, -0.3, 0.0, 0.3, 0.6, 1.2]
    for n in [40, 140, 400]:
        for mu in mus:
            acc = {g: 0 for g in GATES_EVAL}; cost = {g: 0 for g in GATES_EVAL}; gsum = 0.0
            for r in range(args.reps):
                rng = np.random.default_rng(1000 * n + r)
                w = World(rng, n, n_test=2000)
                e = w.edit_effects(mu)
                for g in GATES_EVAL:
                    o = Optimizer(w, g, K=1, rng=rng)
                    o.w.propose = lambda e=e: e
                    o.step()
                    acc[g] += len(o.log); cost[g] += o.cost
                base = Skill(w); gsum += base.add(e).true_acc() - base.true_acc()
            rows.append(dict(n=n, mu=mu, true_gain=100 * gsum / args.reps,
                             **{f"acc_{g}": acc[g] / args.reps for g in GATES_EVAL},
                             **{f"cost_{g}": cost[g] / args.reps for g in GATES_EVAL}))
            print(rows[-1], flush=True)
    return rows


def run_loop(n, gate, budget, rep, **kw):
    rng = np.random.default_rng(77 + rep)
    w = World(rng, n)
    o = Optimizer(w, gate, rng=rng, **kw)
    a0 = o.inc.true_acc()
    steps = 0
    while o.cost < budget and steps < 400:
        o.step(); steps += 1
    gains = np.array(o.log) if o.log else np.zeros(0)
    return dict(final_gain=100 * (o.inc.true_acc() - a0), cost=o.cost, steps=steps,
                n_acc=len(gains), n_harm=int((gains < -0.002).sum()),
                n_null=int((np.abs(gains) <= 0.002).sum()), n_good=int((gains > 0.002).sum()))


def exp_budget(args):
    """Budget-matched optimization runs: final true gain vs evaluation rollouts."""
    rows = []
    configs = [("nogate", {}), ("strict", {}), ("signtest", dict(alpha=0.05)), ("signtest", dict(alpha=0.2)),
               ("seqfreq", dict(alpha=0.05)), ("seqfreq", dict(alpha=0.2)),
               ("seqbayes", dict(q_acc=0.7)), ("seqbayes", dict(q_acc=0.8)), ("seqbayes", dict(q_acc=0.9))]
    for n in [40, 140]:
        for mult in [1, 2, 4, 8]:
            budget = mult * 4 * n * 5  # multiples of 5 strict steps with K = 4
            for g, kw in configs:
                res = [run_loop(n, g, budget, r, **kw) for r in range(args.reps_loop)]
                agg = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
                agg["final_gain_se"] = float(np.std([x["final_gain"] for x in res]) / math.sqrt(len(res)))
                rows.append(dict(n=n, budget=budget, gate=g, cfg=json.dumps(kw), **agg))
                print(rows[-1], flush=True)
    return rows



def exp_paired(reps=200, n=140, budgets=(5600, 11200)):
    """Paired (same world seed) comparison of strict vs seqbayes variants, per-rep results kept."""
    configs = [("strict", {}), ("signtest", dict(alpha=0.2)), ("seqbayes", dict(eta=0.05)), ("seqbayes", dict(eta=0.2))]
    out = []
    for budget in budgets:
        for g, kw in configs:
            res = [run_loop(n, g, budget, r, **kw) for r in range(reps)]
            out.append(dict(n=n, budget=budget, gate=g, cfg=json.dumps(kw), runs=res))
            print(budget, g, kw, np.mean([x["final_gain"] for x in res]), flush=True)
    return out


GATES_EVAL = ["strict", "signtest", "seqfreq", "seqbayes"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--reps_loop", type=int, default=100)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--which", default="single,budget")
    args = ap.parse_args()
    if "single" in args.which:
        json.dump(exp_single(args), open(os.path.join(args.out, "single.json"), "w"), indent=1)
    if "paired" in args.which:
        json.dump(exp_paired(), open(os.path.join(args.out, "paired.json"), "w"))
    if "budget" in args.which:
        json.dump(exp_budget(args), open(os.path.join(args.out, "budget.json"), "w"), indent=1)

