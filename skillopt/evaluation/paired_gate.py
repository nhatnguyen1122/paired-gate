"""Paired acceptance gates (sign test and sequential Bayesian racing).

Pure functions ported verbatim from the reference simulation ``sim/sim.py``
(``sign_test_p``, ``dirichlet_gain`` and the ``seqbayes`` / ``signtest`` /
``strict`` branches of ``Optimizer.step``). Do not change the rules here without
changing ``sim/sim.py`` too.

Terminology: on the same selection items, a *fix* is an item only the candidate
solves, a *break* an item only the incumbent solves, *same* everything else.

``paired_bayes`` = the simulation's ``seqbayes`` with K = 1 candidate per step:
look every ``b`` items; reject when ``p_useful < q_fut`` after at least ``2b``
items (futility); accept only after ``acc_min_frac`` of the pool when
``p_pos >= q_acc`` and ``p_harm <= eta``; at the end of the pool accept iff the
acceptance rule holds ("reject early, accept late").
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import binom, norm

Decision = Literal["accept", "reject", "continue"]


def outcomes_by_id(results: list[dict]) -> dict[str, int]:
    """Turn ``adapter.rollout`` rows into ``{id: hard}`` (0/1)."""
    return {str(r["id"]): int(round(float(r.get("hard", 0) or 0))) for r in results}


def paired_counts(cand: dict, inc: dict, ids: Iterable) -> tuple[int, int, int]:
    """(fix, brk, same) of candidate vs incumbent outcomes on ``ids``."""
    fix = brk = same = 0
    for i in ids:
        c, o = int(cand[i]), int(inc[i])
        if c == 1 and o == 0:
            fix += 1
        elif c == 0 and o == 1:
            brk += 1
        else:
            same += 1
    return fix, brk, same


# ---------------------------------------------------------------- verbatim from sim/sim.py
def sign_test_p(fix, brk):
    """One-sided exact sign test: P(X >= fix), X ~ Bin(fix + brk, 1/2)."""
    m = fix + brk
    if m == 0:
        return 1.0
    return float(binom.sf(fix - 1, m, 0.5))


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


# ---------------------------------------------------------------- decision rules
@dataclass
class PairedGateConfig:
    b: int = 10
    q_acc: float = 0.8
    eta: float = 0.05
    eps_harm: float = 0.01
    d_useful: float = 0.005
    q_fut: float = 0.2
    acc_min_frac: float = 0.5
    alpha: float = 0.05  # signtest only

    def to_dict(self) -> dict:
        return asdict(self)


def _accept_rule(p_pos: float, p_harm: float, cfg: PairedGateConfig) -> bool:
    return p_pos >= cfg.q_acc and p_harm <= cfg.eta


def sequential_decision(stats: tuple[int, int, int], n_seen: int, n_pool: int,
                        cfg: PairedGateConfig) -> Decision:
    """seqbayes rule after a look. ``stats`` = cumulative (fix, brk, same).

    Mirrors the K = 1 case of ``Optimizer.step`` in sim.py: the acceptance test
    is checked first, then futility; at pool exhaustion the acceptance rule alone
    decides.
    """
    fix, brk, same = stats
    p_pos, p_harm, p_useful, _ = dirichlet_gain(fix, brk, same, cfg.eps_harm, cfg.d_useful)
    m = fix + brk + same
    if m >= cfg.acc_min_frac * n_pool and _accept_rule(p_pos, p_harm, cfg):
        return "accept"
    if m >= 2 * cfg.b and p_useful < cfg.q_fut:
        return "reject"
    if n_seen >= n_pool:
        return "accept" if _accept_rule(p_pos, p_harm, cfg) else "reject"
    return "continue"


def posterior_summary(stats: tuple[int, int, int], cfg: PairedGateConfig) -> dict:
    p_pos, p_harm, p_useful, mean = dirichlet_gain(*stats, cfg.eps_harm, cfg.d_useful)
    return {"fix": stats[0], "brk": stats[1], "same": stats[2], "p_pos": p_pos,
            "p_harm": p_harm, "p_useful": p_useful, "mean_gain": float(mean),
            "sign_p": sign_test_p(stats[0], stats[1])}


def signtest_decision(stats: tuple[int, int, int], alpha: float) -> Decision:
    """Full-pool paired one-sided exact sign test."""
    return "accept" if sign_test_p(stats[0], stats[1]) <= alpha else "reject"


def strict_decision(cand_score: float, inc_score: float) -> Decision:
    """SkillOpt's rule: strictly greater than the recorded score."""
    return "accept" if cand_score > inc_score else "reject"


def replay_sequential(cand: dict, inc: dict, order: list, cfg: PairedGateConfig) -> tuple[Decision, int, dict]:
    """Run the sequential gate over ``order`` offline (no API). Returns
    (decision, candidate rollouts used, posterior summary at the stop)."""
    n = len(order)
    fix = brk = same = 0
    looks = math.ceil(n / cfg.b)
    for look in range(looks):
        idx = order[look * cfg.b:(look + 1) * cfg.b]
        f, br, s = paired_counts(cand, inc, idx)
        fix += f
        brk += br
        same += s
        seen = min((look + 1) * cfg.b, n)
        d = sequential_decision((fix, brk, same), seen, n, cfg)
        if d != "continue":
            return d, seen, posterior_summary((fix, brk, same), cfg)
    return "reject", n, posterior_summary((fix, brk, same), cfg)
