"""The ported gate must make exactly the decisions of sim/sim.py (K = 1)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
import sim  # noqa: E402

from skillopt.evaluation import paired_gate as pg  # noqa: E402


def _sim_seqbayes(inc_out, cand_out, seed, eta):
    n = len(inc_out)
    rng = np.random.default_rng(seed)
    w = sim.World(rng, n, n_test=10)
    w.a_pool = np.where(inc_out == 1, 60.0, -60.0)  # deterministic incumbent
    eff_pool = np.where(cand_out == inc_out, 0.0, np.where(cand_out == 1, 120.0, -120.0))
    edit = (eff_pool, np.zeros(10))
    o = sim.Optimizer(w, "seqbayes", K=1, rng=rng, eta=eta)
    o.w.propose = lambda: edit
    o.step()
    cand_cost = o.cost // 2  # sim counts the (lazy) incumbent rollouts on the same items too
    return len(o.log) == 1, cand_cost


def _sim_order(n, seed):
    rng = np.random.default_rng(seed)
    sim.World(rng, n, n_test=10)  # consumes the same random numbers as in _sim_seqbayes
    return list(rng.permutation(n))


@pytest.mark.parametrize("trial", range(300))
def test_seqbayes_matches_sim(trial):
    r = np.random.default_rng(trial)
    n = int(r.choice([40, 120, 200]))
    p_inc = r.uniform(0.3, 0.8)
    inc = (r.random(n) < p_inc).astype(int)
    flip_fix, flip_brk = r.uniform(0, 0.25), r.uniform(0, 0.25)
    cand = inc.copy()
    u = r.random(n)
    cand[(inc == 0) & (u < flip_fix)] = 1
    cand[(inc == 1) & (u < flip_brk)] = 0
    eta = float(r.choice([0.05, 0.2]))
    seed = 1000 + trial
    acc_sim, cost_sim = _sim_seqbayes(inc, cand, seed, eta)
    order = _sim_order(n, seed)
    d, used, _ = pg.replay_sequential({i: int(cand[i]) for i in range(n)},
                                      {i: int(inc[i]) for i in range(n)}, order,
                                      pg.PairedGateConfig(eta=eta))
    assert (d == "accept") == acc_sim
    assert used == cost_sim


def test_sign_test_and_counts():
    assert pg.paired_counts({"a": 1, "b": 0, "c": 1}, {"a": 0, "b": 1, "c": 1}, ["a", "b", "c"]) == (1, 1, 1)
    assert pg.sign_test_p(0, 0) == 1.0
    assert pg.sign_test_p(8, 0) == pytest.approx(sim.sign_test_p(8, 0))
    assert pg.dirichlet_gain(5, 2, 30, 0.01, 0.005) == sim.dirichlet_gain(5, 2, 30, 0.01, 0.005)
    assert pg.strict_decision(0.5, 0.5) == "reject"
