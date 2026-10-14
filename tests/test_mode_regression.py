"""Regression tests for MODE (native modecpp and pure-Python mode).

Covers: SBX must be able to expand beyond the parents, Pareto levels must rank
successive fronts, crowding must use all objectives, NSGA-update MODE must keep
the spread on ZDT2 instead of collapsing to one extreme point, and DE-update
MODE must converge on ZDT4.
"""
import numpy as np
from scipy.optimize import Bounds

from fcmaes import mode
from fcmaes.modecpp import MODE_C


def zdt2(X):
    X = np.atleast_2d(X)
    g = 1 + 9 * X[:, 1:].sum(1) / (X.shape[1] - 1)
    f1 = X[:, 0]
    return np.stack([f1, g * (1 - (f1 / g) ** 2)], 1)


def zdt4_g(X):
    xi = np.atleast_2d(X)[:, 1:]
    return 1 + 10 * xi.shape[1] + (xi * xi - 10 * np.cos(4 * np.pi * xi)).sum(1)


def zdt4(X):
    X = np.atleast_2d(X)
    g = zdt4_g(X)
    f1 = X[:, 0]
    return np.stack([f1, g * (1 - np.sqrt(f1 / g))], 1)


def run_native(fun, lower, upper, nsga_update, evals, seed):
    m = MODE_C(2, 0, Bounds(lower, upper), popsize=64, nsga_update=nsga_update,
               rg=np.random.default_rng(seed))
    n = 0
    while n < evals:
        xs = np.asarray(m.ask())
        m.tell(np.ascontiguousarray(fun(xs)))
        n += len(xs)
    pop = np.asarray(m.population())
    return pop if pop.shape[1] == len(lower) else pop.T


def test_pareto_levels_rank_successive_fronts():
    ys = np.array([[1, 4], [2, 2], [4, 1], [2, 4], [3, 3], [4, 4]], dtype=float)
    # front 1: first three, front 2: (2,4) and (3,3), front 3: (4,4); higher = better
    assert list(mode.pareto_levels(ys)) == [2, 2, 2, 1, 1, 0]


def test_crowd_dist_uses_all_objectives():
    # identical first objective, spread only in the second one
    ys = np.array([[1.0, 0.0], [1.0, 0.1], [1.0, 0.2], [1.0, 1.0]])
    cd = mode.crowd_dist(ys)
    assert np.isinf(cd[0]) and np.isinf(cd[3])
    assert cd[2] > cd[1] > 0   # 0.2 has the larger gap to its neighbours


def test_sbx_can_expand_beyond_parents():
    dim = 200
    pop = np.vstack([np.full(dim, 0.4), np.full(dim, 0.6)])
    off = mode.variation(pop, np.zeros(dim), np.ones(dim), np.random.default_rng(1),
                         pro_c=1.0, dis_c=20.0, pro_m=0.0, dis_m=20.0)
    outside = (off < 0.4 - 1e-12) | (off > 0.6 + 1e-12)
    assert outside.any()


def test_native_nsga_update_keeps_spread_on_zdt2():
    dim = 30
    for seed in (1000, 1001, 1004):  # seeds that collapsed to x0 = 0 before the fix
        pop = run_native(zdt2, np.zeros(dim), np.ones(dim), True, 30000, seed)
        assert np.ptp(pop[:, 0]) > 0.5


def test_native_de_update_converges_on_zdt4():
    dim = 10
    lower = np.array([0.0] + [-5.0] * (dim - 1))
    upper = np.array([1.0] + [5.0] * (dim - 1))
    pop = run_native(zdt4, lower, upper, False, 60000, 1000)
    assert np.median(zdt4_g(pop)) < 5.0   # was ~170 before the fix


def test_variation_stays_finite_and_in_bounds():
    # expanding SBX can leave the box; polynomial mutation must see clipped values
    rg = np.random.default_rng(3)
    dim = 30
    pop = rg.random((64, dim))
    pop[:8, :] = 0.0
    pop[8:16, :] = 1.0
    off = mode.variation(pop, np.zeros(dim), np.ones(dim), rg,
                         pro_c=1.0, dis_c=15.0, pro_m=float(dim), dis_m=20.0)
    assert np.all(np.isfinite(off))
    assert off.min() >= 0.0 and off.max() <= 1.0


def l4(X):
    # min x^2 s.t. 1 - x <= 0 on [-2, 3]; the optimum x = 1 lies on the constraint
    x = np.atleast_2d(X)[:, 0]
    return np.stack([x * x, 1.0 - x], 1)


def best_feasible_l4(pop):
    ys = l4(pop)
    feasible = ys[:, 1] <= 0
    return ys[feasible, 0].min() if feasible.any() else np.inf


def test_native_constrained_population_keeps_feasible_points():
    # an adaptive constraint tolerance let slightly infeasible points displace
    # every strictly feasible one once the population reached the constraint
    for nsga_update in (True, False):
        for seed in (1000, 1001, 1002):
            m = MODE_C(1, 1, Bounds([-2.0], [3.0]), popsize=32, nsga_update=nsga_update,
                       rg=np.random.default_rng(seed))
            for _ in range(40):
                xs = np.asarray(m.ask())
                m.tell(np.ascontiguousarray(l4(xs)))
            pop = np.asarray(m.population())
            pop = pop if pop.shape[1] == 1 else pop.T
            assert best_feasible_l4(pop) < 1.05, (nsga_update, seed)


def test_python_constrained_population_keeps_feasible_points():
    for nsga_update in (True, False):
        for seed in (1000, 1001, 1002):
            m = mode.MODE(1, 1, Bounds([-2.0], [3.0]), popsize=32, nsga_update=nsga_update,
                          rg=np.random.default_rng(seed))
            for _ in range(40):
                xs = np.asarray(m.ask())
                m.tell(l4(xs))
            assert best_feasible_l4(m.x[:m.popsize]) < 1.05, (nsga_update, seed)
