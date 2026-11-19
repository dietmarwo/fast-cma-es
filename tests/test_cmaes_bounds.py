"""CMA-ES on problems whose samples keep hitting the bounds.

Samples outside the box are evaluated at their repaired (clamped) point. The
update must use the z of the repaired point; with the z of the unrepaired
sample the step size grew until the run stopped (stop 3 or 5) after about
1000 evaluations, far from the optimum.
"""

import numpy as np
import pytest
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds

from fcmaes import cmaes, cmaescpp

DIM = 20
BOUNDS = Bounds([-1.0] * DIM, [1.0] * DIM)
# Half of the sphere's optimum lies outside the box.
CENTER = np.array([2.0] * 10 + [0.3] * 10)
SEEDS = (1, 2, 3)
EVALS = 20000


def linear(x):
    return float(np.sum(x))  # constrained optimum -20


def shifted_sphere(x):
    return float(np.sum((np.asarray(x) - CENTER) ** 2))  # constrained optimum 10


def run_minimize(backend, fun, seed):
    kwargs = dict(input_sigma=0.3, popsize=16, max_evaluations=EVALS,
                  rg=Generator(PCG64DXSM(seed)))
    if backend == "cpp":
        r = cmaescpp.minimize(fun, BOUNDS, np.zeros(DIM), **kwargs)
        return r.fun, r.nfev
    state = np.random.get_state()  # cmaes.minimize samples with np.random.randn
    try:
        np.random.seed(seed)
        r = cmaes.minimize(fun, BOUNDS, np.zeros(DIM), **kwargs)
    finally:
        np.random.set_state(state)
    return r.fun, r.nfev


def make_es(backend, seed, bounds=BOUNDS, x0=np.zeros(DIM), popsize=16):
    rg = Generator(PCG64DXSM(seed))
    if backend == "cpp":
        return cmaescpp.ACMA_C(len(x0), bounds, x0, input_sigma=0.3,
                               popsize=popsize, max_evaluations=EVALS, rg=rg)
    sampler = Generator(PCG64DXSM(seed + 1000))
    return cmaes.Cmaes(bounds, x0, input_sigma=0.3, popsize=popsize,
                       max_evaluations=EVALS, rg=rg,
                       randn=lambda *shape: sampler.standard_normal(shape))


@pytest.mark.parametrize("backend", ["cpp", "py"])
# Limits lie between the results with the repaired z (about -19.9 and 10.3 to
# 11.1, depending on platform and numpy version) and those without it (-14 to
# -16 and 13.6 to 16.9).
@pytest.mark.parametrize("fun,limit", [(linear, -18.5), (shifted_sphere, 12.5)],
                         ids=["linear", "shifted_sphere"])
def test_minimize_reaches_the_bounded_optimum(backend, fun, limit):
    for seed in SEEDS:
        y, nfev = run_minimize(backend, fun, seed)
        assert y < limit, f"seed {seed}: y={y}, nfev={nfev}"
        assert nfev > 2000, f"seed {seed}: stopped early after {nfev}"


@pytest.mark.parametrize("backend", ["cpp", "py"])
def test_ask_tell_reaches_the_bounded_optimum(backend):
    for seed in SEEDS:
        es = make_es(backend, seed)
        best, nfev = np.inf, 0
        while nfev < EVALS:
            xs = es.ask()
            ys = np.array([shifted_sphere(x) for x in xs])
            nfev += len(ys)
            best = min(best, ys.min())
            if es.tell(ys) != 0:
                break
        assert best < 12.5, f"seed {seed}: y={best}, nfev={nfev}"
        assert nfev > 2000, f"seed {seed}: stopped early after {nfev}"


@pytest.mark.parametrize("backend", ["cpp", "py"])
def test_tell_with_points_after_ask_updates_from_the_points(backend):
    # tell(ys, xs) after an ask must update from xs, not from the z of the
    # asked samples: telling the asked points in a different order must not
    # change the search distribution.
    def rosen(x):
        x = np.asarray(x)
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    dim, popsize = 5, 10
    bounds = Bounds([-2.0] * dim, [2.0] * dim)
    perm = np.random.default_rng(0).permutation(popsize)
    states = []
    for order in (None, perm):
        es = make_es(backend, 5, bounds, np.full(dim, 0.5), popsize)
        for _ in range(3 if backend == "cpp" else 1):
            xs = es.ask()
            ys = np.array([rosen(x) for x in xs])
            if order is not None:
                xs, ys = xs[order], ys[order]
            es.tell(ys, xs)
        if backend == "cpp":
            states.append([es.ask()])
        else:
            # Compare the distribution after one update, not later samples:
            # samples also depend on the eigenbasis of C, which is not
            # stable under rounding while eigenvalues are nearly equal, and
            # the rounding of the BLAS product depends on the column order.
            states.append([es.xmean.copy(), np.array(es.sigma), es.C.copy()])
    for a, b in zip(*states):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
