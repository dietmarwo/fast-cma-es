"""CMA-ES on problems whose samples keep hitting the bounds.

Samples outside the box are evaluated at their repaired (clamped) point. The
update must use the z of the repaired point; with the z of the unrepaired
sample the step size grew until the run stopped (stop 3 or 5) after about
1000 evaluations, far from the optimum. ``repair_z=False`` restores that
update for comparison.
"""

import numpy as np
import pytest
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds, OptimizeResult

from fcmaes import cmaes, cmaescpp, optimizer

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


def run_minimize(backend, fun, seed, **options):
    kwargs = dict(input_sigma=0.3, popsize=16, max_evaluations=EVALS,
                  rg=Generator(PCG64DXSM(seed)), **options)
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


def make_es(backend, seed, bounds=BOUNDS, x0=np.zeros(DIM), popsize=16,
            **options):
    rg = Generator(PCG64DXSM(seed))
    if backend == "cpp":
        return cmaescpp.ACMA_C(len(x0), bounds, x0, input_sigma=0.3,
                               popsize=popsize, max_evaluations=EVALS, rg=rg,
                               **options)
    sampler = Generator(PCG64DXSM(seed + 1000))
    return cmaes.Cmaes(bounds, x0, input_sigma=0.3, popsize=popsize,
                       max_evaluations=EVALS, rg=rg,
                       randn=lambda *shape: sampler.standard_normal(shape),
                       **options)


def run_ask_tell(backend, fun, seed, max_evals=EVALS, **options):
    es = make_es(backend, seed, **options)
    best, nfev, stop = np.inf, 0, 0
    while nfev < max_evals:
        xs = es.ask()
        ys = np.array([fun(x) for x in xs])
        nfev += len(ys)
        best = min(best, ys.min())
        stop = es.tell(ys)
        if stop != 0:
            break
    return best, nfev, stop


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


# repair_z=False restores the update with the z of the unrepaired samples.
# Those runs stopped after 800 to 1200 evaluations (repaired z: 2300 to 4300)
# at -13 to -16.4 on linear and 12.3 to 16.9 on the sphere (repaired z: about
# -19.9 and 10.2 to 11.1). The limits keep a wide margin to both; y is
# checked as the mean over the seeds, single runs scatter more.
@pytest.mark.parametrize("entry", ["cpp_minimize", "cpp_ask_tell",
                                   "py_ask_tell"])
@pytest.mark.parametrize("fun,limit", [(linear, -18.0), (shifted_sphere, 12.0)],
                         ids=["linear", "shifted_sphere"])
def test_without_repair_z_stops_early_far_from_the_optimum(entry, fun, limit):
    ys = []
    for seed in SEEDS:
        if entry == "cpp_minimize":
            y, nfev = run_minimize("cpp", fun, seed, repair_z=False)
        else:
            y, nfev, _ = run_ask_tell(entry[:entry.index("_")], fun, seed,
                                      repair_z=False)
        assert nfev < 1800, f"seed {seed}: y={y}, nfev={nfev}"
        ys.append(y)
    assert np.mean(ys) > limit, f"ys={ys}"


@pytest.mark.parametrize("entry", ["cpp_minimize", "cpp_ask_tell",
                                   "py_ask_tell"])
def test_repair_z_defaults_to_true(entry):
    # Same computation in the same order, so the results are identical.
    results = []
    for options in ({}, {"repair_z": True}):
        if entry == "cpp_minimize":
            results.append(run_minimize("cpp", shifted_sphere, 1, **options))
        else:
            results.append(run_ask_tell(entry[:entry.index("_")],
                                        shifted_sphere, 1, max_evals=1600,
                                        **options))
    assert results[0] == results[1]


def test_cma_wrappers_forward_repair_z(monkeypatch):
    seen = []

    def fake_minimize(*args, **kwargs):
        seen.append(kwargs.get("repair_z", "missing"))
        return OptimizeResult(x=np.zeros(2), fun=0.0, nfev=1)

    monkeypatch.setattr(optimizer.cmaescpp, "minimize", fake_minimize)
    monkeypatch.setattr(optimizer.cmaes, "minimize", fake_minimize)
    bounds = Bounds([-1.0] * 2, [1.0] * 2)
    opts = [optimizer.Cma_cpp(100), optimizer.Cma_cpp(100, repair_z=False),
            optimizer.Cma_python(100), optimizer.Cma_python(100, repair_z=False),
            optimizer.de_cma(1500).optimizers[1],
            optimizer.de_cma(1500, repair_z=False).optimizers[1],
            optimizer.da_cma(1500, repair_z=False).optimizers[1],
            optimizer.bite_cma(1500, repair_z=False).optimizers[1],
            optimizer.cma_bite(1500, repair_z=False).optimizers[0]]
    for opt in opts:
        opt.minimize(linear, bounds, np.zeros(2), 0.3)
    assert seen == [True, False, True, False, True, False, False, False, False]
