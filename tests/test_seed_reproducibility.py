"""A C++ optimizer run depends only on its seed and inputs.

The normal samplers of CMA-ES, DE (with ``x0`` and ``input_sigma``),
CR-FM-NES, PGPE and Dual Annealing use ``std::normal_distribution``, which
caches the second value of each generated pair. Each optimizer instance owns
that cache, so a run with the same seed gives the same ``x`` and ``f`` whatever
ran before it in the process. The run in between uses an odd number of normal
draws, so a cache shared between instances would shift the sample stream.
PGPE is checked through ``PGPE_C``.
"""

import threading

import numpy as np
import pytest
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds
from scipy.optimize import rosen as scipy_rosen

from fcmaes import cmaescpp, crfmnescpp, dacpp, decpp, pgpecpp

GENERATIONS = 30


def rosen(x):
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def bounds(dim):
    return Bounds([-5.0] * dim, [5.0] * dim)


def rg(seed):
    return Generator(PCG64DXSM(seed))


def x0(dim):
    return np.full(dim, 0.5)


def cma_minimize(seed, dim=5, popsize=13):
    r = cmaescpp.minimize(rosen, bounds(dim), x0(dim), input_sigma=1.0,
                          popsize=popsize, max_evaluations=40 * popsize,
                          rg=rg(seed))
    return r.x, r.fun


def de_minimize(seed, dim=5, popsize=13):
    r = decpp.minimize(rosen, dim, bounds(dim), popsize=popsize,
                       max_evaluations=40 * popsize, rg=rg(seed), x0=x0(dim),
                       input_sigma=1.0)
    return r.x, r.fun


def crfmnes_minimize(seed, dim=5, popsize=14):
    r = crfmnescpp.minimize(rosen, bounds(dim), x0(dim), input_sigma=1.0,
                            popsize=popsize, max_evaluations=40 * popsize,
                            rg=rg(seed))
    return r.x, r.fun


def da_minimize(seed, dim=5):
    r = dacpp.minimize(rosen, bounds(dim), x0(dim), max_evaluations=500,
                       rg=rg(seed))
    return r.x, r.fun


def make_cma(seed, dim, popsize=13):
    return cmaescpp.ACMA_C(dim, bounds(dim), x0(dim), input_sigma=1.0,
                           popsize=popsize, max_evaluations=10**6,
                           rg=rg(seed))


def make_de(seed, dim, popsize=13):
    return decpp.DE_C(dim, bounds(dim), popsize=popsize, rg=rg(seed),
                      x0=x0(dim), input_sigma=1.0)


def make_crfmnes(seed, dim, popsize=14):
    return crfmnescpp.CRFMNES_C(dim, bounds(dim), x0(dim), input_sigma=1.0,
                                popsize=popsize, rg=rg(seed))


def make_pgpe(seed, dim, popsize=14):
    return pgpecpp.PGPE_C(dim, bounds(dim), x0(dim), input_sigma=1.0,
                          popsize=popsize, rg=rg(seed))


def ask_tell(make, seed, dim=5, generations=GENERATIONS):
    es = make(seed, dim)
    best_x, best_y = None, np.inf
    for _ in range(generations):
        xs = es.ask()
        ys = np.array([rosen(x) for x in xs])
        es.tell(ys)
        i = int(np.argmin(ys))
        if ys[i] < best_y:
            best_x, best_y = np.array(xs[i]), float(ys[i])
    return best_x, best_y


MINIMIZE = {
    "cma": cma_minimize,
    "de": de_minimize,
    "crfmnes": crfmnes_minimize,
    "da": da_minimize,
}

ASK_TELL = {
    "cma": make_cma,
    "de": make_de,
    "crfmnes": make_crfmnes,
    "pgpe": make_pgpe,
}


def assert_same(a, b):
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1]


def other_run(name, dim):
    """One generation of another instance of the same optimizer.

    It draws an odd multiple of ``dim`` normal values, an odd number for one
    of the two dimensions 6 and 7. Dual Annealing draws normal values in
    pairs, so its test checks only that results repeat.
    """
    if name == "da":
        da_minimize(12, dim=dim)
    elif name == "crfmnes":
        ask_tell(lambda seed, d: make_crfmnes(seed, d, popsize=12), 12,
                 dim=dim, generations=1)
    else:
        ask_tell(ASK_TELL[name], 12, dim=dim, generations=1)


@pytest.mark.parametrize("name", sorted(MINIMIZE))
def test_minimize_same_seed_after_other_run(name):
    run = MINIMIZE[name]
    first = run(11)
    for dim in (7, 6):
        other_run(name, dim)
        assert_same(first, run(11))


@pytest.mark.parametrize("name", sorted(ASK_TELL))
def test_ask_tell_same_seed_after_other_run(name):
    make = ASK_TELL[name]
    first = ask_tell(make, 11)
    for dim in (7, 6):
        other_run(name, dim)
        assert_same(first, ask_tell(make, 11))


@pytest.mark.parametrize("name", sorted(ASK_TELL))
def test_ask_tell_interleaved_objects(name):
    """Two objects stepped alternately give the results of separate runs."""
    make = ASK_TELL[name]
    objects = [make(seed, 5) for seed in (11, 12)]
    last = [None, None]
    for _ in range(GENERATIONS):
        for k, es in enumerate(objects):
            xs = es.ask()
            es.tell(np.array([rosen(x) for x in xs]))
            last[k] = xs
    for k, seed in enumerate((11, 12)):
        es = make(seed, 5)
        for _ in range(GENERATIONS):
            xs = es.ask()
            es.tell(np.array([rosen(x) for x in xs]))
        assert np.array_equal(last[k], xs)


@pytest.mark.parametrize("name", sorted(ASK_TELL))
def test_ask_tell_objects_in_threads(name):
    """Ask/tell objects stepped in parallel Python threads."""
    make = ASK_TELL[name]
    seeds = (11, 12, 13, 14)
    expected = [ask_tell(make, seed, generations=100) for seed in seeds]
    results = [None] * len(seeds)

    def work(i):
        results[i] = ask_tell(make, seeds[i], generations=100)

    threads = [threading.Thread(target=work, args=(i,))
               for i in range(len(seeds))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for got, want in zip(results, expected):
        assert_same(got, want)


def test_cma_parallel_batch_same_seed_after_other_run():
    """``workers > 1`` with ``delayed_update=False`` evaluates each
    population in worker processes and returns the values in order, so the
    run is deterministic. (``delayed_update=True`` and parallel retry depend
    on the order in which results arrive and are not.)"""

    def run():
        r = cmaescpp.minimize(scipy_rosen, bounds(5), x0(5), input_sigma=1.0,
                              popsize=13, max_evaluations=520, rg=rg(11),
                              workers=2, delayed_update=False)
        return r.x, r.fun

    first = run()
    for dim in (7, 6):
        other_run("cma", dim)
        assert_same(first, run())
