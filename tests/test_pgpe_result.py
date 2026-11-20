"""The result of PGPE is the best evaluated point and its objective value.

``pgpecpp.minimize`` and ``PGPE_C.result()`` return ``x`` as the decoded point
passed to the objective and ``fun`` as the objective value at ``x``, with and
without ranking. With ranking the optimizer works on rank values, which are
never reported.
"""

import numpy as np
import pytest
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds

from fcmaes import pgpecpp

DIM = 5
BOUNDS = Bounds([-3.0] * DIM, [5.0] * DIM)
X0 = np.ones(DIM)
EVALS = 2000
SEEDS = (1, 2, 3)


def sphere(x):
    return float(np.sum((np.asarray(x) - 0.7) ** 2))


class Recorder:
    """Objective that records every evaluated point and value."""

    def __init__(self):
        self.xs = []
        self.ys = []

    def __call__(self, x):
        x = np.array(x, dtype=np.float64)
        y = sphere(x)
        self.xs.append(x)
        self.ys.append(y)
        return y


def check_result(x, fun, rec):
    x = np.asarray(x, dtype=np.float64)
    assert x.shape == (DIM,)
    assert np.all(np.isfinite(x))
    assert np.all(x >= BOUNDS.lb) and np.all(x <= BOUNDS.ub)
    assert sphere(x) == fun
    assert fun == min(rec.ys)
    assert any(np.array_equal(x, e) for e in rec.xs)


@pytest.mark.parametrize("use_ranking", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
def test_minimize_returns_best_evaluated_point(seed, use_ranking):
    rec = Recorder()
    r = pgpecpp.minimize(rec, BOUNDS, X0, input_sigma=0.5, popsize=16,
                         max_evaluations=EVALS,
                         rg=Generator(PCG64DXSM(seed)),
                         use_ranking=use_ranking)
    assert r.nfev == len(rec.ys) == EVALS
    check_result(r.x, r.fun, rec)


@pytest.mark.parametrize("use_ranking", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
def test_ask_tell_result_is_best_evaluated_point(seed, use_ranking):
    es = pgpecpp.PGPE_C(DIM, BOUNDS, X0, input_sigma=0.5, popsize=16,
                        rg=Generator(PCG64DXSM(seed)),
                        use_ranking=use_ranking)
    rec = Recorder()
    for _ in range(EVALS // es.popsize):
        xs = es.ask()
        es.tell(np.array([rec(x) for x in xs]))
    r = es.result()
    assert r.nfev == EVALS
    check_result(r.x, r.fun, rec)
