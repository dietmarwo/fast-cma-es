"""PGPE search: start point of the center and the stop test.

With ``normalize=True`` the center is kept in encoded coordinates
(the bounds mapped to [-1, 1]), and ADAM updates it there. The first
generation after the start is therefore sampled near the start point.

Bound used below, for bounds [-5, 5], start point x0 = 1, input_sigma = 0.1
(encoded; 0.5 in raw coordinates) and default parameters:

- encoded start c0 = 0.2;
- the first ADAM step changes each coordinate by
  center_learning_rate * |g| / (|g| + eps) < 0.15, so the encoded center c
  of the second generation lies in [0.05, 0.35];
- a mirrored pair c + n, c - n, clipped to [-1, 1], has its midpoint between
  0 and c for every n, so the mean of the second generation lies in
  [0, 0.35] encoded, [0, 1.75] raw: at most 1.0 = 2 initial standard
  deviations from x0, for any random stream.

The tests allow 2.5 initial standard deviations (1.25).
"""

import numpy as np
import pytest
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds

from fcmaes import pgpecpp

LOWER, UPPER = -5.0, 5.0
X0 = 1.0
SIGMA = 0.1
SDEV0_RAW = SIGMA * (UPPER - LOWER) / 2
MAX_DEVIATION = 2.5 * SDEV0_RAW
POPSIZE = 16
SEEDS = (1, 2, 3, 4, 5)


def shifted_sphere(shift):
    return lambda x: float(np.sum((np.asarray(x) - shift) ** 2))


def bounds(dim):
    return Bounds([LOWER] * dim, [UPPER] * dim)


@pytest.mark.parametrize("shift", [-3.0, 3.0])
@pytest.mark.parametrize("use_ranking", [True, False])
@pytest.mark.parametrize("dim", [1, 4])
@pytest.mark.parametrize("seed", SEEDS)
def test_ask_tell_center_stays_near_start(seed, dim, use_ranking, shift):
    fun = shifted_sphere(shift)
    es = pgpecpp.PGPE_C(dim, bounds(dim), np.full(dim, X0), SIGMA,
                        popsize=POPSIZE, rg=Generator(PCG64DXSM(seed)),
                        use_ranking=use_ranking)
    for _ in range(2):
        xs = es.ask()
        es.tell(np.array([fun(x) for x in xs]))
    # mean of the second generation
    mean = np.mean(es.population(), axis=0)
    assert np.all(np.abs(mean - X0) <= MAX_DEVIATION), mean


@pytest.mark.parametrize("shift", [-3.0, 3.0])
@pytest.mark.parametrize("use_ranking", [True, False])
@pytest.mark.parametrize("dim", [1, 4])
@pytest.mark.parametrize("seed", SEEDS)
def test_minimize_center_stays_near_start(seed, dim, use_ranking, shift):
    fun = shifted_sphere(shift)
    xs = []

    def recorder(x):
        xs.append(np.array(x, dtype=np.float64))
        return fun(x)

    pgpecpp.minimize(recorder, bounds(dim), np.full(dim, X0), SIGMA,
                     popsize=POPSIZE, max_evaluations=2 * POPSIZE,
                     rg=Generator(PCG64DXSM(seed)), use_ranking=use_ranking)
    assert len(xs) == 2 * POPSIZE
    mean = np.mean(xs[POPSIZE:], axis=0)
    assert np.all(np.abs(mean - X0) <= MAX_DEVIATION), mean
