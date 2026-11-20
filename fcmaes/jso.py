# Copyright (c) Dietmar Wolz.
# Licensed under the MIT license in the repository root.

"""Python jSO, a bounded adaptive differential evolution optimizer.

The algorithm follows the stock implementation in fcmaes-rust and the
description by Brest, Maučec and Bošković (CEC 2017, doi:10.1109/CEC.2017.7969456).
"""

from __future__ import annotations

from functools import partial
import math
from operator import index

import numpy as np
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds, OptimizeResult

from fcmaes.evaluator import parallel


def _setup(bounds, popsize, max_evaluations, stop_fitness, rg, runid,
           min_popsize, memory_size, p_max, p_min, archive_rate):
    if isinstance(bounds, Bounds):
        lower, upper = bounds.lb, bounds.ub
    else:
        pairs = np.asarray(bounds, dtype=np.float64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("jSO needs finite bounds as Bounds or (lower, upper) pairs")
        lower, upper = pairs.T
    lower = np.array(lower, dtype=np.float64, order="C", copy=True)
    upper = np.array(upper, dtype=np.float64, order="C", copy=True)
    if lower.ndim != 1 or lower.shape != upper.shape or not lower.size:
        raise ValueError("jSO needs matching, non-empty, one-dimensional bounds")
    with np.errstate(over="ignore", invalid="ignore"):
        width = upper - lower
    if (not np.all(np.isfinite(lower) & np.isfinite(upper) & np.isfinite(width))
            or np.any(width < 0)):
        raise ValueError("jSO bounds must be finite and ordered with finite widths")

    max_evaluations = index(max_evaluations)
    if not 0 < max_evaluations < 2**64:
        raise ValueError("max_evaluations must be a positive 64-bit count")
    stop_fitness = float(stop_fitness)
    if math.isnan(stop_fitness):
        raise ValueError("stop_fitness must not be NaN")
    p_min, p_max, archive_rate = float(p_min), float(p_max), float(archive_rate)
    if not 0 < p_min <= p_max <= 1:
        raise ValueError("jSO requires 0 < p_min <= p_max <= 1")
    if not math.isfinite(archive_rate) or archive_rate < 0:
        raise ValueError("archive_rate must be finite and non-negative")
    popsize = index(popsize)
    min_popsize = max(4, index(min_popsize))
    memory_size = max(2, index(memory_size))
    if popsize <= 0:
        dim = lower.size
        popsize = _round_positive(25 * math.log(dim) * math.sqrt(dim))
    popsize = max(popsize, min_popsize)
    if max(popsize, memory_size) > np.iinfo(np.int32).max:
        raise ValueError("population and memory sizes must fit in a signed 32-bit count")
    archive_size = archive_rate * popsize
    if not math.isfinite(archive_size) or archive_size > np.iinfo(np.int32).max:
        raise ValueError("archive capacity must fit in a signed 32-bit count")

    runid = index(runid)
    rg = Generator(PCG64DXSM()) if rg is None else rg
    seed = (int(rg.integers(0, 2**64, dtype=np.uint64)) + runid) % 2**64
    return lower, upper, seed, dict(
        popsize=popsize, min_popsize=min_popsize, max_evaluations=max_evaluations,
        stop_fitness=stop_fitness, memory_size=memory_size, p_max=p_max, p_min=p_min,
        archive_rate=archive_rate,
    )


def _round_positive(value):
    return math.floor(value + 0.5)


def _result(x, fun, nfev, nit, status):
    return OptimizeResult(x=x, fun=fun, nfev=nfev, nit=nit, status=status,
                          success=math.isfinite(fun))


def _score(fun, x):
    try:
        value = float(fun(x))
    except Exception:
        return np.inf
    return value if math.isfinite(value) else np.inf


def _minimize(state, fun, workers):
    score = partial(_score, fun)
    workers = 1 if workers is None else index(workers)
    evaluator = parallel(score, min(state.popsize, workers)) if workers > 1 else None
    try:
        while len(xs := state.ask()):
            if evaluator is not None:
                state.tell(evaluator(xs))
                continue
            values = []
            for x in xs:
                value = score(x)
                values.append(value)
                if value < state._stop_fitness:
                    break
            state._tell_prefix(values)
        return state.result()
    finally:
        if evaluator is not None:
            evaluator.stop()


def _lehmer(values, weights):
    """Fitness-weighted Lehmer mean with finite scaling of the weights.

    The sums are exactly rounded (``math.fsum``). ``np.dot`` goes through
    BLAS, whose result can depend on the memory alignment of its arguments,
    so that the same seed would not always repeat a run.
    """
    weighted = weights * values
    num = math.fsum(weighted * values)
    den = math.fsum(weighted)
    return num / den if den > 0 else None


class Jso:
    """Synchronous ask/tell for the Python jSO implementation.

    ``bounds`` is a SciPy ``Bounds`` or a sequence of ``(lower, upper)``
    pairs. Nonpositive ``popsize`` selects ``round(25 * ln(D) * sqrt(D))``.
    The population decreases linearly to ``min_popsize`` (at least four).
    ``max_evaluations`` is a hard cap, including initialization. A final
    batch can be shorter than the current population.
    """

    def __init__(self, bounds, popsize=0, max_evaluations=100_000,
                 stop_fitness=-np.inf, rg=None, runid=0, *, min_popsize=4,
                 memory_size=5, p_max=0.25, p_min=0.125, archive_rate=1.0):
        self._lower, self._upper, seed, options = _setup(
            bounds, popsize, max_evaluations, stop_fitness, rg, runid,
            min_popsize, memory_size, p_max, p_min, archive_rate)
        self._rng = Generator(PCG64DXSM(seed))
        self._initial_size = options['popsize']
        self._min_size = options['min_popsize']
        self._budget = options['max_evaluations']
        self._stop_fitness = options['stop_fitness']
        self._p_max, self._p_min = options['p_max'], options['p_min']
        self._archive_rate = options['archive_rate']
        self._memory_f = np.full(options['memory_size'], 0.3)
        self._memory_cr = np.full(options['memory_size'], 0.8)
        self._memory_index = 0
        self._archive = []
        width = self._upper - self._lower
        n = min(self._initial_size, self._budget)
        with np.errstate(over="ignore"):
            self._pop = np.clip(self._lower + self._rng.random((n, self.dim)) * width,
                                self._lower, self._upper)
        self._fitness = None
        self._pending = None
        self._best_x, self._best_y = None, np.inf
        self._evaluations = self._iterations = self._status = 0

    @property
    def dim(self):
        return self._lower.size

    @property
    def popsize(self):
        return len(self._pop)

    def ask(self):
        """Return a snapshot of the next batch, or an empty batch when done."""
        if self._pending is not None:
            raise RuntimeError("jSO ask called before telling the pending batch")
        remaining = self._budget - self._evaluations
        if self._status or not remaining:
            return np.empty((0, self.dim))
        self._pending = ((self._pop, None, None) if self._fitness is None else
                         self._trials(min(remaining, self.popsize)))
        return self._pending[0].copy()

    def tell(self, values):
        """Submit one objective value per asked row, in order."""
        return self._tell(values, prefix=False)

    def _tell_prefix(self, values):
        return self._tell(values, prefix=True)

    def population(self):
        """Return a copy of the current parent population."""
        return self._pop.copy()

    def result(self):
        """Return the best evaluated point and actual evaluation count."""
        x = self._best_x
        if x is None:
            x = self._lower + 0.5 * (self._upper - self._lower)
        return _result(x.copy(), self._best_y, self._evaluations, self._iterations, self._status)

    def _trials(self, length):
        rng = self._rng
        n = self.popsize
        progress = self._evaluations / self._budget
        p = self._p_max - (self._p_max - self._p_min) * progress
        order = np.argsort(self._fitness, kind='stable')
        top = max(2, _round_positive(p * n))
        fw = 0.7 if progress < 0.2 else (0.8 if progress < 0.4 else 1.2)
        ncells = len(self._memory_f)
        points = np.empty((length, self.dim))
        factors = np.empty(length)
        crossover = np.empty(length)
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(length):
                cell = rng.integers(ncells)
                mean_f, mean_cr = ((0.9, 0.9) if cell == ncells - 1 else
                                   (self._memory_f[cell], self._memory_cr[cell]))
                cr = 0.0 if mean_cr < 0 else float(np.clip(rng.normal(mean_cr, 0.1), 0, 1))
                if progress < 0.25:
                    cr = max(cr, 0.7)
                elif progress < 0.5:
                    cr = max(cr, 0.6)
                f = mean_f + 0.1 * rng.standard_cauchy()
                while not f > 0:
                    f = mean_f + 0.1 * rng.standard_cauchy()
                f = min(f, 0.7 if progress < 0.6 else 1.0)

                pbest = order[rng.integers(top)]
                while progress < 0.5 and pbest == i:
                    pbest = order[rng.integers(top)]
                r1 = rng.integers(n - 1)
                r1 += r1 >= i
                r2 = rng.integers(n + len(self._archive) - 2)
                r2 += r2 >= min(i, r1)
                r2 += r2 >= max(i, r1)
                donor = self._pop[r2] if r2 < n else self._archive[r2 - n]
                parent = self._pop[i]
                mutant = parent + (fw * f) * (self._pop[pbest] - parent) + f * (self._pop[r1] - donor)
                mutant = np.where(mutant < self._lower, self._lower + 0.5 * (parent - self._lower), mutant)
                mutant = np.where(mutant > self._upper, parent + 0.5 * (self._upper - parent), mutant)
                mutant = np.where(np.isnan(mutant), parent, mutant).clip(self._lower, self._upper)
                mask = rng.random(self.dim) <= cr
                mask[rng.integers(self.dim)] = True
                points[i] = np.where(mask, mutant, parent)
                factors[i], crossover[i] = f, cr
        return points, factors, crossover

    def _tell(self, values, prefix):
        if self._pending is None:
            raise RuntimeError("jSO tell called without a pending ask batch")
        points, factors, crossover = self._pending
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not 0 < len(values) <= len(points):
            raise ValueError("jSO needs a one-dimensional vector of asked fitness values")
        values = np.where(np.isfinite(values), values, np.inf)
        if len(values) != len(points) and (not prefix or not np.min(values) < self._stop_fitness):
            raise ValueError("jSO tell batch length must equal the asked batch length")

        best = int(np.argmin(values))
        if self._best_x is None or values[best] < self._best_y:
            self._best_x, self._best_y = points[best].copy(), float(values[best])
        self._evaluations += len(values)
        if self._fitness is None:
            self._pop = points[:len(values)]
            self._fitness = values
        else:
            self._update_population(points, values, factors, crossover)
            self._iterations += 1
        self._pending = None
        self._status = int(self._best_y < self._stop_fitness)
        return self._status

    def _update_population(self, points, values, factors, crossover):
        previous = self._fitness[:len(values)]
        improved = np.flatnonzero(values < previous)
        capacity = _round_positive(self._archive_rate * self.popsize)
        if improved.size:
            if capacity:
                for i in improved:
                    old = self._pop[i].copy()
                    if len(self._archive) < capacity:
                        self._archive.append(old)
                    else:
                        self._archive[self._rng.integers(capacity)] = old
            with np.errstate(over="ignore"):
                weights = previous[improved] - values[improved]
            infinite = np.isinf(weights)
            if infinite.any():
                weights = infinite.astype(np.float64)
            else:
                weights /= weights.max()
            cell = self._memory_index
            mf = _lehmer(factors[improved], weights)
            cr = crossover[improved]
            mcr = _lehmer(cr, weights) if cr.any() else -1.0
            if mf is not None:
                self._memory_f[cell] = 0.5 * (self._memory_f[cell] + mf)
            if mcr is not None:
                self._memory_cr[cell] = 0.5 * (self._memory_cr[cell] + mcr)
            self._memory_index = (cell + 1) % len(self._memory_f)

        accepted = np.flatnonzero(values <= previous)
        self._pop[accepted] = points[accepted]
        self._fitness[accepted] = values[accepted]
        size = _round_positive(self._initial_size + (self._min_size - self._initial_size)
                               * (self._evaluations / self._budget))
        if size < self.popsize:
            keep = np.argsort(self._fitness, kind='stable')[:size]
            self._pop, self._fitness = self._pop[keep], self._fitness[keep]
        capacity = _round_positive(self._archive_rate * self.popsize)
        while len(self._archive) > capacity:
            i = self._rng.integers(len(self._archive))
            self._archive[i] = self._archive[-1]
            self._archive.pop()


def minimize(fun, bounds, popsize=0, max_evaluations=100_000,
             stop_fitness=-np.inf, workers=None, rg=None, runid=0, *,
             min_popsize=4, memory_size=5, p_max=0.25, p_min=0.125, archive_rate=1.0):
    """Minimize a scalar objective with Python jSO.

    ``workers > 1`` evaluates each generation in parallel processes. Serial
    evaluation stops at the first value strictly below ``stop_fitness``;
    parallel evaluation finishes the batch. Both obey the hard budget.
    Non-finite values and objective exceptions count as failed evaluations.
    """
    state = Jso(bounds, popsize, max_evaluations, stop_fitness, rg, runid,
                min_popsize=min_popsize, memory_size=memory_size, p_max=p_max,
                p_min=p_min, archive_rate=archive_rate)
    return _minimize(state, fun, workers)
