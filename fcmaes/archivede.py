# Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
"""Archive-gated, rank-space differential evolution on bounded continuous problems.

The native C++ state (``_fcmaescpp/archivede.cpp``, derived from
``archive_de.cpp`` in Aleksey Vaneev's BiteOpt repository,
https://github.com/avaneev/biteopt, MIT license) implements DE/rand/1/bin
(F=0.5, CR=0.3) without constraint handling. Python evaluates objectives.
The returned archive retains each nondominated duplicate, including its
corresponding decision vector. With the same seed, fcmaes-rust's
``ArchiveDE`` performs the same run.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from operator import index

import numpy as np
from scipy.optimize import Bounds

from fcmaes._fcmaes_ext import ArchiveDE as _State


class ArchiveDE:
    """Synchronous ask/tell. Bounds are scipy Bounds or (lower, upper) pairs.

    seed accepts a uint32, a NumPy Generator, or None. The budget counts
    initialization and may end partway through a generation. gate controls
    offspring admission. tiered selects rank-space spread with objective
    champions. Setting tiered=False selects by stable Borda truncation.
    """
    def __init__(self, nobj, bounds, *, popsize=100, max_evaluations=100_000,
                 seed=None, gate=True, tiered=True):
        if isinstance(bounds, Bounds):
            lo, hi = bounds.lb, bounds.ub
        else:
            pairs = np.asarray(bounds, dtype=np.float64)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("expected Bounds or (lower, upper) pairs")
            lo, hi = pairs.T
        lo = np.require(lo, dtype=np.float64, requirements=["C", "A"])
        hi = np.require(hi, dtype=np.float64, requirements=["C", "A"])
        if lo.ndim != 1 or hi.ndim != 1:
            raise ValueError("bounds must be vectors")
        nobj, popsize, budget = index(nobj), index(popsize), index(max_evaluations)
        if nobj < 1 or popsize < 4 or popsize <= nobj or budget < 0:
            raise ValueError("require nobj >= 1, popsize >= 4, popsize > nobj and a nonnegative budget")
        if seed is None:
            seed = np.random.default_rng()
        if isinstance(seed, np.random.Generator):
            seed = seed.integers(0, 2**32, dtype=np.uint32)
        seed = index(seed)
        if not 0 <= seed < 2**32:
            raise ValueError("integer seed must fit uint32")
        self._native = _State(lo, hi, nobj, popsize, budget, seed, gate, tiered)

    def ask(self) -> np.ndarray:
        """Return a copy of the pending decision matrix, or (0, dim) when finished."""
        return self._native.ask()

    def tell(self, ys) -> None:
        """Submit all asked objective rows in order. Invalid values leave the batch pending."""
        ys = np.require(ys, dtype=np.float64, requirements=["C", "A"])
        if ys.ndim != 2:
            raise ValueError("expected a matrix of objective rows")
        self._native.tell(ys)

    def population(self) -> tuple[np.ndarray, np.ndarray]:
        """Return independent copies (xs, ys) of the evaluated breeding population."""
        return self._native.population()

    def front(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact archive (xs, ys) of all evaluations, including duplicates."""
        return self._native.front()

    @property
    def nfev(self) -> int:
        return self._native.nfev


def minimize(fun, nobj, bounds, *, popsize=100, max_evaluations=100_000,
             seed=None, gate=True, tiered=True, workers=1, vectorized=False):
    """Return the strict Pareto archive (xs, ys).

    By default fun(x) returns one length-nobj objective vector. With
    vectorized=True, fun(xs) returns the entire (rows, nobj) matrix. With
    workers > 1, an ordered process pool evaluates a pickleable row objective.
    Objective exceptions propagate to the caller.
    """
    workers = 1 if workers is None else index(workers)
    if workers < 1 or (vectorized and workers != 1):
        raise ValueError("use positive workers; vectorized evaluation uses one caller")
    state = ArchiveDE(nobj, bounds, popsize=popsize, max_evaluations=max_evaluations,
                      seed=seed, gate=gate, tiered=tiered)
    with (ProcessPoolExecutor(workers) if workers > 1 else nullcontext()) as pool:
        while len(xs := state.ask()):
            if vectorized:
                ys = fun(xs)
            elif pool is None:
                ys = [fun(x) for x in xs]
            else:
                ys = list(pool.map(fun, xs, chunksize=max(1, len(xs) // (4 * workers))))
            state.tell(ys)
    return state.front()
