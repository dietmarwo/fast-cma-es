# Copyright (c) Dietmar Wolz.
# Licensed under the MIT license in the repository root.

"""Native C++ jSO with the same public API as :mod:`fcmaes.jso`."""

import numpy as np

from fcmaes._fcmaes_ext import Jso as _Jso
from fcmaes.jso import _minimize, _result, _setup


def _values(values):
    values = np.asarray(values, dtype=np.float64, order="C")
    if values.ndim != 1:
        raise ValueError("jSO needs a one-dimensional vector of asked fitness values")
    return values


class Jso:
    """Synchronous ask/tell using the native C++ search state.

    Bounds, parameter defaults and budget behavior match ``fcmaes.jso.Jso``.
    Batches and population snapshots are independent NumPy arrays.
    """

    def __init__(self, bounds, popsize=0, max_evaluations=100_000,
                 stop_fitness=-np.inf, rg=None, runid=0, *, min_popsize=4,
                 memory_size=5, p_max=0.25, p_min=0.125, archive_rate=1.0):
        lower, upper, seed, options = _setup(
            bounds, popsize, max_evaluations, stop_fitness, rg, runid,
            min_popsize, memory_size, p_max, p_min, archive_rate)
        self._stop_fitness = options['stop_fitness']
        self._native = _Jso(lower, upper, seed=seed, **options)

    @property
    def dim(self):
        return self._native.dim

    @property
    def popsize(self):
        return self._native.popsize

    def ask(self):
        """Return the next variable-length batch, or an empty batch when done."""
        return self._native.ask()

    def tell(self, values):
        """Submit one objective value per asked row, in order."""
        return self._native.tell(_values(values))

    def _tell_prefix(self, values):
        return self._native._tell_prefix(_values(values))

    def population(self):
        """Return a copy of the current parent population."""
        return self._native.population()

    def result(self):
        """Return the best evaluated point and actual evaluation count."""
        return _result(*self._native.result())


def minimize(fun, bounds, popsize=0, max_evaluations=100_000,
             stop_fitness=-np.inf, workers=None, rg=None, runid=0, *,
             min_popsize=4, memory_size=5, p_max=0.25, p_min=0.125, archive_rate=1.0, vectorized=False):
    """Minimize with native jSO and optional process-parallel evaluation.

    The evaluation cap, target stops, and failure handling match
    ``fcmaes.jso.minimize`` for scalar objectives. With vectorized=True,
    fun takes a matrix and returns one score per candidate. This mode uses
    one worker and counts every evaluated row of a batch before a target stop.
    """
    state = Jso(bounds, popsize, max_evaluations, stop_fitness, rg, runid,
                min_popsize=min_popsize, memory_size=memory_size, p_max=p_max,
                p_min=p_min, archive_rate=archive_rate)
    if vectorized:
        if workers is not None and workers > 1:
            raise ValueError("vectorized objectives require one worker")
        while len(xs := state.ask()):
            if state.tell(fun(xs)):
                break
        return state.result()
    return _minimize(state, fun, workers)
