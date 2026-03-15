# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""Eigen based active CMA-ES exposed through nanobind."""

import os
import sys
import numpy as np
np.set_printoptions(legacy='1.25') 
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds, OptimizeResult

from fcmaes._fcmaes_ext import ACMA as _ACMA
from fcmaes._fcmaes_ext import optimize_acma
from fcmaes.evaluator import _check_bounds, _get_bounds, parallel, libcmalib

from typing import Callable, Optional, Tuple, Union
from numpy.typing import ArrayLike

Objective = Callable[[ArrayLike], float]
NativeResult = Tuple[np.ndarray, float, int, int, int]
SigmaArg = Optional[Union[float, ArrayLike, Callable]]


def _as_optional_vector(values: Optional[ArrayLike]) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    return np.ascontiguousarray(values, dtype=np.float64)


def _prepare_sigma(dim: int,
                   input_sigma: SigmaArg
                   ) -> np.ndarray:
    sigma = input_sigma
    if callable(sigma):
        sigma = sigma()
    if np.ndim(sigma) == 0:
        sigma = [sigma] * dim
    return np.ascontiguousarray(sigma, dtype=np.float64)


def _result_from_native(result: NativeResult) -> OptimizeResult:
    x, val, evals, iterations, stop = result
    return OptimizeResult(
        x=x,
        fun=val,
        nfev=evals,
        nit=iterations,
        status=stop,
        success=True,
    )


def minimize(fun: Objective,
             bounds: Optional[Bounds] = None,
             x0: Optional[ArrayLike] = None,
             input_sigma: SigmaArg = 0.3,
             popsize: Optional[int] = 31,
             max_evaluations: Optional[int] = 100000,
             accuracy: Optional[float] = 1.0,
             stop_fitness: Optional[float] = -np.inf,
             stop_hist: Optional[float] = -1,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             runid: Optional[int] = 0,
             workers: Optional[int] = 1,
             normalize: Optional[bool] = True,
             delayed_update: Optional[bool] = True,
             update_gap: Optional[int] = None) -> OptimizeResult:
    """Minimize a scalar objective with the native ACMA implementation."""

    lower, upper, guess = _check_bounds(bounds, x0, rg)
    dim = guess.size
    popsize = 31 if popsize is None else popsize
    max_evaluations = 100000 if max_evaluations is None else max_evaluations
    accuracy = 1.0 if accuracy is None else accuracy
    stop_fitness = -np.inf if stop_fitness is None else stop_fitness
    stop_hist = -1 if stop_hist is None else stop_hist
    workers = 0 if workers is None else workers
    normalize = True if normalize is None else normalize
    delayed_update = True if delayed_update is None else delayed_update
    update_gap = -1 if update_gap is None else update_gap
    runid = 0 if runid is None else runid
    rg = Generator(PCG64DXSM()) if rg is None else rg
    mu = int(popsize / 2)

    batch_fun = None
    try:
        if workers > 1 and not delayed_update:
            batch_fun = parallel(fun, workers)
        result = optimize_acma(
            fun,
            batch_fun,
            np.ascontiguousarray(guess, dtype=np.float64),
            _as_optional_vector(lower),
            _as_optional_vector(upper),
            _prepare_sigma(dim, input_sigma),
            seed=int(rg.uniform(0, 2**32 - 1)),
            runid=runid,
            max_evaluations=max_evaluations,
            stop_fitness=stop_fitness,
            stop_hist=stop_hist,
            mu=mu,
            popsize=popsize,
            accuracy=accuracy,
            normalize=normalize,
            delayed_update=delayed_update,
            update_gap=update_gap,
            workers=workers,
        )
        return _result_from_native(result)
    except Exception:
        return OptimizeResult(
            x=None,
            fun=sys.float_info.max,
            nfev=0,
            nit=0,
            status=-1,
            success=False,
        )
    finally:
        if batch_fun is not None:
            batch_fun.stop()


class ACMA_C:

    def __init__(self,
                 dim: int,
                 bounds: Optional[Bounds] = None,
                 x0: Optional[ArrayLike] = None,
                 input_sigma: SigmaArg = 0.3,
                 popsize: Optional[int] = 31,
                 max_evaluations: Optional[int] = 100000,
                 accuracy: Optional[float] = 1.0,
                 stop_fitness: Optional[float] = -np.inf,
                 stop_hist: Optional[float] = -1,
                 rg: Optional[Generator] = Generator(PCG64DXSM()),
                 runid: Optional[int] = 0,
                 normalize: Optional[bool] = True,
                 delayed_update: Optional[bool] = True,
                 update_gap: Optional[int] = None) -> None:
        lower, upper, guess = _get_bounds(dim, bounds, x0, rg)
        popsize = 31 if popsize is None else popsize
        max_evaluations = 100000 if max_evaluations is None else max_evaluations
        accuracy = 1.0 if accuracy is None else accuracy
        stop_fitness = -np.inf if stop_fitness is None else stop_fitness
        stop_hist = -1 if stop_hist is None else stop_hist
        normalize = True if normalize is None else normalize
        delayed_update = True if delayed_update is None else delayed_update
        update_gap = -1 if update_gap is None else update_gap
        runid = 0 if runid is None else runid
        rg = Generator(PCG64DXSM()) if rg is None else rg
        mu = int(popsize / 2)

        self._native = _ACMA(
            np.ascontiguousarray(guess, dtype=np.float64),
            _as_optional_vector(lower),
            _as_optional_vector(upper),
            _prepare_sigma(dim, input_sigma),
            max_evaluations=max_evaluations,
            stop_fitness=stop_fitness,
            stop_hist=stop_hist,
            mu=mu,
            popsize=popsize,
            accuracy=accuracy,
            seed=int(rg.uniform(0, 2**32 - 1)),
            runid=runid,
            normalize=normalize,
            delayed_update=delayed_update,
            update_gap=update_gap,
        )
        self.popsize = self._native.popsize
        self.dim = self._native.dim

    def ask(self) -> Optional[np.ndarray]:
        try:
            return self._native.ask()
        except Exception:
            return None

    def tell(self, ys: np.ndarray, xs: Optional[np.ndarray] = None) -> int:
        if xs is not None:
            return self.tell_x_(ys, xs)
        try:
            return self._native.tell(np.ascontiguousarray(ys, dtype=np.float64))
        except Exception:
            return -1

    def tell_x_(self, ys: np.ndarray, xs: np.ndarray) -> int:
        try:
            return self._native.tell_x(
                np.ascontiguousarray(ys, dtype=np.float64),
                np.ascontiguousarray(xs, dtype=np.float64),
            )
        except Exception:
            return -1

    def population(self) -> Optional[np.ndarray]:
        try:
            return self._native.population()
        except Exception:
            return None

    def result(self) -> OptimizeResult:
        try:
            return _result_from_native(self._native.result())
        except Exception:
            return OptimizeResult(
                x=None,
                fun=sys.float_info.max,
                nfev=0,
                nit=0,
                status=-1,
                success=False,
            )
