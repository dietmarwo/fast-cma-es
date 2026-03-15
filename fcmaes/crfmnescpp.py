# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""Eigen based implementation of Fast Moving Natural Evolution Strategy
for High-Dimensional Problems (CR-FM-NES).
"""

import os
import sys

import numpy as np
np.set_printoptions(legacy='1.25') 
from numpy.random import Generator, PCG64DXSM
from numpy.typing import ArrayLike
from scipy.optimize import Bounds, OptimizeResult

from typing import Callable, Optional, Tuple, Union

from fcmaes._fcmaes_ext import CRFMNES as _CRFMNES
from fcmaes._fcmaes_ext import optimize_crfmnes
from fcmaes.evaluator import _check_bounds, _get_bounds, parallel, serial

Objective = Callable[[ArrayLike], float]
BatchObjective = Callable[[ArrayLike], np.ndarray]
NativeResult = Tuple[np.ndarray, float, int, int, int]
SigmaArg = Optional[Union[float, ArrayLike, Callable]]


def _result_tuple_to_result(values: NativeResult) -> OptimizeResult:
    x, val, evals, iterations, stop = values
    return OptimizeResult(
        x=x,
        fun=val,
        nfev=evals,
        nit=iterations,
        status=stop,
        success=True,
    )


def _prepare_batch_fun(
    fun: Objective, workers: Optional[int]
) -> Tuple[BatchObjective, Optional[parallel]]:
    parfun = None if (workers is None or workers <= 1) else parallel(fun, workers)
    base = serial(fun) if parfun is None else parfun

    def batch_fun(xs: ArrayLike) -> np.ndarray:
        return np.ascontiguousarray(base(xs), dtype=np.float64)

    return batch_fun, parfun


def minimize(
    fun: Objective,
    bounds: Optional[Bounds] = None,
    x0: Optional[ArrayLike] = None,
    input_sigma: SigmaArg = 0.3,
    popsize: Optional[int] = 32,
    max_evaluations: Optional[int] = 100000,
    workers: Optional[int] = None,
    stop_fitness: Optional[float] = -np.inf,
    rg: Optional[Generator] = Generator(PCG64DXSM()),
    runid: Optional[int] = 0,
    normalize: Optional[bool] = False,
    use_constraint_violation: Optional[bool] = True,
    penalty_coef: Optional[float] = 1e5,
) -> OptimizeResult:
    """Minimization using the nanobind-backed C++ CR-FM-NES implementation."""

    lower, upper, guess = _check_bounds(bounds, x0, rg)
    if popsize is None:
        popsize = 32
    if popsize % 2 == 1:
        popsize += 1
    if callable(input_sigma):
        input_sigma = input_sigma()
    if np.ndim(input_sigma) > 0:
        input_sigma = float(np.mean(input_sigma))

    guess = np.ascontiguousarray(guess, dtype=np.float64)
    if lower is None:
        lower = np.empty(0, dtype=np.float64)
        upper = np.empty(0, dtype=np.float64)
    else:
        lower = np.ascontiguousarray(lower, dtype=np.float64)
        upper = np.ascontiguousarray(upper, dtype=np.float64)

    batch_fun, parfun = _prepare_batch_fun(fun, workers)
    try:
        return _result_tuple_to_result(
            optimize_crfmnes(
                batch_fun,
                guess,
                lower,
                upper,
                sigma=float(input_sigma),
                seed=int(rg.uniform(0, 2**32 - 1)),
                runid=runid,
                max_evaluations=max_evaluations,
                stop_fitness=stop_fitness,
                popsize=popsize,
                penalty_coef=penalty_coef,
                use_constraint_violation=use_constraint_violation,
                normalize=normalize,
            )
        )
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
        if parfun is not None:
            parfun.stop()


class CRFMNES_C:
    def __init__(
        self,
        dim: int,
        bounds: Optional[Bounds] = None,
        x0: Optional[ArrayLike] = None,
        input_sigma: SigmaArg = 0.3,
        popsize: Optional[int] = 32,
        rg: Optional[Generator] = Generator(PCG64DXSM()),
        runid: Optional[int] = 0,
        normalize: Optional[bool] = False,
        use_constraint_violation: Optional[bool] = True,
        penalty_coef: Optional[float] = 1e5,
    ) -> None:
        """Ask/tell CR-FM-NES state backed by nanobind."""

        lower, upper, guess = _get_bounds(dim, bounds, x0, rg)
        if popsize is None:
            popsize = 32
        if popsize % 2 == 1:
            popsize += 1
        if callable(input_sigma):
            input_sigma = input_sigma()
        if np.ndim(input_sigma) > 0:
            input_sigma = float(np.mean(input_sigma))

        guess = np.ascontiguousarray(guess, dtype=np.float64)
        if lower is None:
            lower = np.empty(0, dtype=np.float64)
            upper = np.empty(0, dtype=np.float64)
        else:
            lower = np.ascontiguousarray(lower, dtype=np.float64)
            upper = np.ascontiguousarray(upper, dtype=np.float64)

        self._native = _CRFMNES(
            guess,
            lower,
            upper,
            sigma=float(input_sigma),
            popsize=popsize,
            seed=int(rg.uniform(0, 2**32 - 1)),
            runid=runid,
            penalty_coef=penalty_coef,
            use_constraint_violation=use_constraint_violation,
            normalize=normalize,
        )
        self.dim = self._native.dim
        self.popsize = self._native.popsize

    def ask(self) -> Optional[np.ndarray]:
        try:
            return self._native.ask()
        except Exception:
            return None

    def tell(self, ys: np.ndarray) -> int:
        try:
            return self._native.tell(np.ascontiguousarray(ys, dtype=np.float64))
        except Exception:
            return -1

    def population(self) -> Optional[np.ndarray]:
        try:
            return self._native.population()
        except Exception:
            return None

    def result(self) -> OptimizeResult:
        try:
            return _result_tuple_to_result(self._native.result())
        except Exception:
            return OptimizeResult(
                x=None,
                fun=sys.float_info.max,
                nfev=0,
                nit=0,
                status=-1,
                success=False,
            )
