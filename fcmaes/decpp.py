# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""Eigen based Differential Evolution exposed through nanobind."""

import os
import sys
import numpy as np
np.set_printoptions(legacy='1.25') 
from numpy.random import PCG64DXSM, Generator
from scipy.optimize import Bounds, OptimizeResult

from fcmaes._fcmaes_ext import DE as _DE
from fcmaes._fcmaes_ext import optimize_de
from fcmaes.de import _check_bounds

from typing import Callable, Optional, Tuple, Union
from numpy.typing import ArrayLike

Objective = Callable[[ArrayLike], float]
TerminateFn = Callable[[ArrayLike, ArrayLike], bool]
NativeResult = Tuple[np.ndarray, float, int, int, int]
SigmaArg = Optional[Union[float, ArrayLike, Callable]]


def _as_optional_vector(values: Optional[ArrayLike]) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_optional_bool_vector(values: Optional[ArrayLike]) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.bool_)
    return np.ascontiguousarray(values, dtype=np.bool_)


def _prepare_guess_and_sigma(dim: int,
                             x0: Optional[ArrayLike],
                             input_sigma: SigmaArg
                             ) -> Tuple[np.ndarray, np.ndarray]:
    sigma = input_sigma
    if sigma is not None:
        if callable(sigma):
            sigma = sigma()
        if np.ndim(sigma) == 0:
            sigma = [sigma] * dim
    if x0 is None or sigma is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return (
        np.ascontiguousarray(x0, dtype=np.float64),
        np.ascontiguousarray(sigma, dtype=np.float64),
    )


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
             dim: Optional[int] = None,
             bounds: Optional[Bounds] = None,
             popsize: Optional[int] = 31,
             max_evaluations: Optional[int] = 100000,
             stop_fitness: Optional[float] = -np.inf,
             keep: Optional[int] = 200,
             f: Optional[float] = 0.5,
             cr: Optional[float] = 0.9,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,
             workers: Optional[int] = 1,
             is_terminate: Optional[TerminateFn] = None,
             x0: Optional[ArrayLike] = None,
             input_sigma: SigmaArg = None,
             min_sigma: Optional[float] = 0,
             runid: Optional[int] = 0) -> OptimizeResult:
    """Minimize a scalar objective with the native DE implementation."""

    dim, lower, upper = _check_bounds(bounds, dim)
    popsize = 31 if popsize is None else popsize
    max_evaluations = 100000 if max_evaluations is None else max_evaluations
    stop_fitness = -np.inf if stop_fitness is None else stop_fitness
    keep = 200 if keep is None else keep
    f = 0.5 if f is None else f
    cr = 0.9 if cr is None else cr
    min_mutate = 0.1 if min_mutate is None else min_mutate
    max_mutate = 0.5 if max_mutate is None else max_mutate
    workers = 0 if workers is None else workers
    min_sigma = 0 if min_sigma is None else min_sigma
    runid = 0 if runid is None else runid
    rg = Generator(PCG64DXSM()) if rg is None else rg

    try:
        lower = _as_optional_vector(lower)
        upper = _as_optional_vector(upper)
        guess, sigma = _prepare_guess_and_sigma(dim, x0, input_sigma)
        ints_array = _as_optional_bool_vector(ints)
        result = optimize_de(
            fun,
            dim,
            lower,
            upper,
            guess,
            sigma,
            ints_array,
            seed=int(rg.uniform(0, 2**32 - 1)),
            runid=runid,
            max_evaluations=max_evaluations,
            keep=keep,
            stop_fitness=stop_fitness,
            popsize=popsize,
            F=f,
            CR=cr,
            min_sigma=min_sigma,
            min_mutate=min_mutate,
            max_mutate=max_mutate,
            workers=workers,
            terminate=is_terminate,
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


class DE_C:

    def __init__(self,
                 dim: Optional[int] = None,
                 bounds: Optional[Bounds] = None,
                 popsize: Optional[int] = 31,
                 keep: Optional[int] = 200,
                 f: Optional[float] = 0.5,
                 cr: Optional[float] = 0.9,
                 rg: Optional[Generator] = Generator(PCG64DXSM()),
                 ints: Optional[ArrayLike] = None,
                 min_mutate: Optional[float] = 0.1,
                 max_mutate: Optional[float] = 0.5,
                 x0: Optional[ArrayLike] = None,
                 input_sigma: SigmaArg = 0.3,
                 min_sigma: Optional[float] = 0) -> None:
        dim, lower, upper = _check_bounds(bounds, dim)
        popsize = 31 if popsize is None else popsize
        keep = 200 if keep is None else keep
        f = 0.5 if f is None else f
        cr = 0.9 if cr is None else cr
        min_mutate = 0.1 if min_mutate is None else min_mutate
        max_mutate = 0.5 if max_mutate is None else max_mutate
        min_sigma = 0 if min_sigma is None else min_sigma
        rg = Generator(PCG64DXSM()) if rg is None else rg

        guess, sigma = _prepare_guess_and_sigma(dim, x0, input_sigma)
        self._native = _DE(
            dim,
            _as_optional_vector(lower),
            _as_optional_vector(upper),
            guess,
            sigma,
            _as_optional_bool_vector(ints),
            popsize=popsize,
            keep=keep,
            F=f,
            CR=cr,
            min_sigma=min_sigma,
            min_mutate=min_mutate,
            max_mutate=max_mutate,
            seed=int(rg.uniform(0, 2**32 - 1)),
        )
        self.popsize = self._native.popsize
        self.dim = self._native.dim

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
