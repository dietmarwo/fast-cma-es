# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Implements a stochastic non-linear
    bound-constrained derivative-free optimization method.
    Description is available at https://github.com/avaneev/biteopt
"""

import sys
import os
import numpy as np
np.set_printoptions(legacy='1.25')
from numpy.random import PCG64DXSM, Generator
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import _check_bounds, _get_bounds
from fcmaes._fcmaes_ext import Bite as _Bite
from fcmaes._fcmaes_ext import optimize_bite

from typing import Callable, Optional, Tuple
from numpy.typing import ArrayLike

Objective = Callable[[ArrayLike], float]
NativeResult = Tuple[np.ndarray, float, int, int, int]


def _as_optional_vector(values: Optional[ArrayLike]) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    return np.ascontiguousarray(values, dtype=np.float64)


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
             max_evaluations: Optional[int] = 100000,
             stop_fitness: Optional[float] = -np.inf,
             M: Optional[int] = 1,
             popsize: Optional[int] = 0,
             stall_criterion: Optional[int] = 0,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             runid: Optional[int] = 0) -> OptimizeResult:
    """Minimization of a scalar function of one or more variables using a
    C++ BiteOpt implementation exposed through nanobind.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x) -> float``
        where ``x`` is an 1-D array with shape (dim,)
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    x0 : ndarray, shape (dim,)
        Initial guess. Array of real elements of size (dim,),
        where 'dim' is the number of independent variables.  
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    M : int, optional 
        Depth to use, 1 for plain CBiteOpt algorithm, >1 for CBiteOptDeep. Expected range is [1; 36].
    popsize = int, optional
        initial population size.
    stall_criterion : int, optional 
        Terminate if stall_criterion*128*evaluations stalled, Not used if <= 0
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used to identify the run for debugging / logging. 
           
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, 
        ``nfev`` the number of function evaluations,
        ``nit`` the number of CMA-ES iterations, 
        ``status`` the stopping critera and
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
    
    lower, upper, guess = _check_bounds(bounds, x0, rg)
    try:
        result = optimize_bite(
            fun,
            np.ascontiguousarray(guess, dtype=np.float64),
            _as_optional_vector(lower),
            _as_optional_vector(upper),
            seed=int(rg.uniform(0, 2**32 - 1)),
            runid=runid,
            max_evaluations=max_evaluations,
            stop_fitness=stop_fitness,
            M=M,
            popsize=popsize,
            stall_criterion=stall_criterion,
        )
        return _result_from_native(result)
    except Exception:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)


class Bite_C:
    def __init__(self,
                 dim: int,
                 bounds: Optional[Bounds] = None,
                 x0: Optional[ArrayLike] = None,
                 max_evaluations: Optional[int] = 100000,
                 stop_fitness: Optional[float] = -np.inf,
                 M: Optional[int] = 1,
                 popsize: Optional[int] = 0,
                 batch_size: Optional[int] = 8,
                 stall_criterion: Optional[int] = 0,
                 rg: Optional[Generator] = Generator(PCG64DXSM()),
                 runid: Optional[int] = 0) -> None:
        """Ask/tell BiteOpt state backed by the nanobind extension."""

        lower, upper, guess = _get_bounds(dim, bounds, x0, rg)
        max_evaluations = 100000 if max_evaluations is None else max_evaluations
        stop_fitness = -np.inf if stop_fitness is None else stop_fitness
        M = 1 if M is None else M
        popsize = 0 if popsize is None else popsize
        batch_size = 8 if batch_size is None else batch_size
        stall_criterion = 0 if stall_criterion is None else stall_criterion
        rg = Generator(PCG64DXSM()) if rg is None else rg
        runid = 0 if runid is None else runid

        self._native = _Bite(
            np.ascontiguousarray(guess, dtype=np.float64),
            _as_optional_vector(lower),
            _as_optional_vector(upper),
            M=M,
            popsize=popsize,
            batch_size=batch_size,
            max_evaluations=max_evaluations,
            stop_fitness=stop_fitness,
            stall_criterion=stall_criterion,
            seed=int(rg.uniform(0, 2**32 - 1)),
            runid=runid,
        )
        self.dim = self._native.dim
        self.popsize = self._native.popsize
        self.population_size = self._native.population_size

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
