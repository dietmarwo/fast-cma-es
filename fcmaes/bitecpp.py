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
from fcmaes.evaluator import _check_bounds
from fcmaes._fcmaes_ext import optimize_bite

from typing import Callable, Optional
from numpy.typing import ArrayLike

Objective = Callable[[ArrayLike], float]

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
        guess = np.ascontiguousarray(guess, dtype=np.float64)
        if lower is None:
            lower = np.empty(0, dtype=np.float64)
            upper = np.empty(0, dtype=np.float64)
        else:
            lower = np.ascontiguousarray(lower, dtype=np.float64)
            upper = np.ascontiguousarray(upper, dtype=np.float64)
        x, val, evals, iterations, stop = optimize_bite(
            fun,
            guess,
            lower,
            upper,
            seed=int(rg.uniform(0, 2**32 - 1)),
            runid=runid,
            max_evaluations=max_evaluations,
            stop_fitness=stop_fitness,
            M=M,
            popsize=popsize,
            stall_criterion=stall_criterion,
        )
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)
