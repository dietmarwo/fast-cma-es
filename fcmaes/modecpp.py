# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""Eigen based implementation of multi objective
    Differential Evolution using the DE/pareto/1 strategy. 
    Derived and adapted for MO from its C++ counterpart 
    https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/deoptimizer.cpp
    
    Can switch to NSGA-II like population update via parameter 'nsga_update'.
    Then it works essentially like NSGA-II but instead of the tournament selection
    the whole population is sorted and the best individuals survive. To do this
    efficiently the crowd distance ordering is slightly inaccurate. 
    
    Supports parallel fitness function evaluation. 
    
    Features enhanced multiple constraint ranking (https://www.jstage.jst.go.jp/article/tjpnsec/11/2/11_18/_article/-char/en/)
    improving its performance in handling constraints for engineering design optimization.
    
    Enables the comparison of DE and NSGA-II population update mechanism with everything else
    kept completely identical.
       
    Requires python 3.5 or higher. 
    
    Uses the following deviation from the standard DE algorithm:
    a) oscillating CR/F parameters. 
    
    You may keep parameters F and CR at their defaults since this implementation works well with the given settings for most problems,
    since the algorithm oscillates between different F and CR settings. 
    
    For expensive objective functions (e.g. machine learning parameter optimization) use the workers
    parameter to parallelize objective function evaluation. This causes delayed population update.
    It is usually preferrable if popsize > workers and workers = mp.cpu_count() to improve CPU utilization.  
    
    The ints parameter is a boolean array indicating which parameters are discrete integer values. This 
    parameter was introduced after observing non optimal DE-results for the ESP2 benchmark problem: 
    https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py
    If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
    there is an additional mutation to avoid getting stuck to local minima. 
    
    See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/MODE.adoc for a detailed description.
"""

import os
import time
import threadpoolctl
import ctypes as ct
import multiprocessing as mp 
from multiprocessing import Process
import numpy as np
from scipy.optimize import Bounds
from fcmaes import mode, moretry
from fcmaes.mode import _filter, store
from numpy.random import Generator, MT19937, SeedSequence
from fcmaes.optimizer import dtime
from fcmaes.evaluator import mo_call_back_type, callback_mo, parallel_mo, libcmalib
from fcmaes.de import _check_bounds

import logging
from typing import Optional, Callable, Tuple
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(mofun: Callable[[ArrayLike], ArrayLike],
             nobj: int,
             ncon: int,
             bounds: Bounds,
             popsize: Optional[int] = 64,
             max_evaluations: Optional[int] = 100000,
             workers: Optional[int] = 1,
             f: Optional[float] = 0.5,
             cr: Optional[float] = 0.9,
             pro_c: Optional[float] = 1.0,
             dis_c: Optional[float] = 20.0,
             pro_m: Optional[float] = 1.0,
             dis_m: Optional[float] = 20.0,
             nsga_update: Optional[bool] = True,
             pareto_update: Optional[int] = 0,
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,           
             log_period: Optional[int] = 10000000,
             rg: Optional[Generator] = Generator(MT19937()),
             plot_name: Optional[str] = None,
             store: Optional[store] = None,
             is_terminate: Optional[Callable[[ArrayLike, ArrayLike], bool]] = None,
             runid: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray]:
     
    """Minimization of a multi objjective function of one or more variables using
    Differential Evolution.
     
    Parameters
    ----------
    mofun : callable
        The objective function to be minimized.
            ``mofun(x) -> ndarray(float)``
        where ``x`` is an 1-D array with shape (n,)
    nobj : int
        number of objectives
    ncon : int
        number of constraints, default is 0. 
        The objective function needs to return vectors of size nobj + ncon
    bounds : sequence or `Bounds`
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    popsize : int, optional
        Population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions     
    f = float, optional
        The mutation constant. In the literature this is also known as differential weight, 
        being denoted by F. Should be in the range [0, 2], usually leave at default.  
    cr = float, optional
        The recombination constant. Should be in the range [0, 1]. 
        In the literature this is also known as the crossover probability, usually leave at default.     
    pro_c, dis_c, pro_m, dis_m = float, optional
        NSGA population update parameters, usually leave at default. 
    nsga_update = boolean, optional
        Use of NSGA-II/SBX or DE population update. Default is True    
    pareto_update = float, optional
        Only applied if nsga_update = False. Favor better solutions for sample generation. Default 0 - 
        use all population members with the same probability.   
    ints = list or array of bool, optional
        indicating which parameters are discrete integer values. If defined these parameters will be
        rounded to the next integer and some additional mutation of discrete parameters are performed.  
    min_mutate = float, optional
        Determines the minimal mutation rate for discrete integer parameters.
    max_mutate = float, optional
        Determines the maximal mutation rate for discrete integer parameters.   
    log_period = int, optional
        The log callback is called each log_period iterations. As default the callback is never called.
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    store : result store, optional
        if defined the optimization results are added to the result store. For multi threaded execution.
        use workers=1 if you call minimize from multiple threads
    is_terminate : callable, optional
        Callback to be used if the caller of minimize wants to decide when to terminate.
    runid : int, optional
        id used to identify the run for debugging / logging. 

    Returns
    -------
    x, y: list of argument vectors and corresponding value vectors of the optimization results. """
    
    dim, lower, upper = _check_bounds(bounds, None)
    if popsize is None:
        popsize = 64
    if popsize % 2 == 1 and nsga_update: # nsga update requires even popsize
        popsize += 1
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim  
    if ints is None or nsga_update: # nsga update doesn't support mixed integer
        ints = [False]*dim
    if workers is None:
        workers = 0        
    array_type = ct.c_double * dim   
    bool_array_type = ct.c_bool * dim 
    c_callback = mo_call_back_type(callback_mo(mofun, dim, nobj + ncon, is_terminate))
    c_log = mo_call_back_type(log_mo(plot_name, dim, nobj, ncon))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(2*dim*popsize) # stores the resulting pareto front parameters
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeMODE_C(runid, c_callback, c_log, dim, nobj, ncon, seed,
                           array_type(*lower), array_type(*upper), bool_array_type(*ints), 
                           max_evaluations, popsize, workers, f, cr, 
                           pro_c, dis_c, pro_m, dis_m,
                           nsga_update, pareto_update, min_mutate, max_mutate, 
                           log_period, res_p)
        x = np.empty((2*popsize,dim))
        for p in range(2*popsize):
            x[p] = res[p*dim : (p+1)*dim]
        y = np.array([mofun(xi) for xi in x])
        x, y = _filter(x, y)
        if not store is None:
            store.add_results(x, y)
        return x, y
    except Exception as ex:
        return None, None
  
def retry(mofun: Callable[[ArrayLike], ArrayLike], 
            nobj: int,
            ncon: int, 
            bounds: Bounds,
            num_retries: Optional[int] = 64,
            popsize: Optional[int] = 64, 
            max_evaluations: Optional[int] = 100000, 
            workers: Optional[int] = mp.cpu_count(),
            nsga_update: Optional[bool] = False,
            pareto_update: Optional[int] = 0,
            ints: Optional[ArrayLike] = None,
            logger: Optional[logging.Logger] = None,
            is_terminate: Optional[Callable[[ArrayLike, ArrayLike], bool]] = None):
             
    """Minimization of a multi objjective function of one or more variables using parallel 
     optimization retry.
     
    Parameters
    ----------
        mofun : callable
        The objective function to be minimized.
            ``mofun(x, *args) -> ndarray(float)``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    nobj : int
        number of objectives
    ncon : int
        number of constraints, default is 0. 
        The objective function needs to return vectors of size nobj + ncon
    bounds : sequence or `Bounds`
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    num_retries : int, optional
        Number of optimization retries. 
    popsize : int, optional
        Population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    workers : int or None, optional
        If not workers is None, optimization is performed in parallel.  
    nsga_update = boolean, optional
        Use of NSGA-II/SBX or DE population update. Default is False    
    logger : logger, optional
        logger for log output for tell_one, If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``.
    is_terminate : callable, optional
        Callback to be used if the caller of minimize wants to decide when to terminate. """
    
    dim, _, _ = _check_bounds(bounds, None)
    store = mode.store(dim, nobj + ncon, 100*popsize*2)
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
           args=(num_retries, pid, rgs, mofun, nobj, ncon, bounds, popsize, 
                 max_evaluations, workers, nsga_update, pareto_update, 
                 is_terminate, store, logger, ints))
                for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    xs, ys = store.get_front()   
    if not logger is None:
        logger.info(str([tuple(y) for y in ys]))            
    return xs, ys

def _retry_loop(num_retries, pid, rgs, mofun, nobj, ncon, bounds, popsize, 
                max_evaluations, workers, nsga_update, pareto_update, 
                is_terminate, store, logger, ints):
    t0 = time.perf_counter()
    num = max(1, num_retries - workers)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        while store.num_added.value < num: 
            if not is_terminate is None and hasattr(is_terminate, 'reinit'):
                is_terminate.reinit()
            minimize(mofun, nobj, ncon, bounds, popsize,
                        max_evaluations = max_evaluations, 
                        nsga_update=nsga_update, pareto_update=pareto_update,
                        workers = 1, rg = rgs[pid], store = store, is_terminate=is_terminate, ints=ints) 
            if not logger is None:
                logger.info("retries = {0}: time = {1:.1f} i = {2}"
                            .format(store.num_added.value, dtime(t0), store.num_stored.value))

class log_mo(object):
    
    def __init__(self, name, dim, nobj, ncon):
        self.name = name
        self.dim = dim
        self.nobj = nobj + ncon
        self.ncon = ncon
        self.calls = 0
    
    def __call__(self, n, x, y):
        try:
            if not self.name is None:
                self.calls += 1
                arrTypeX = ct.c_double*(self.dim*n)
                arrTypeY = ct.c_double*(self.nobj*n)
                xaddr = ct.addressof(x.contents)
                yaddr = ct.addressof(y.contents)            
                xbuf = np.frombuffer(arrTypeX.from_address(xaddr))
                ybuf = np.frombuffer(arrTypeY.from_address(yaddr))
                xs = []; ys = []
                for p in range(n):
                    x = xbuf[p*self.dim : (p+1)*self.dim]
                    xs.append(x)
                    y = ybuf[p*self.nobj : (p+1)*self.nobj]
                    ys.append(y)
                xs = np.array(xs)
                ys = np.array(ys)
                print("callback", np.min(ys[:,0]), np.min(ys[:,1]))
                name = self.name + '_' + str(self.calls)
                np.savez_compressed(name, xs=xs, ys=ys)
                moretry.plot(name, self.ncon, xs, ys)
            return False # don't terminate optimization
        except Exception as ex:
            print (ex)
            return False

optimizeMODE_C = libcmalib.optimizeMODE_C
optimizeMODE_C.argtypes = [ct.c_long, mo_call_back_type, mo_call_back_type, ct.c_int, ct.c_int, \
            ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_bool), \
            ct.c_int, ct.c_int, ct.c_int,\
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
            ct.c_bool, ct.c_double, ct.c_double, ct.c_double, 
            ct.c_int, ct.POINTER(ct.c_double)]

class MODE_C:

    def __init__(self,
             nobj: int,
             ncon: int, 
             bounds: Bounds,
             popsize: Optional[int] = 64, 
             max_evaluations: Optional[int] = 100000, 
             f: Optional[float] = 0.5, 
             cr: Optional[float] = 0.9, 
             pro_c: Optional[float] = 1.0,
             dis_c: Optional[float] = 20.0,
             pro_m: Optional[float] = 1.0,
             dis_m: Optional[float] = 20.0,
             nsga_update: Optional[bool] = True,
             pareto_update: Optional[int] = 0,
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,           
             rg: Optional[Generator] = Generator(MT19937()),
             runid: Optional[int] = 0):  
       
        """    Parameters
        ----------
        nobj : int
            number of objectives
        ncon : int
            number of constraints, default is 0. 
            The objective function needs to return vectors of size nobj + ncon
        bounds : sequence or `Bounds`
            Bounds on variables. There are two ways to specify the bounds:
                1. Instance of the `scipy.Bounds` class.
                2. Sequence of ``(min, max)`` pairs for each element in `x`. None
                   is used to specify no bound.
        popsize : int, optional
            Population size.
        max_evaluations : int, optional
            Forced termination after ``max_evaluations`` function evaluations.
        f = float, optional
            The mutation constant. In the literature this is also known as differential weight, 
            being denoted by F. Should be in the range [0, 2], usually leave at default.  
        cr = float, optional
            The recombination constant. Should be in the range [0, 1]. 
            In the literature this is also known as the crossover probability, usually leave at default.     
        pro_c, dis_c, pro_m, dis_m = float, optional
            NSGA population update parameters, usually leave at default. 
        nsga_update = boolean, optional
            Use of NSGA-II or DE population update. Default is True    
        pareto_update = float, optional
            Only applied if nsga_update = False. Favor better solutions for sample generation. Default 0 - 
            use all population members with the same probability.   
        ints = list or array of bool, optional
            indicating which parameters are discrete integer values. If defined these parameters will be
            rounded to the next integer and some additional mutation of discrete parameters are performed.  
        min_mutate = float, optional
            Determines the minimal mutation rate for discrete integer parameters.
        max_mutate = float, optional
            Determines the maximal mutation rate for discrete integer parameters.   
        rg = numpy.random.Generator, optional
            Random generator for creating random guesses.
        runid : int, optional
            id used to identify the run for debugging / logging."""

        dim, lower, upper = _check_bounds(bounds, None)
        if popsize is None:
            popsize = 64
        if popsize % 2 == 1 and nsga_update: # nsga update requires even popsize
            popsize += 1
        if lower is None:
            lower = [0]*dim
            upper = [0]*dim  
        if ints is None or nsga_update: # nsga update doesn't support mixed integer
            ints = [False]*dim   
        array_type = ct.c_double * dim   
        bool_array_type = ct.c_bool * dim 
        seed = int(rg.uniform(0, 2**32 - 1))
        try:
            self.ptr = initMODE_C(runid, dim, nobj, ncon, seed,
                               array_type(*lower), array_type(*upper), bool_array_type(*ints), 
                               max_evaluations, popsize, f, cr, 
                               pro_c, dis_c, pro_m, dis_m,
                               nsga_update, pareto_update, min_mutate, max_mutate)
            self.popsize = popsize
            self.dim = dim    
            self.nobj = nobj  
            self.ncon = ncon
            self.bounds = bounds        
        except Exception as ex:
            print (ex)
            pass
     
    def __del__(self):
        destroyMODE_C(self.ptr)
            
    def ask(self) -> np.ndarray:
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askMODE_C(self.ptr, res_p)
            xs = np.empty((popsize, n))
            for p in range(popsize):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (ex)
            return None

    def tell(self, ys: np.ndarray) -> int:
        try:
            flat_ys = ys.flatten()
            array_type_ys = ct.c_double * len(flat_ys)
            return tellMODE_C(self.ptr, array_type_ys(*flat_ys))
        except Exception as ex:
            print (ex)
            return -1       
    
    def tell_switch(self, ys: np.ndarray, 
                        nsga_update: Optional[bool] = True,
                        pareto_update: Optional[int] = 0) -> int:
        try:
            flat_ys = ys.flatten()
            array_type_ys = ct.c_double * len(flat_ys)
            return tellMODE_switchC(self.ptr, array_type_ys(*flat_ys), nsga_update, pareto_update)
        except Exception as ex:
            print (ex)
            return -1        
 
    def population(self) -> np.ndarray:
        try:
            lamb = self.popsize
            n = self.dim
            res = np.empty(lamb*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationMODE_C(self.ptr, res_p)
            xs = np.array(lamb, n)
            for p in range(self.popsize):
                xs[p] = res[p*n : (p+1)*n]
                return xs
        except Exception as ex:
            print (ex)
            return None
        
    def minimize_par(self, 
                     fun: Callable[[ArrayLike], ArrayLike], 
                     max_evaluations: Optional[int] = 100000, 
                     workers: Optional[int] = mp.cpu_count()) -> Tuple[np.ndarray, np.ndarray]:
        fit = parallel_mo(fun, self.nobj, workers)
        evals = 0
        stop = 0
        while stop == 0 and evals < max_evaluations:
            xs = es.ask()
            ys = fit(xs)
            stop = es.tell(ys)
            evals += self.popsize
        fit.stop()
        return xs, ys

initMODE_C = libcmalib.initMODE_C
initMODE_C.argtypes = [ct.c_long, ct.c_int, ct.c_int, \
            ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_bool), \
            ct.c_int, ct.c_int,\
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
            ct.c_bool, ct.c_double, ct.c_double, ct.c_double]

initMODE_C.restype = ct.c_void_p   

destroyMODE_C = libcmalib.destroyMODE_C
destroyMODE_C.argtypes = [ct.c_void_p]

askMODE_C = libcmalib.askMODE_C
askMODE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]

tellMODE_C = libcmalib.tellMODE_C
tellMODE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
tellMODE_C.restype = ct.c_int

tellMODE_switchC = libcmalib.tellMODE_switchC
tellMODE_switchC.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.c_bool, ct.c_double]
tellMODE_switchC.restype = ct.c_int

populationMODE_C = libcmalib.populationMODE_C
populationMODE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]


