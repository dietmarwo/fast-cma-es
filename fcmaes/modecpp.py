# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

from __future__ import annotations

"""Native multi-objective DE exposed through nanobind."""

import multiprocessing as mp
import os
import time
from multiprocessing import Process
from typing import Callable, Optional, Tuple, Union

import numpy as np
import threadpoolctl
from loguru import logger
from numpy.random import Generator, PCG64DXSM, SeedSequence
from numpy.typing import ArrayLike
from scipy.optimize import Bounds

from fcmaes import mode
from fcmaes._fcmaes_ext import MODE as _MODE
from fcmaes.de import _check_bounds
from fcmaes.evaluator import is_debug_active, parallel_mo
from fcmaes.mode import store
from fcmaes.optimizer import dtime

MultiObjective = Callable[[ArrayLike], ArrayLike]
GuessArg = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"


def _as_float_vector(values: ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_float_matrix(values: ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_optional_bool_vector(values: Optional[ArrayLike]) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.bool_)
    return np.ascontiguousarray(values, dtype=np.bool_)


def minimize(mofun: MultiObjective,
             nobj: int,
             ncon: int,
             bounds: Bounds,
             guess: Optional[np.ndarray] = None,
             popsize: Optional[int] = 64,
             max_evaluations: Optional[int] = 100000,
             workers: Optional[int] = 1,
             f: Optional[float] = 0.5,
             cr: Optional[float] = 0.9,
             pro_c: Optional[float] = 0.5,
             dis_c: Optional[float] = 15.0,
             pro_m: Optional[float] = 0.9,
             dis_m: Optional[float] = 20.0,
             nsga_update: Optional[bool] = True,
             pareto_update: Optional[int] = 0,
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             store: Optional[store] = None,
             runid: Optional[int] = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Minimize a multi-objective function using the native MODE implementation."""

    try:
        mode_c = MODE_C(
            nobj, ncon, bounds, popsize, f, cr, pro_c, dis_c, pro_m, dis_m,
            nsga_update, pareto_update, ints, min_mutate, max_mutate, rg, runid
        )
        mode_c.set_guess(guess, mofun, rg)
        if workers is None or workers <= 1:
            xs, ys = mode_c.minimize_ser(mofun, max_evaluations)
        else:
            xs, ys = mode_c.minimize_par(mofun, max_evaluations, workers)
        if store is not None:
            store.create_views()
            store.add_results(xs, ys)
        return xs, ys
    except Exception as ex:
        print(str(ex))
        return None, None


def retry(mofun: MultiObjective,
          nobj: int,
          ncon: int,
          bounds: Bounds,
          guess: Optional[np.ndarray] = None,
          num_retries: Optional[int] = 64,
          popsize: Optional[int] = 64,
          max_evaluations: Optional[int] = 100000,
          workers: Optional[int] = mp.cpu_count(),
          nsga_update: Optional[bool] = False,
          pareto_update: Optional[int] = 0,
          ints: Optional[ArrayLike] = None,
          capacity: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Perform parallel retry runs of the native MODE implementation."""

    dim, _, _ = _check_bounds(bounds, None)
    if capacity is None:
        capacity = 2048 * popsize
    result_store = mode.store(dim, nobj + ncon, capacity)
    sg = SeedSequence()
    rgs = [Generator(PCG64DXSM(s)) for s in sg.spawn(workers)]
    proc = [
        Process(
            target=_retry_loop,
            args=(
                num_retries, pid, rgs, mofun, nobj, ncon, bounds, guess,
                popsize, max_evaluations, workers, nsga_update,
                pareto_update, result_store, ints
            ),
        )
        for pid in range(workers)
    ]
    for p in proc:
        p.start()
    for p in proc:
        p.join()
    xs, ys = result_store.get_front()
    return xs, ys


def _retry_loop(num_retries: int,
                pid: int,
                rgs: list[Generator],
                mofun: MultiObjective,
                nobj: int,
                ncon: int,
                bounds: Bounds,
                guess: GuessArg,
                popsize: int,
                max_evaluations: int,
                workers: int,
                nsga_update: bool,
                pareto_update: int,
                result_store: store,
                ints: Optional[ArrayLike]) -> None:
    result_store.create_views()
    t0 = time.perf_counter()
    num = max(1, num_retries - workers)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        while result_store.num_added.value < num:
            minimize(
                mofun, nobj, ncon, bounds, guess, popsize,
                max_evaluations=max_evaluations,
                nsga_update=nsga_update,
                pareto_update=pareto_update,
                rg=rgs[pid],
                store=result_store,
                ints=ints,
            )
            if is_debug_active():
                logger.debug(
                    "retries = {0}: time = {1:.1f} i = {2}".format(
                        result_store.num_added.value,
                        dtime(t0),
                        result_store.num_stored.value,
                    )
                )


class MODE_C:

    def __init__(self,
                 nobj: int,
                 ncon: int,
                 bounds: Bounds,
                 popsize: Optional[int] = 64,
                 f: Optional[float] = 0.5,
                 cr: Optional[float] = 0.9,
                 pro_c: Optional[float] = 0.5,
                 dis_c: Optional[float] = 15.0,
                 pro_m: Optional[float] = 0.9,
                 dis_m: Optional[float] = 20.0,
                 nsga_update: Optional[bool] = True,
                 pareto_update: Optional[int] = 0,
                 ints: Optional[ArrayLike] = None,
                 min_mutate: Optional[float] = 0.1,
                 max_mutate: Optional[float] = 0.5,
                 rg: Optional[Generator] = Generator(PCG64DXSM()),
                 runid: Optional[int] = 0) -> None:
        dim, lower, upper = _check_bounds(bounds, None)
        popsize = 64 if popsize is None else popsize
        if popsize % 2 == 1 and nsga_update:
            popsize += 1
        f = 0.5 if f is None else f
        cr = 0.9 if cr is None else cr
        pro_c = 0.5 if pro_c is None else pro_c
        dis_c = 15.0 if dis_c is None else dis_c
        pro_m = 0.9 if pro_m is None else pro_m
        dis_m = 20.0 if dis_m is None else dis_m
        nsga_update = True if nsga_update is None else nsga_update
        pareto_update = 0 if pareto_update is None else pareto_update
        min_mutate = 0.1 if min_mutate is None else min_mutate
        max_mutate = 0.5 if max_mutate is None else max_mutate
        rg = Generator(PCG64DXSM()) if rg is None else rg
        runid = 0 if runid is None else runid

        ints_array = (
            np.empty(0, dtype=np.bool_)
            if ints is None or nsga_update
            else _as_optional_bool_vector(ints)
        )

        self._native = _MODE(
            dim,
            nobj,
            ncon,
            _as_float_vector(lower),
            _as_float_vector(upper),
            ints_array,
            popsize=popsize,
            F=f,
            CR=cr,
            pro_c=pro_c,
            dis_c=dis_c,
            pro_m=pro_m,
            dis_m=dis_m,
            nsga_update=nsga_update,
            pareto_update=pareto_update,
            min_mutate=min_mutate,
            max_mutate=max_mutate,
            seed=int(rg.uniform(0, 2**32 - 1)),
            runid=runid,
        )
        self.popsize = self._native.popsize
        self.dim = self._native.dim
        self.nobj = self._native.nobj
        self.ncon = self._native.ncon
        self.bounds = bounds

    def set_guess(self,
                  guess: GuessArg,
                  mofun: MultiObjective,
                  rg: Optional[Generator] = None) -> None:
        if guess is None:
            return
        if isinstance(guess, np.ndarray):
            ys = np.ascontiguousarray([mofun(x) for x in guess], dtype=np.float64)
        else:
            guess, ys = guess
        if rg is None:
            rg = Generator(PCG64DXSM())
        choice = rg.choice(len(ys), self.popsize, replace=(len(ys) < self.popsize))
        self.tell(np.asarray(ys)[choice], np.asarray(guess)[choice])

    def ask(self) -> Optional[np.ndarray]:
        try:
            return self._native.ask()
        except Exception as ex:
            print(str(ex))
            return None

    def tell(self, ys: np.ndarray, xs: Optional[np.ndarray] = None) -> int:
        try:
            ys_array = _as_float_matrix(ys)
            if xs is None:
                return self._native.tell(ys_array)
            stop = self._native.set_population(_as_float_matrix(xs), ys_array)
            self.popsize = self._native.popsize
            return stop
        except Exception as ex:
            print(str(ex))
            return -1

    def tell_switch(self, ys: np.ndarray,
                    nsga_update: Optional[bool] = True,
                    pareto_update: Optional[int] = 0) -> int:
        try:
            return self._native.tell_switch(
                _as_float_matrix(ys),
                nsga_update=True if nsga_update is None else nsga_update,
                pareto_update=0 if pareto_update is None else pareto_update,
            )
        except Exception as ex:
            print(str(ex))
            return -1

    def population(self) -> Optional[np.ndarray]:
        try:
            return self._native.population()
        except Exception as ex:
            print(str(ex))
            return None

    def minimize_ser(self,
                     fun: MultiObjective,
                     max_evaluations: Optional[int] = 100000
                     ) -> Tuple[np.ndarray, np.ndarray]:
        max_evaluations = 100000 if max_evaluations is None else max_evaluations
        evals = 0
        stop = 0
        xs = np.empty((self.popsize, self.dim), dtype=np.float64)
        ys = np.empty((self.popsize, self.nobj + self.ncon), dtype=np.float64)
        while stop == 0 and evals < max_evaluations:
            xs = self.ask()
            ys = np.ascontiguousarray([fun(x) for x in xs], dtype=np.float64)
            stop = self.tell(ys)
            evals += self.popsize
        return xs, ys

    def minimize_par(self,
                     fun: MultiObjective,
                     max_evaluations: Optional[int] = 100000,
                     workers: Optional[int] = mp.cpu_count()
                     ) -> Tuple[np.ndarray, np.ndarray]:
        max_evaluations = 100000 if max_evaluations is None else max_evaluations
        workers = mp.cpu_count() if workers is None else workers
        fit = parallel_mo(fun, self.nobj + self.ncon, workers)
        evals = 0
        stop = 0
        xs = np.empty((self.popsize, self.dim), dtype=np.float64)
        ys = np.empty((self.popsize, self.nobj + self.ncon), dtype=np.float64)
        while stop == 0 and evals < max_evaluations:
            xs = self.ask()
            ys = fit(xs)
            stop = self.tell(ys)
            evals += self.popsize
        fit.stop()
        return xs, ys
