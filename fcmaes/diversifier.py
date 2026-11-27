# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
from __future__ import annotations

""" Numpy based implementation of an diversifying wrapper / parallel retry mechanism.

Uses the archive from CVT MAP-Elites (https://arxiv.org/abs/1610.05729)
and generalizes ideas from CMA-ME (https://arxiv.org/pdf/1912.02400.pdf)
to other wrapped algorithms.

Both the parallel retry and the archive based modification of the fitness
function enhance the diversification of the optimization result.
The resulting archive may be stored and can be used to continue the
optimization later.

Requires a QD-fitness function returning both an fitness value and a
behavior vector used to determine the corresponding archive niche using
Voronoi tesselation.

Returns an archive of niche-elites containing also for each niche statistics
about the associated solutions.
"""

import numpy as np
np.set_printoptions(legacy='1.25')
from numpy.random import Generator, PCG64DXSM, SeedSequence
from scipy.optimize import Bounds
from fcmaes.optimizer import dtime, de_cma, Optimizer
import multiprocessing as mp
import ctypes as ct
from time import perf_counter
from fcmaes.mapelites import Archive, update_archive, rng, run_workers
from fcmaes import advretry
from fcmaes.evaluator import is_debug_active
from loguru import logger
import threadpoolctl

from typing import Any, Optional, Callable, Tuple, Dict
from numpy.typing import ArrayLike

QDFitness = Callable[[ArrayLike], Tuple[float, np.ndarray]]
OptParams = Dict[str, Any]

def minimize(qd_fitness: QDFitness,
            bounds: Bounds,
            qd_bounds: Bounds,
            niche_num: Optional[int] = 10000,
            samples_per_niche: Optional[int] = 20,
            max_evals: Optional[int] = None,
            workers: Optional[int] = mp.cpu_count(),
            archive: Optional[Archive] = None,
            opt_params: Optional[OptParams] = {},
            use_stats: Optional[bool] = False,
            seed: Optional[int] = None,
            inline: bool = False,
            ) -> Archive:

    """Wraps an fcmaes optmizer/solver by hijacking its tell function.
    Works as CVT Map-Elites in maintaining an archive of diverse elites.
    But this archive is not used to derive solution vectors, but to reevaluate them.
    For each fitness result it determines its niche. The "told" fitness is
    determined relative to its local elite. If it is better the evaluated solution
    becomes the new niche-elite.
    This way the wrapped solver is "tricked" to follow a QD-goal: Finding empty niches
    and improving all niches. This works not only for CMA-ES, but also for other
    solvers: DE, CR-FM-NES and PGPE. Both their Python and C++ versions are supported.

    Parameters
    ----------
    solver : evolutionary algorithm, needs to support ask/tell
    qd_fitness : callable
        The objective function to be minimized. Returns a fitness value and a behavior vector.
            ``qd_fitness(x) -> float, array``
        where ``x`` is an 1-D array with shape (n,)
    bounds : `Bounds`
        Bounds on variables. Instance of the `scipy.Bounds` class.
    qd_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.
    niche_num : int, optional
        Number of niches.
    samples_per_niche : int, optional
        Number of samples used for niche computation.
        If samples_per_niche > 0 cvt-clustering is used, else grid-clustering is used.
    max_evals : int, optional
        Number of fitness evaluations.
    workers : int, optional
        Number of spawned parallel worker processes.
    archive : Archive, optional
        If defined MAP-elites is continued for this archive.
    opt_params : dictionary, optional (or a list/tuple/array of these)
        Parameters selecting and configuring the wrapped solver.
        'solver' - supported are 'CMA','CMA_CPP','CRMFNES','CRMFNES_CPP','DE','DE_CPP','PGPE'
                    default is 'CMA_CPP'
        'popsize' - population size, default = 32
        'sigma' -  initial distribution sigma, default = rg.uniform(0.03, 0.3)**2)
        'mean' - initial distribution mean, default=rg.uniform(bounds.lb, bounds.ub))
        'max_evals' - maximal number of evaluations per run, default = 50000
        'stall_criterion' - how many iterations without progress allowed.
        'batch_size' - proposal width for BITE_CPP, default = 1.
        'batch_fitness' - optional matrix callback for native solver emitters.
            It takes candidate rows and returns (scores, descriptors).
        If a list/tuple/array of parameters are given, the corresponding solvers are called in a
        sequence.
    use_stats : bool, optional
        If True, archive accumulates statistics of the solutions
    seed : int, optional
        Root seed for initialization, CVT centers and workers. Seeded
        single-worker runs replay exactly for deterministic objectives.
    inline : bool, optional
        Run a single worker in the caller's process.

    Returns
    -------
    archive : Archive
        Resulting archive of niches. Can be stored for later continuation of MAP-elites."""

    if max_evals is None:
        max_evals = workers*50000
    dim = len(bounds.lb)
    initial_seed, center_seed, worker_seed = SeedSequence(seed).spawn(3)
    if archive is None:
        archive = Archive(dim, qd_bounds, niche_num, use_stats=use_stats)
        archive.init_niches(samples_per_niche, seed=(
            None if seed is None else int(center_seed.generate_state(1)[0])
        ))
        initial_rng = rng if seed is None else Generator(PCG64DXSM(initial_seed))
        archive.xs_view[:] = initial_rng.uniform(bounds.lb, bounds.ub, (niche_num, dim))
    t0 = perf_counter()
    qd_fitness.archive = archive # attach archive for logging
    minimize_parallel_(archive, qd_fitness, bounds, workers, opt_params, max_evals, worker_seed, inline)
    if is_debug_active():
        ys = np.sort(archive.get_ys())[:100]
        ys = ys[np.isfinite(ys)]
        if ys.size:
            logger.debug(f'best {ys[0]:.3f} worst {ys[-1]:.3f} '
                         f'mean {ys.mean():.3f} stdev {ys.std():.3f} time {dtime(t0)} s')
    return archive

def apply_advretry(fitness: Callable[[ArrayLike], float],
                   qd_fitness: QDFitness,
                   bounds: Bounds,
                   archive: Archive,
                   optimizer: Optional[Optimizer] = None,
                   num_retries: Optional[int] = 1000,
                   workers: Optional[int] = mp.cpu_count(),
                   max_eval_fac: Optional[float] = 5.0,
                   xs: Optional[np.ndarray] = None,
                   ys: Optional[np.ndarray] = None,
                   x_conv: Optional[Callable[[ArrayLike], ArrayLike]] = None) -> None:

    """Unifies the QD world with traditional optimization. It converts
    a QD-archive into a multiprocessing store used by the fcmaes smart
    boundary management meta algorithm (advretry). Then advretry is applied
    to find the global optimum. Finally the updated store is feed back into
    the QD-archive. For this we need a descriptor generating function
    'descriptors' which may require reevaluation of the new solutions.

    Parameters
    ----------
    solver : evolutionary algorithm, needs to support ask/tell
    fitness : callable
        The objective function to be minimized. Returns a fitness value.
            ``fitness(x) -> float``
    qf_fun : callable
        Generates the descriptors for a solution. Returns a behavior vector.
            ``descriptors(x) -> array``
        where ``x`` is an 1-D array with shape (n,)
    bounds : `Bounds`
        Bounds on variables. Instance of the `scipy.Bounds` class.
    archive : Archive
        Improves the solutions if this archive.
    optimizer : optimizer.Optimizer, optional
        Optimizer to use. Default is a sequence of differential evolution and CMA-ES.
    num_retries : int, optional
        Number of optimization runs.
    workers : int, optional
        Number of spawned parallel worker processes.
    max_eval_fac : int, optional
        Final limit of the number of function evaluations = max_eval_fac*min_evaluations
    xs : ndarray, optional
        Used to initialize advretry. If undefined the archive content is used.
        If xs is defined, ys must be too
    ys : ndarray, optional
        Used to initialize advretry. If undefined the archive content is used.
    x_conv : callable, optional
        If defined converts the x in xs to solutions suitable for the given archive.
        If undefined it is assumed that the x in xs are valid archive solutons.
    """

    if optimizer is None:
        optimizer = de_cma(1500)
    # generate advretry store
    store = advretry.Store(fitness, bounds, num_retries=num_retries,
                           max_eval_fac=max_eval_fac)

    # select only occupied entries
    if xs is None:
        ys = archive.get_ys()
        valid = (ys < np.inf)
        ys = ys[valid]
        xs = archive.xs_view[valid]
    t0 = perf_counter()
    # transfer to advretry store
    for i in range(len(ys)):
        store.add_result(ys[i], xs[i], 1)
    # perform parallel retry
    advretry.retry(store, optimizer.minimize, workers=workers)
    # transfer back to archive
    xs = store.xs_view
    if not x_conv is None:
        xs = [x_conv(x) for x in xs]
    yds = [qd_fitness(x) for x in xs]
    descs = np.array([yd[1] for yd in yds])
    ys = np.array([yd[0] for yd in yds])
    niches = archive.index_of_niches(descs)
    for i in range(len(ys)):
        archive.set(niches[i], (ys[i], descs[i]), xs[i])
    archive.argsort()
    if is_debug_active():
        ys = np.sort(archive.get_ys())[:min(100, archive.capacity)] # best fitness values
        logger.debug(f'best {min(ys):.3f} worst {max(ys):.3f} ' +
                 f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')

def reserve_evaluations(evals, requested, limit):
    # The worker loops also accept an unsynchronized local counter.
    if not hasattr(evals, "get_lock"):
        count = min(requested, limit - evals.value)
        evals.value += count
        return count
    with evals.get_lock():
        count = min(requested, limit - evals.value)
        evals.value += count
        return count


def minimize_parallel_(archive: Archive,
                       fitness: QDFitness,
                       bounds: Bounds,
                       workers: int,
                       opt_params: Any,
                       max_evals: int,
                       seed: Optional[SeedSequence] = None,
                       inline: bool = False) -> None:
    seed = SeedSequence() if seed is None else seed
    streams = [Generator(PCG64DXSM(s)) for s in seed.spawn(workers)]
    evals = mp.Value(ct.c_long, 0)
    if inline:
        if workers != 1:
            raise ValueError("inline Diversifier expects workers=1")
        run_minimize_(archive, fitness, bounds, streams[0], opt_params, 0, 1, evals, max_evals)
        return
    run_workers(run_minimize_, (
        (archive, fitness, bounds, streams[p], opt_params, p, workers, evals, max_evals)
        for p in range(workers)
    ))

def run_minimize_(archive: Archive,
                  fitness: QDFitness,
                  bounds: Bounds,
                  rg: Generator,
                  opt_params: Any,
                  p: int,
                  workers: int,
                  evals: Any,
                  max_evals: int,
                  cancelled: Optional[Any] = None) -> None:
    archive.create_views()
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        if isinstance(opt_params, (list, tuple, np.ndarray)):
            default_workers = int(workers/2) if len(opt_params) > 1 else workers
            for params in opt_params: # call MAP-Elites
                if 'elites' == params.get('solver'):
                    elites_workers = params.get('workers', default_workers)
                    if p < elites_workers:
                        run_map_elites_(archive, fitness, bounds, rg, evals, max_evals,
                                        params, cancelled)
                        return
        if isinstance(opt_params, (list, tuple, np.ndarray)) and all(
                params.get('solver') == 'elites' for params in opt_params):
            return
        while evals.value < max_evals and (cancelled is None or not cancelled.is_set()):
            best_x = None
            if isinstance(opt_params, (list, tuple, np.ndarray)):
                for params in opt_params: # call in sequence
                    if evals.value >= max_evals or (cancelled is not None and cancelled.is_set()):
                        break
                    if 'elites' == params.get('solver'):
                        continue # ignore in loop
                    best_x = minimize_(archive, fitness, bounds, rg, evals, max_evals,
                                       params, x0=best_x, cancelled=cancelled)
            else:
                minimize_(archive, fitness, bounds, rg, evals, max_evals, opt_params,
                          cancelled=cancelled)

from fcmaes.mapelites import variation_,  iso_dd_

def run_map_elites_(archive: Archive,
                    fitness: QDFitness,
                    bounds: Bounds,
                    rg: Generator,
                    evals: Any,
                    max_evals: int,
                    opt_params: OptParams = {},
                    cancelled: Optional[Any] = None) -> None:
    popsize = opt_params.get('popsize', 32)
    use_sbx = opt_params.get('use_sbx', True)
    if popsize < (2 if use_sbx else 1):
        raise ValueError('invalid population size for MAP-Elites emitter')
    dis_c = opt_params.get('dis_c', 20)
    dis_m = opt_params.get('dis_m', 20)
    iso_sigma = opt_params.get('iso_sigma', 0.01)
    line_sigma = opt_params.get('line_sigma', 0.2)
    select_n = archive.capacity
    while evals.value < max_evals and (cancelled is None or not cancelled.is_set()):
        if use_sbx:
            pop = archive.random_xs(select_n, popsize, rg)
            xs = variation_(pop, bounds.lb, bounds.ub, rg, dis_c, dis_m)
        else:
            x1 = archive.random_xs(select_n, popsize, rg)
            x2 = archive.random_xs(select_n, popsize, rg)
            xs = iso_dd_(x1, x2, bounds.lb, bounds.ub, rg, iso_sigma, line_sigma)
        count = reserve_evaluations(evals, len(xs), max_evals)
        if count == 0:
            return
        xs = xs[:count]
        yds = [fitness(x) for x in xs]
        descs = np.array([yd[1] for yd in yds])
        niches = archive.index_of_niches(descs)
        for i in range(len(yds)):
            archive.set(niches[i], yds[i], xs[i])
        archive.argsort()
        select_n = archive.get_occupied() or archive.capacity

def minimize_(archive: Archive,
              fitness: QDFitness,
              bounds: Bounds,
              rg: Generator,
              evals: Any,
              max_evals: int,
              opt_params: OptParams,
              x0: Optional[np.ndarray] = None,
              cancelled: Optional[Any] = None) -> Optional[np.ndarray]:
    if opt_params.get('solver') == 'BITE_CPP':
        return run_bite_(archive, fitness, bounds, rg, evals, max_evals,
                         opt_params, x0, cancelled)

    es = get_solver_(bounds, opt_params, rg, x0)
    stall_criterion = opt_params.get('stall_criterion', 20)
    remaining = opt_params.get('max_evals', 50000)
    if remaining <= 0:
        raise ValueError('emitter max_evals must be positive')
    old_ys = None
    last_improve = 0
    best_x = None
    best_y = np.inf
    iteration = 0
    while remaining > 0 and evals.value < max_evals:
        if cancelled is not None and cancelled.is_set():
            break
        xs = es.ask()
        if xs is None or len(xs) == 0:
            raise RuntimeError('emitter returned no candidates')
        count = reserve_evaluations(evals, min(remaining, len(xs)), max_evals)
        if count == 0:
            break
        ys, real_ys = update_archive(archive, xs[:count], fitness,
                                     batch_fitness=opt_params.get('batch_fitness'))
        remaining -= count
        yi = np.argmin(real_ys)
        if real_ys[yi] < best_y:
            best_y = real_ys[yi]
            best_x = xs[yi]
        if count < len(xs):
            break  # Only a full generation can be told to this solver.
        sorted_ys = np.sort(ys)
        if old_ys is not None and (sorted_ys < old_ys).any():
            last_improve = iteration
        if last_improve + stall_criterion < iteration:
            break
        stop = es.tell(ys)
        if stop < 0:
            raise RuntimeError('emitter rejected its objective values')
        if stop != 0:
            break
        old_ys = sorted_ys
        iteration += 1
    return best_x

from fcmaes import cmaes, cmaescpp, crfmnescpp, pgpecpp, decpp, crfmnes, de, bitecpp

def run_bite_(archive: Archive,
              fitness: QDFitness,
              bounds: Bounds,
              rg: Generator,
              evals: Any,
              max_evals: int,
              opt_params: OptParams,
              x0: Optional[np.ndarray] = None,
              cancelled: Optional[Any] = None) -> Optional[np.ndarray]:
    budget = opt_params.get('max_evals', 50000)
    if budget <= 0:
        raise ValueError('emitter max_evals must be positive')
    # Keep single-candidate feedback as the default. Larger batches use the
    # archive snapshot from before their first update.
    batch_size = opt_params.get('batch_size', 1)
    if batch_size <= 0:
        raise ValueError('emitter batch_size must be positive')
    es = bitecpp.Bite_C(len(bounds.lb), bounds, x0=x0, M=1, batch_size=batch_size,
                        stall_criterion=opt_params.get('stall_criterion', 20),
                        max_evaluations=budget, rg=rg)
    best_x = None
    best_y = np.inf
    while evals.value < max_evals and (cancelled is None or not cancelled.is_set()):
        xs = es.ask()
        if xs is None:
            raise RuntimeError('Bite emitter failed to propose candidates')
        if not len(xs):
            break
        count = reserve_evaluations(evals, len(xs), max_evals)
        if count == 0:
            break
        scores, values = update_archive(archive, xs[:count], fitness,
                                        batch_fitness=opt_params.get('batch_fitness'))
        i = np.argmin(values)
        if values[i] < best_y:
            best_y = values[i]
            best_x = xs[i]
        if count != len(xs):
            break
        stop = es.tell(scores)
        if stop < 0:
            raise RuntimeError('Bite emitter rejected its objective values')
        if stop != 0:
            break
    return best_x


def get_solver_(bounds: Bounds,
                opt_params: OptParams,
                rg: Generator,
                x0: Optional[np.ndarray] = None) -> Any:
    dim = len(bounds.lb)
    popsize = opt_params.get('popsize', 31)
    #sigma = opt_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    sigma = opt_params.get('sigma',rg.uniform(0.1, 0.5)**2)
    #sigma = opt_params.get('sigma',rg.uniform(0.2, 0.5)**2)
    #sigma = opt_params.get('sigma',rg.uniform(0.1, 0.5))
    mean = opt_params.get('mean', rg.uniform(bounds.lb, bounds.ub)) \
                if x0 is None else x0
    name = opt_params.get('solver', 'CMA_CPP')
    if name == 'CMA':
        return cmaes.Cmaes(bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CMA_CPP':
        return cmaescpp.ACMA_C(dim, bounds, x0 = mean, #stop_hist = 0,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CRMFNES':
        return crfmnes.CRFMNES(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CRMFNES_CPP':
        return crfmnescpp.CRFMNES_C(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'DE':
        return de.DE(dim, bounds, popsize = popsize, rg = rg)
    elif name == 'DE_CPP':
        return decpp.DE_C(dim, bounds, popsize = popsize, rg = rg)
    elif name == 'PGPE':
        return pgpecpp.PGPE_C(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    else:
        print ("invalid solver")
        return None

