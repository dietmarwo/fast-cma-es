# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
from __future__ import annotations

""" Numpy based implementation of CVT MAP-Elites including CMA-ES emitter and CMA-ES drilldown.

See https://arxiv.org/abs/1610.05729 and https://arxiv.org/pdf/1912.02400.pdf

MAP-Elites implementations differ in the following details:

1) Initialisation of the behavior space:

a) Generated from some solution distribution by applying the fitness function to determine their behavior.
b) Generated from uniform samples of the behavior space.

We use b) because random solutions may cover only parts of the behavior space. Some parts may only be reachable
by optimization. Another reason: Fitness computations may be expensive. Therefore we don't compute fitness
values for the initial solution population.

2) Initialization of the niches:

a) Generated from some solution distribution.
b) Generated from uniform samples of the solution space. These solutions are never evaluated but serve as
initial population for SBX or Iso+LineDD. Their associated fitness value is set to math.inf (infinity).

We use b) because this way we:
- Avoid computing fitness values for the initial population.
- Enhance the diversity of initial solutions emitted by SBX or Iso+LineDD.

3) Iso+LineDD (https://arxiv.org/pdf/1804.03906) is implemented but doesn't work well with extremely ragged solution
landscapes. Therefore SBX+mutation is the default setting.

4) SBX (Simulated binary crossover) is taken from mode.py and simplified. It is combined with mutation.
Both spread factors - for crossover and mutation - are randomized for each application.

5) Candidates for CMA-ES are sampled with a bias to better niches. As for SBX only a subset of the archive is
used, the worst niches are ignored.

6) There is a CMA-ES drill down for specific niches - in this mode all solutions outside the niche
are rejected. Restricted solution box bounds are used derived from statistics maintained by the archive
during the addition of new solution candidates.

7) The QD-archive uses shared memory to reduce inter-process communication overhead.
"""

import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
from multiprocessing.connection import wait
import multiprocessing as mp
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from scipy.optimize import Bounds
from pathlib import Path
from tempfile import NamedTemporaryFile
from fcmaes.retry import Shared2d
from fcmaes.optimizer import dtime
from fcmaes import cmaescpp
from numpy.random import default_rng
import ctypes as ct
from time import perf_counter
import threadpoolctl
from fcmaes.evaluator import is_debug_active
from loguru import logger

from typing import Any, Optional, Callable, Tuple, Dict
from numpy.typing import ArrayLike

from fcmaes._native import _fcmaes_ext

try:
    from numba import njit
    _has_numba = True
except ModuleNotFoundError:
    _has_numba = False
    # MAP-Elites still works without numba; only the JIT speedup is missing.
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

QDFitness = Callable[[ArrayLike], Tuple[float, np.ndarray]]
OptParams = Dict[str, Any]

rng = default_rng()

def optimize_map_elites(qd_fitness: QDFitness,
                        bounds: Bounds,
                        qd_bounds: Bounds,
                        niche_num: Optional[int] = 4000,
                        samples_per_niche: Optional[int] = 20,
                        workers: Optional[int] = mp.cpu_count(),
                        iterations: Optional[int] = 100,
                        archive: Optional[Archive] = None,
                        me_params: Optional[OptParams] = {},
                        cma_params: Optional[OptParams] = {},
                        use_stats: Optional[bool] = False,
                        seed: Optional[int] = None,
                        persistent_workers: bool = False,
                        ) -> Archive:

    """Application of CVT-Map Elites with additional CMA-ES emmitter.

    Parameters
    ----------
    qd_fitness : callable
        The objective function to be minimized.
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
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    workers : int, optional
        Number of spawned parallel worker processes.
    iterations : int, optional
        Number of MAP-elites iterations.
    archive : Archive, optional
        If defined MAP-elites is continued for this archive.
    me_params : dictionary, optional
        Parameters for MAP-elites.
    cma_params : dictionary, optional
        Parameters for the CMA-ES emitter.
    use_stats : bool, optional
        If True, archive accumulates statistics of the solutions
    seed : int, optional
        Root seed for initialization, CVT centers and worker streams. With
        multiple workers the order of archive updates depends on scheduling.
    persistent_workers : bool, optional
        Keep the same worker processes and callback state across iterations.

    Returns
    -------
    archive : Archive
        Resulting archive of niches. Can be stored for later continuation of MAP-elites."""

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
    if persistent_workers and iterations > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context(),
            initializer=_init_map_worker,
            initargs=(archive, qd_fitness, bounds, me_params, cma_params),
        ) as executor:
            for iteration in range(iterations):
                archive.argsort()
                list(executor.map(_run_map_worker, worker_seed.spawn(workers)))
                _log_map_iteration(archive, iteration, t0)
        return archive
    for iteration in range(iterations):
        archive.argsort()
        optimize_map_elites_(archive, qd_fitness, bounds, workers, me_params,
                             cma_params, worker_seed)
        _log_map_iteration(archive, iteration, t0)
    return archive

def _log_map_iteration(archive, iteration, started):
    if is_debug_active():
        ys = np.sort(archive.get_ys())[:100]
        ys = ys[np.isfinite(ys)]
        if ys.size:
            logger.debug(f'best 100 iter {iteration} best {ys[0]:.3f} worst {ys[-1]:.3f} '
                         f'mean {ys.mean():.3f} stdev {ys.std():.3f} time {dtime(started)} s')


def empty_archive(dim: int,
                  qd_bounds: Bounds,
                  niche_num: int,
                  samples_per_niche: int,
                  use_stats: Optional[bool] = False) -> Archive:

    """Creates an empty archive.

    Parameters
    ----------
    archive: Archive
    qd_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.
    niche_num : int, optional
        Number of niches.
    samples_per_niche : int, optional
        Number of samples used for niche computation.
    use_stats : bool, optional
        If True, archive accumulates statistics of the solutions

    Returns
    -------
    archive : Archive
        Empty archive of niches."""

    archive = Archive(dim, qd_bounds, niche_num, use_stats=use_stats)
    archive.init_niches(samples_per_niche)
    return archive

def set_KDTree(archive: Archive,
                        centers:Optional[np.ndarray] = None,
                        niche_num: Optional[int]  = None,
                        qd_bounds: Optional[Bounds] = None,
                        samples_per_niche: Optional[int] = 100,
                        seed: Optional[int] = None) -> None:

    """Returns a function deciding niche membership.

    Parameters
    ----------
    archive: Archive
    centers : ndarray, shape (n,m), optional
        If defined, these behavior vectors are used as niche centers
    niche_num : int, optional
        Number of niches. Required if centers is None.
    qd_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.
        Required if centers is None.
    samples_per_niche : int, optional
        Number of samples used for niche computation.
        If samples_per_niche > 0 cvt-clustering is used, else grid-clustering is used.

    Returns
    -------
    index_of_niches : callable
        Maps an array of description vectors to their corresponding niche indices.
    centers : ndarray, shape (n,m)
        behavior vectors used as niche centers."""

    if centers is None: # cache centers
        centers = get_centers_(niche_num, len(qd_bounds.lb), samples_per_niche, seed)
    archive.kdt = KDTree(centers, leaf_size=30, metric='euclidean')
    archive.centers = centers

def load_archive(name: str,
                 bounds: Bounds,
                 qd_bounds: Bounds,
                 niche_num: Optional[int] = 10000,
                 use_stats: Optional[bool] = False,
                 ) -> Archive:

    """Loads an archive from disk.

    Parameters
    ----------
    name: string
        Name of the archive.
    bounds : `Bounds`
        Bounds on variables. Instance of the `scipy.Bounds` class.
    qd_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.
    niche_num : int, optional
        Number of niches.
    use_stats : bool, optional
        If True, archive accumulates statistics of the solutions

    Returns
    -------
    archive : Archive
        Archive of niches. Can be used for continuation of MAP-elites."""

    dim = len(bounds.lb)
    archive = Archive(dim, qd_bounds, niche_num, name, use_stats)
    archive.load(name)
    return archive

def _init_map_worker(archive, fitness, bounds, me_params, cma_params):
    global _map_worker_args
    _map_worker_args = archive, fitness, bounds, me_params, cma_params


def _run_map_worker(seed):
    archive, fitness, bounds, me_params, cma_params = _map_worker_args
    run_map_elites_(archive, fitness, bounds, Generator(PCG64DXSM(seed)), me_params, cma_params)


def run_workers(target, args_per_worker):
    pending = {}
    started = []
    cancelled = mp.Event()
    try:
        for args in args_per_worker:
            worker = Process(target=target, args=args, kwargs={"cancelled": cancelled})
            worker.start()
            started.append(worker)
            pending[worker.sentinel] = worker
        while pending:
            for sentinel in wait(pending):
                worker = pending.pop(sentinel)
                worker.join()
                if worker.exitcode:
                    raise RuntimeError(
                        f"{target.__name__} worker {worker.pid} exited with code {worker.exitcode}"
                    )
    finally:
        # Stop between batches so a sibling process cannot be killed while
        # holding a shared archive lock.
        cancelled.set()
        for worker in started:
            worker.join()
            worker.close()


def optimize_map_elites_(archive: Archive,
                         fitness: QDFitness,
                         bounds: Bounds,
                         workers: int,
                         me_params: OptParams,
                         cma_params: OptParams,
                         seed: Optional[SeedSequence] = None) -> None:
    seed = SeedSequence() if seed is None else seed
    streams = (Generator(PCG64DXSM(s)) for s in seed.spawn(workers))
    run_workers(run_map_elites_, (
        (archive, fitness, bounds, rg, me_params, cma_params) for rg in streams
    ))

def run_map_elites_(archive: Archive,
                    fitness: QDFitness,
                    bounds: Bounds,
                    rg: Generator,
                    me_params: OptParams,
                    cma_params: OptParams,
                    cancelled: Optional[Any] = None) -> None:
    generations = me_params.get('generations', 10)
    chunk_size = me_params.get('chunk_size', 20)
    use_sbx = me_params.get('use_sbx', True)
    if chunk_size < (2 if use_sbx else 1):
        raise ValueError("invalid batch size for MAP-Elites variation")
    dis_c = me_params.get('dis_c', 20)
    dis_m = me_params.get('dis_m', 20)
    iso_sigma = me_params.get('iso_sigma', 0.02)
    line_sigma = me_params.get('line_sigma', 0.2)
    cma_generations = cma_params.get('cma_generations', 20)
    archive.create_views()
    select_n = archive.capacity
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        for _ in range(generations):
            if cancelled is not None and cancelled.is_set():
                return
            if use_sbx:
                pop = archive.random_xs(select_n, chunk_size, rg)
                xs = variation_(pop, bounds.lb, bounds.ub, rg, dis_c, dis_m)
            else:
                x1 = archive.random_xs(select_n, chunk_size, rg)
                x2 = archive.random_xs(select_n, chunk_size, rg)
                xs = iso_dd_(x1, x2, bounds.lb, bounds.ub, rg, iso_sigma, line_sigma)
            yds = [fitness(x) for x in xs]
            descs = np.array([yd[1] for yd in yds])
            niches = archive.index_of_niches(descs)
            for i in range(len(yds)):
                archive.set(niches[i], yds[i], xs[i])
            archive.argsort()
            select_n = archive.get_occupied() or archive.capacity

        for _ in range(cma_generations):
            if cancelled is not None and cancelled.is_set():
                return
            optimize_cma_(archive, fitness, bounds, rg, cma_params, cancelled)

def optimize_cma_(archive: Archive,
                  fitness: QDFitness,
                  bounds: Bounds,
                  rg: Generator,
                  cma_params: OptParams,
                  cancelled: Optional[Any] = None) -> None:
    select_n = cma_params.get('best_n', 100)
    x0, _, _ = archive.random_xs_one(select_n, rg)
    sigma = cma_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    popsize = cma_params.get('popsize', 31)
    es = cmaescpp.ACMA_C(archive.dim, bounds, x0 = x0,
                         popsize = popsize, input_sigma = sigma, rg = rg)
    maxiters = cma_params.get('maxiters', 100)
    stall_criterion = cma_params.get('stall_criterion', 5)
    old_ys = None
    last_improve = 0
    for iter in range(maxiters):
        if cancelled is not None and cancelled.is_set():
            return
        xs = es.ask()
        improvement, ys = update_archive(archive, xs, fitness)
        sorted_ys = np.sort(ys)
        if old_ys is not None and (sorted_ys < old_ys).any():
            last_improve = iter
        if last_improve + stall_criterion < iter:
            # no improvement
            break
        if es.tell(improvement) != 0:
            break
        old_ys = sorted_ys

@np.errstate(invalid='ignore')
def update_archive(archive: Archive, xs: np.ndarray,
                   fitness: Optional[QDFitness] = None,
                   yds: Optional[ArrayLike] = None,
                   batch_fitness: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not len(xs):
        return np.empty(0), np.empty(0)
    # evaluate population, update archive and determine ranking
    popsize = len(xs)
    if batch_fitness is not None and yds is None:
        ys, descs = batch_fitness(xs)
        ys = np.asarray(ys, dtype=float)
        descs = np.asarray(descs, dtype=float)
        if ys.shape != (popsize,) or descs.shape != (popsize, archive.qd_dim):
            raise ValueError("batch_fitness must return one score and descriptor per candidate")
    else:
        if yds is None:
            yds = [fitness(x) for x in xs]
        descs = np.array([yd[1] for yd in yds])
        ys = np.fromiter((yd[0] for yd in yds), dtype=float, count=popsize)
    niches = archive.index_of_niches(descs)
    oldys = np.frombuffer(archive.ys, dtype=np.float64)[niches]
    improvement = ys - oldys
    for i in np.flatnonzero(improvement < 0.):
        archive.set(niches[i], (ys[i], descs[i]), xs[i])
    empty = np.isneginf(improvement)
    if empty.any():
        norm_min = np.min(improvement, where=~empty, initial=np.inf)
        if empty.all():
            norm_min = 0.
        improvement[empty] = norm_min + (ys[empty] - ys.max() - 1E-9)
    # return both improvement compared to archive elites  and real fitness
    return improvement, ys

@njit()
def get_grid_indices(ds: np.ndarray, capacity: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    rdim = int(capacity ** (1/ds.shape[1]) + 0.5)
    ds_norm = (ds - lb) / (ub - lb)
    indices = np.empty(len(ds), dtype=np.int32)
    for i, d in enumerate(ds_norm):
        index = 0
        f = 1
        for di in d:
            index += f * int(rdim*di)
            f *= rdim
        indices[i] = max(0, min(capacity-1, int(index)))
    return indices

class Archive(object):
    """Multi-processing map elites archive.
        Stores decision vectors, fitness values ,
        description vectors, niche centers and statistics for the x-values"""

    def __init__(self,
                 dim: int,
                 qd_bounds: Bounds,
                 capacity: int,
                 name: Optional[str] = "",
                 use_stats: bool = False
                ) -> None:
        """Creates an empty archive."""
        if dim <= 0 or capacity <= 0 or len(qd_bounds.lb) == 0:
            raise ValueError("archive dimensions and capacity must be positive")
        self.dim = dim
        self.qd_dim = len(qd_bounds.lb)
        self.qd_bounds = Bounds(np.array(qd_bounds.lb, dtype=np.float64),
                                np.array(qd_bounds.ub, dtype=np.float64))
        self.desc_lb = self.qd_bounds.lb
        self.desc_scale = self.qd_bounds.ub - self.qd_bounds.lb
        self.capacity = capacity
        self.name = name
        self.cs = None
        self.locks = tuple(mp.Lock() for _ in range(min(capacity, 32)))
        self.use_stats = use_stats
        self.reset()

    def reset(self) -> None:
        """Resets all submitted solutions but keeps the niche centers."""
        self.xs = Shared2d(np.zeros((self.capacity, self.dim), dtype=np.float64))
        self.ds = Shared2d(np.empty((self.capacity, self.qd_dim), dtype = np.float64))
        self.create_views()
        self.ys = mp.RawArray(ct.c_double, self.capacity)
        self.counts = mp.RawArray(ct.c_long, self.capacity) # count
        self.occupied = mp.Value(ct.c_long, 0)
        self.stats = mp.RawArray(ct.c_double, self.capacity * self.dim * 4 if self.use_stats else 0)
        np.frombuffer(self.ys, dtype=np.float64).fill(np.inf)
        self.ds_view.fill(np.inf)
        if self.use_stats:
            stats = np.frombuffer(self.stats, dtype=np.float64).reshape(self.capacity, 4, self.dim)
            stats[:, 2, :] = np.inf
            stats[:, 3, :] = -np.inf

    def init_niches(self, samples_per_niche: int = 10, seed: Optional[int] = None) -> None:
        """Computes the niche centers using KMeans and builds the KDTree for niche determination."""
        # If samples_per_niche > 0 cvt-clustering is used, else grid-clustering is used.
        self.cvt_clustering = samples_per_niche > 0
        if self.cvt_clustering:
            set_KDTree(self, None, self.capacity, self.qd_bounds, samples_per_niche, seed)
            self.cs = mp.RawArray(ct.c_double, self.capacity * self.qd_dim)
            self.set_cs(self.centers)
        elif hasattr(self, "kdt"):
            del self.kdt
            self.cs = None

    def __getstate__(self):
        # These arrays must be recreated from the shared storage after spawn.
        state = self.__dict__.copy()
        state.pop("xs_view", None)
        state.pop("ds_view", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.create_views()

    def get_occupied_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ys = self.get_ys()
        occupied = (ys < np.inf)
        return ys[occupied], self.ds_view[occupied], self.xs_view[occupied]

    def join(self, archive: Archive) -> None:
        ys, ds, xs = archive.get_occupied_data()
        if len(ys):
            niches = self.index_of_niches(ds)
            for niche, y, d, x in zip(niches, ys, ds, xs):
                self.set(niche, (y, d), x)
        archive.argsort()

    def fname(self, name: str) -> str:
        """Archive file name."""
        return f'arch.{name}.{self.capacity}.{self.dim}.{self.qd_dim}'

    def save(self, name: str) -> None:
        """Saves the archive to disc."""
        np.savez_compressed(self.fname(name),
                            xs=self.xs_view,
                            ds=self.ds_view,
                            ys=self.get_ys(),
                            cs=self.get_cs() if self.cvt_clustering else np.empty(0),
                            stats=self.get_stats(),
                            counts=self.get_counts()
                            )

    def load(self, name: str) -> None:
        """Loads the archive from disc."""
        self.cs = mp.RawArray(ct.c_double, self.capacity * self.qd_dim)
        with np.load(self.fname(name) + '.npz') as data:
            self.cvt_clustering = len(data['cs']) > 0
            xs = data['xs']
            ds = data['ds']
            self.xs.view()[:] = xs
            self.ds.view()[:] = ds
            self.set_ys(data['ys'])
            if self.cvt_clustering:
                self.set_cs(data['cs'])
            self.counts[:] = data['counts']
            stats = data['stats']
            if len(stats) == len(self.stats):
                self.set_stats(stats)
        self.occupied.value = np.count_nonzero(self.get_ys() < np.inf)
        self.dim = xs.shape[1]
        self.qd_dim = ds.shape[1]
        self.capacity = xs.shape[0]
        if self.cvt_clustering:
            set_KDTree(self, self.get_cs(), None, None, None)
        elif hasattr(self, "kdt"):
            del self.kdt

    def index_of_niches(self, ds: ArrayLike) -> np.ndarray:
        if hasattr(self, "kdt"): # use k-means clusters
            return self.kdt.query(self.encode_d(ds), k=1, return_distance=False,
                                  sort_results=False)[:, 0]
        if not _has_numba:
            values = np.asarray(ds, dtype=np.float64, order="C")
            if values.ndim == 2 and values.shape[1] == self.qd_dim:
                side = int(self.capacity ** (1 / self.qd_dim) + .5)
                try:
                    return _fcmaes_ext._grid_indices(values, self.desc_lb, self.desc_scale,
                                                     self.capacity, side)
                except OverflowError:
                    pass
        return get_grid_indices(ds, self.capacity, self.qd_bounds.lb, self.qd_bounds.ub)

    def in_niche_filter(self,
                        fit: Callable[[ArrayLike], float],
                        index: int) -> in_niche_filter:
        """Creates a fitness function wrapper rejecting out of niche arguments."""
        return in_niche_filter(fit, index, self.index_of_niches)

    def set(self,
            i: int,
            yd: np.ndarray,
            x: np.ndarray) -> None:
        """Adds a solution to the archive if it improves the corresponding niche.
        Updates solution."""
        y, d = yd
        with self.locks[i % len(self.locks)]:
            self._update_stats(i, x)
            yold = self.get_y(i)
            if y < yold:
                self.xs_view[i] = x
                self.ds_view[i] = d
                self.set_y(i, y)
                if yold == np.inf:
                    with self.occupied.get_lock():
                        self.occupied.value += 1

    def update_stats(self, i: int, x: np.ndarray) -> None:
        """Updates solution statistics."""
        with self.locks[i % len(self.locks)]:
            self._update_stats(i, x)

    def _update_stats(self, i: int, x: np.ndarray) -> None:
        self.counts[i] += 1
        count = self.counts[i]
        if self.use_stats:
            mean, m2, minimum, maximum = np.frombuffer(self.stats, dtype=np.float64,
                count=4*self.dim, offset=i*4*self.dim*8).reshape(4,self.dim)
            diff = x - mean
            mean += diff * (1./count)
            m2 += np.multiply(diff, diff) * ((count-1)/count)
            np.minimum(x, minimum, out=minimum)
            np.maximum(x, maximum, out=maximum)

    def get_occupied(self) -> int:
        return self.occupied.value

    def get_count(self, i: int) -> int:
        return self.counts[i]

    def get_counts(self) -> np.ndarray:
        return np.array(self.counts[:])

    def get_x_mean(self, i: int) -> np.ndarray:
        return self.get_stat(i, 0)

    def get_x_stdev(self, i: int) -> np.ndarray:
        count = self.get_count(i)
        if count == 0:
            return np.zeros(self.dim)
        else:
            qmean = np.array(self.get_stat(i, 1))
            return np.sqrt(qmean * (1./count))

    def get_x_min(self, i: int) -> np.ndarray:
        return self.get_stat(i, 2)

    def get_x_max(self, i: int) -> np.ndarray:
        return self.get_stat(i, 3)

    def create_views(self) -> None: # needs to be called in the target process
        self.xs_view = self.xs.view()
        self.ds_view = self.ds.view()

    def encode_d(self, d: ArrayLike) -> np.ndarray:
        return (d - self.desc_lb) / self.desc_scale

    def decode_d(self, d: ArrayLike) -> np.ndarray:
        return (d * self.desc_scale) + self.desc_lb

    def get_x(self, i: int) -> np.ndarray:
        return self.xs_view[i]

    def get_d(self, i: int) -> np.ndarray:
        return self.ds_view[i]

    def get_xs(self) -> np.ndarray:
        return self.xs.view()

    def get_ds(self) -> np.ndarray:
        return self.ds.view()

    def get_y(self, i: int) -> float:
        return self.ys[i]

    def get_ys(self) -> np.ndarray:
        return np.array(self.ys[:])

    def get_qd_score(self) -> float:
        ys = self.get_ys()
        occupied = (ys != np.inf)
        ys = ys[occupied]
        if len(ys) == 0:
            return 0
        min_y = np.amin(ys)
        if min_y > 0: # if all y > 0 use sum of reciprocal
            return np.sum(np.reciprocal(ys, where = ys!=0, out=np.zeros_like(ys, dtype=float)))
        else: # else use only the negative ones
            neg = (ys < 0)
            ys = ys[neg]
            return np.sum(-ys)

    def set_y(self, i: int, y: float) -> None:
        self.ys[i] = y

    def set_ys(self, ys: ArrayLike) -> None:
        for i in range(len(ys)):
            self.set_y(i, ys[i])

    def get_c(self, i: int) -> np.ndarray:
        return self.cs[i*self.qd_dim:(i+1)*self.qd_dim]

    def get_cs(self) -> np.ndarray:
        return np.array([self.get_c(i) for i in range(self.capacity)])

    def get_cs_decoded(self) -> np.ndarray:
        return self.decode_d(np.array([self.get_c(i) for i in range(self.capacity)]))

    def set_c(self, i: int, c: ArrayLike) -> None:
        self.cs[i*self.qd_dim:(i+1)*self.qd_dim] = c[:]

    def set_cs(self, cs: ArrayLike) -> None:
        for i in range(len(cs)):
            self.set_c(i, cs[i])

    def get_stat(self, i: int, j: int) -> np.ndarray:
        p = 4*i+j
        return self.stats[p*self.dim:(p+1)*self.dim]

    def get_stats(self) ->  np.ndarray:
        return np.array(self.stats[:])

    def set_stat(self, i: int, j: int, stat: ArrayLike) -> None:
        p = 4*i+j
        self.stats[p*self.dim:(p+1)*self.dim] = stat[:]

    def set_stats(self, stats: ArrayLike) -> None:
        self.stats[:] = stats[:]

    def random_xs(self, best_n: int, chunk_size: int, rg: Generator) -> np.ndarray:
        selection = rg.integers(0, best_n, chunk_size)
        if best_n < self.capacity:
            selection = self.si[selection]
        return self.xs_view[selection]

    def random_xs_one(self, best_n: int, rg: Generator) -> Tuple[np.ndarray, float, int]:
        best_n = min(best_n, self.get_occupied())
        if best_n <= 0:
            i = int(rg.integers(self.capacity))
        else:
            order = self.argsort()
            i = int(order[rg.integers(best_n)])
        with self.locks[i % len(self.locks)]:
            return self.get_x(i).copy(), self.get_y(i), i

    def argsort(self) -> np.ndarray:
        """Sorts the archive according to its niche values."""
        self.si = np.argsort(np.frombuffer(self.ys, dtype=np.float64))
        return self.si

    def dump(self, n: Optional[int] = None) -> None:
        """Dumps the archive content."""
        if n is None:
            n = self.capacity
        ys = self.get_ys()
        si = np.argsort(ys)
        for i in range(n):
            print(si[i], ys[si[i]], self.get_d(si[i]), self.get_x(si[i]))

    def info(self) -> str:
        occ = self.get_occupied()
        score = self.get_qd_score()
        best_y = np.amin(self.get_ys())
        count = np.sum(self.get_counts())
        return f'{occ} {score:.3f} {best_y:.3f} {count}'

class wrapper(object):
    """Fitness function wrapper for multi processing logging."""

    def __init__(self,
                 fit: QDFitness,
                 qd_dim: int,
                 interval: Optional[int] = 1000000,
                 save_interval: Optional[int] = 1E20) -> None:

        self.fit = fit
        self.evals = mp.RawValue(ct.c_int, 0)
        self.best_y = mp.RawValue(ct.c_double, np.inf)
        self.t0 = perf_counter()
        self.qd_dim = qd_dim
        self.interval = interval
        self.save_interval = save_interval
        self.lock = mp.Lock()

    def __call__(self, x: ArrayLike) -> Tuple[float, np.ndarray]:
        try:
            if np.isnan(x).any():
                return np.inf, np.zeros(self.qd_dim)
            with self.lock:
                self.evals.value += 1
            log = self.evals.value % self.interval == 0
            save = self.evals.value % self.save_interval == 0
            y, desc = self.fit(x)
            if np.isnan(y) or np.isnan(desc).any():
                return np.inf, np.zeros(self.qd_dim)
            y0 = y if np.isscalar(y) else sum(y)
            if y0 < self.best_y.value:
                self.best_y.value = y0
                log = True
            if log:
                archinfo = self.archive.info() if hasattr(self, 'archive') else ''
                logger.info(
                    f'{dtime(self.t0)} {archinfo} {self.evals.value:.0f} {self.evals.value/(1E-9 + dtime(self.t0)):.1f} {self.best_y.value:.3f} {list(x)}'
                )
            if save and hasattr(self, 'archive'):
                self.archive.save(f'{self.evals.value}')
            return y, desc
        except Exception as ex:
            print(str(ex))
            return np.inf, np.zeros(self.qd_dim)

class in_niche_filter(object):
    """Fitness function wrapper rejecting out of niche arguments."""

    def __init__(self,
                 fit: QDFitness,
                 index: int,
                 index_of_niches: Callable[[ArrayLike], np.ndarray]) -> None:
        self.fit = fit
        self.index_of_niches = index_of_niches
        self.index = index

    def __call__(self, x: ArrayLike) -> float:
        y, desc = self.fit(x)
        if self.index_of_niches([desc])[0] == self.index:
            return y
        else:
            return np.inf

def variation_(pop: np.ndarray,
               lower: np.ndarray,
               upper: np.ndarray,
               rg: Generator,
               dis_c: float = 20,
               dis_m: float = 20) -> np.ndarray:
    """Generate offspring individuals using SBX (Simulated Binary Crossover) and mutation."""
    dis_c *= 0.5 + 0.5*rg.random() # vary spread factors randomly
    dis_m *= 0.5 + 0.5*rg.random()
    pop = pop[:(len(pop) // 2) * 2][:]
    (n, d) = np.shape(pop)
    parent_1 = pop[:n // 2, :]
    parent_2 = pop[n // 2:, :]
    shape = (n // 2, d)
    mu = rg.random(shape)
    signs = rg.integers(2, size=shape)
    keep = rg.random(shape) >= .5
    beta = np.ones(shape)
    left = keep & (mu <= .5)
    right = keep & ~left
    beta[left] = np.power(2 * mu[left], 1 / (dis_c + 1))
    beta[right] = np.power(2 * mu[right], -1 / (dis_c + 1))
    beta[keep & (signs == 1)] *= -1
    parent_mean = (parent_1 + parent_2) / 2
    parent_diff = (parent_1 - parent_2) / 2
    offspring = np.vstack((parent_mean + beta * parent_diff, parent_mean - beta * parent_diff))
    site = rg.random((n, d)) < 1.0 / d
    mu = rg.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.broadcast_to(lower, (n, d)), np.broadcast_to(upper, (n, d))
    norm = (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(np.abs(1. - norm), dis_m + 1.),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (1. - np.power(
                               2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(np.abs(1. - norm), dis_m + 1.),
                               1. / (dis_m + 1.)))
    return np.clip(offspring, lower, upper)

def iso_dd_(x1: np.ndarray,
            x2: np.ndarray,
            lower: np.ndarray,
            upper: np.ndarray,
            rg: Generator,
            iso_sigma: float = 0.01,
            line_sigma: float = 0.2) -> np.ndarray:
    """Generate offspring individuals using Iso+Line."""
    a = rg.normal(0, iso_sigma, x1.shape)
    b = rg.normal(0, line_sigma, x2.shape)
    a += x1
    b *= x1 - x2
    a += b
    return np.clip(a, lower, upper, out=a)

def get_centers_(niche_num: int, dim: int, samples_per_niche: int,
                 seed: Optional[int] = None) -> np.ndarray:
    cache = Path("voronoi_cache")
    cache.mkdir(exist_ok=True)
    stem = f"centers_{niche_num}_{dim}_{samples_per_niche}"
    if seed is not None:
        stem += f"_v1_seed{seed}"
    path = cache / f"{stem}.npz"
    if path.exists():
        with np.load(path) as data:
            return data["cs"]
    center_rng = rng if seed is None else np.random.default_rng(seed)
    descs = center_rng.uniform(0, 1, (niche_num * samples_per_niche, dim))
    k_means = KMeans(init="k-means++", n_clusters=niche_num, n_init=1,
                     random_state=seed)
    k_means.fit(descs)
    centers = k_means.cluster_centers_
    # Another process may compute the same centers. Readers see complete files.
    with NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            np.savez_compressed(temp, cs=centers)
            temp.flush()
            temp.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return centers
