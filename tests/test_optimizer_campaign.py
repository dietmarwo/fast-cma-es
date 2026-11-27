from multiprocessing import Value
from threading import Condition

import numpy as np
import pytest
from scipy.optimize import Bounds

from fcmaes import bitecpp, cmaescpp, decpp, diversifier, mapelites, modecpp


def box(dim, low=-1., high=1.):
    return Bounds(np.full(dim, low), np.full(dim, high))


@pytest.mark.parametrize("budget", [1, 3, 4, 61])
def test_delayed_cma_decodes_once_and_counts_actual_calls(budget):
    points = []

    def objective(x):
        points.append(x.copy())
        return float(x @ x)

    result = cmaescpp.minimize(
        objective, box(4, 10., 20.), x0=np.full(4, 15.), popsize=12,
        workers=4, max_evaluations=budget, rg=np.random.default_rng(101),
    )
    assert result.success
    assert result.nfev == len(points) == budget
    assert any(np.all((10. < x) & (x < 20.)) for x in points[:4])
    assert result.fun == min(float(x @ x) for x in points)
    assert result.fun == pytest.approx(float(result.x @ result.x))


@pytest.mark.parametrize("depth", [1, 3])
def test_bite_initialization_uses_distinct_population_rows(depth):
    kwargs = dict(dim=12, bounds=box(12), x0=np.zeros(12), M=depth,
                  max_evaluations=19)
    batched = bitecpp.Bite_C(**kwargs, rg=np.random.default_rng(104), batch_size=16)
    single = bitecpp.Bite_C(**kwargs, rg=np.random.default_rng(104), batch_size=1)
    rows = batched.ask()
    assert len(np.unique(rows, axis=0)) == 16
    assert not rows.flags.writeable
    saved = rows.copy()
    for row in rows:
        next_x = single.ask()
        np.testing.assert_array_equal(next_x[0], row)
        assert single.tell(np.array([row @ row])) == 0
    assert batched.tell(np.arange(16., 0., -1.)) == 0
    np.testing.assert_array_equal(batched.result().x, rows[-1])
    last = batched.ask()
    assert last.shape == (3, 12)
    assert batched.tell(np.array([2., 3., 4.])) == 0
    assert batched.ask().shape == (0, 12)
    del batched
    np.testing.assert_array_equal(rows, saved)


def test_de_best_is_independent_of_population_slot_and_nonfinite_rows():
    de = decpp.DE_C(bounds=box(4), popsize=8, rg=np.random.default_rng(11))
    xs = de.ask()
    assert de.tell(np.r_[0., np.arange(1., 7.), np.inf]) == 0
    result = de.result()
    assert result.fun == 0.
    np.testing.assert_array_equal(result.x, xs[0])
    assert result.nfev == 8
    with pytest.raises(ValueError, match="four population"):
        decpp.DE_C(bounds=box(2), popsize=3)


def test_de_slow_completion_still_uses_the_original_candidate():
    started = Condition()
    points = []

    def objective(x):
        with started:
            points.append(x.copy())
            if len(points) == 1:
                assert started.wait_for(lambda: len(points) >= 55, timeout=5)
                return -10.
            started.notify_all()
        return 0.

    result = decpp.minimize(
        objective, bounds=box(4), popsize=4,
        workers=2, max_evaluations=70, rg=np.random.default_rng(13),
    )
    assert result.success and result.nfev == len(points) == 70
    assert result.fun == -10.
    np.testing.assert_array_equal(result.x, points[0])


@pytest.mark.parametrize("budget", [1, 7, 33])
def test_de_strict_budget_includes_temporal_trials(budget):
    calls = []

    def objective(x):
        calls.append(float(x @ x))
        return calls[-1]

    result = decpp.minimize(objective, bounds=box(6), popsize=8, strict_budget=True,
                            max_evaluations=budget, rg=np.random.default_rng(11))
    assert result.success and result.nfev == len(calls) == budget
    assert result.fun == min(calls)


def test_vectorized_native_optimizers_match_row_callbacks():
    bounds = box(4, 0., 1.)
    seed = lambda: np.random.default_rng(49)
    cma = dict(bounds=bounds, x0=np.full(4, .5), max_evaluations=128,
               popsize=16)
    scalar = cmaescpp.minimize(lambda x: np.sum(x*x), rg=seed(), **cma)
    vector = cmaescpp.minimize(lambda x: np.sum(x*x, axis=1), rg=seed(),
                              vectorized=True, **cma)
    assert scalar.fun == vector.fun
    np.testing.assert_array_equal(scalar.x, vector.x)
    invalid = cmaescpp.minimize(lambda xs: np.empty(len(xs) - 1), rg=seed(),
                               vectorized=True, **cma)
    assert not invalid.success and invalid.nfev == 0

    def rows(x):
        return np.stack((x[..., 0], np.sum((x - .5)**2, axis=-1)), axis=-1)

    args = dict(nobj=2, ncon=0, bounds=bounds, popsize=16, max_evaluations=256)
    expected = modecpp.minimize(rows, rg=seed(), **args)
    actual = modecpp.minimize(rows, rg=seed(), vectorized=True, **args)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    assert modecpp.minimize(lambda xs: np.empty((len(xs), 3)), rg=seed(),
                            vectorized=True, **args) == (None, None)


class SphereQD:
    def __init__(self):
        self.count = Value("l", 0)

    def __call__(self, x):
        with self.count.get_lock():
            self.count.value += 1
        return 1. + float(x @ x), x[:2]

    def batch(self, xs):
        with self.count.get_lock():
            self.count.value += len(xs)
        return 1. + np.einsum("ij,ij->i", xs, xs), xs[:, :2]


def test_map_fresh_archive_native_emitter_pool_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = dict(bounds=box(6), qd_bounds=box(2), niche_num=25, samples_per_niche=0,
                iterations=3, workers=1, seed=33, use_stats=True,
                me_params=dict(chunk_size=8, generations=2),
                cma_params=dict(cma_generations=1, popsize=8, maxiters=2))
    first = mapelites.optimize_map_elites(SphereQD(), **args)
    pooled = mapelites.optimize_map_elites(SphereQD(), persistent_workers=True, **args)
    for left, right in zip(first.get_occupied_data(), pooled.get_occupied_data()):
        np.testing.assert_array_equal(left, right)
    assert first.get_counts().sum() > 3*2*8  # Native CMA also evaluated samples.
    first.save("checkpoint")
    restored = mapelites.load_archive("checkpoint", args["bounds"], args["qd_bounds"],
                                      niche_num=25, use_stats=True)
    np.testing.assert_array_equal(restored.get_stats(), first.get_stats())
    for left, right in zip(first.get_occupied_data(), restored.get_occupied_data()):
        np.testing.assert_array_equal(left, right)
    resumed = mapelites.optimize_map_elites(SphereQD(), archive=restored, **args)
    assert resumed.get_occupied() >= first.get_occupied()
    assert np.all(resumed.get_ys() <= first.get_ys())


@pytest.mark.parametrize("workers,solver", [(1, "BITE_CPP"), (2, "CMA_CPP"), (2, "DE_CPP")])
def test_diversifier_short_batches_fit_total_budget(workers, solver):
    fit = SphereQD()
    result = diversifier.minimize(
        fit, box(6), box(2), niche_num=25, samples_per_niche=0, use_stats=True,
        max_evals=71, workers=workers, seed=36,
        opt_params=dict(solver=solver, popsize=8, batch_size=4, batch_fitness=fit.batch,
                        max_evals=32),
    )
    assert result.use_stats
    assert fit.count.value == 71
    ys, ds, xs = result.get_occupied_data()
    assert result.get_occupied() == len(xs) > 0
    expected_y, expected_d = fit.batch(xs)
    np.testing.assert_allclose(ys, expected_y)
    np.testing.assert_array_equal(ds, expected_d)


class BrokenQD:
    def __call__(self, x):
        raise RuntimeError("injected evaluation failure")


@pytest.mark.parametrize("entry", [mapelites.optimize_map_elites, diversifier.minimize])
def test_process_failure_is_reported_and_archive_can_be_reused(entry):
    archive = mapelites.empty_archive(4, box(2), 16, 0)
    archive.get_xs()[:] = 0.
    config = (dict(iterations=1, me_params=dict(generations=1))
              if entry is mapelites.optimize_map_elites else dict(max_evals=16))
    with pytest.raises(RuntimeError, match="worker .* exited with code"):
        entry(BrokenQD(), box(4), box(2), archive=archive, seed=99, workers=2, **config)
    archive.set(0, (1., np.zeros(2)), np.zeros(4))
    assert archive.get_occupied() == 1


def test_grid_indices_keep_legacy_clamping_and_integer_bounds():
    archive = mapelites.empty_archive(4, Bounds([0, 0], [1, 1]), 19, 0)
    grid = np.array([[-.15, -.15], [0., 0.], [.25, .25], [.99, .99],
                     [1., 1.], [1.45, -.50], [np.nextafter(.5, 0), .5]])
    expected = mapelites.get_grid_indices(grid, 19, np.zeros(2), np.ones(2))
    np.testing.assert_array_equal(archive.index_of_niches(grid), expected)


def test_cvt_seed_and_archive_centers_survive_cache_and_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = dict(niche_num=16, samples_per_niche=4, max_evals=64,
                  seed=84965, inline=True, workers=1, use_stats=True,
                  opt_params=dict(solver="CMA_CPP", popsize=8, max_evals=32))
    a = diversifier.minimize(SphereQD(), box(6), box(2), **params)
    b = diversifier.minimize(SphereQD(), box(6), box(2), **params)
    np.testing.assert_array_equal(a.get_cs(), b.get_cs())
    for left, right in zip(a.get_occupied_data(), b.get_occupied_data()):
        np.testing.assert_array_equal(left, right)
    a.save("cvt")
    c = mapelites.load_archive("cvt", box(6), box(2), niche_num=16, use_stats=True)
    np.testing.assert_array_equal(c.get_cs(), a.get_cs())
    ds = a.get_occupied_data()[1]
    np.testing.assert_array_equal(c.index_of_niches(ds), a.index_of_niches(ds))


def _cma_stop_generation(flat_from, flat_generations, generations=60):
    """Tell CMA-ES generation bests that fall by one, except for a flat stretch."""
    dim, popsize = 3, 6
    es = cmaescpp.ACMA_C(dim, box(dim), x0=[0.2]*dim, popsize=popsize,
                         rg=np.random.default_rng(5))
    level = 1000.0
    for generation in range(generations):
        xs = np.asarray(es.ask())
        if not flat_from <= generation < flat_from + flat_generations:
            level -= 1.0
        # Ranks follow a sphere, so the search state stays ordinary.
        ys = np.empty(popsize)
        ys[np.argsort(np.sum(xs**2, axis=1))] = level + np.arange(popsize)
        stop = es.tell(ys)
        if stop != 0:
            return generation, stop
    return None, 0


def test_cma_history_stop_needs_a_full_window_of_equal_bests():
    # The history holds 10 + int(30*dim/popsize) = 25 generation bests. Before
    # the shift was fixed every entry was a copy of the newest one, and five
    # equal bests stopped the run.
    assert _cma_stop_generation(20, 5) == (None, 0)
    assert _cma_stop_generation(20, 20) == (None, 0)
    assert _cma_stop_generation(20, 30) == (44, 5)


def test_map_elites_runs_its_cma_emitter_unless_it_is_switched_off():
    args = dict(bounds=box(4), qd_bounds=box(2), niche_num=32, samples_per_niche=4,
                iterations=2, workers=1, seed=7,
                me_params=dict(generations=3, chunk_size=8))
    quiet, emitting = SphereQD(), SphereQD()
    mapelites.optimize_map_elites(quiet, cma_params=dict(cma_generations=0), **args)
    assert quiet.count.value == 2*3*8
    mapelites.optimize_map_elites(emitting, cma_params=dict(cma_generations=2, popsize=8,
                                                             maxiters=3), **args)
    assert 2*3*8 < emitting.count.value <= 2*(3*8 + 2*3*8)
