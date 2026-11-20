"""Public API and wheel tests for the Python and native jSO ports."""

import numpy as np
import pytest
from scipy.optimize import Bounds

from fcmaes import jso, jsocpp, retry
from fcmaes.optimizer import Jso_cpp, Jso_python, Sequence, jso_cma


@pytest.fixture(params=[jso, jsocpp], ids=['python', 'native'])
def solver(request):
    return request.param


def sphere(x):
    delta = x - 0.7
    return float(delta @ delta)


def zero(x):
    return 0.0


def bad_score(x):
    raise ValueError("simulation failed")


def test_ask_tell_matches_serial_and_parallel(solver):
    bounds = Bounds([-5.0] * 3, [5.0] * 3)
    options = dict(popsize=24, max_evaluations=509, min_popsize=5,
                   memory_size=3, p_max=0.35, p_min=0.2, archive_rate=0.5, runid=11)
    seed = 941
    state = solver.Jso(bounds, rg=np.random.default_rng(seed), **options)
    batches = []
    bests = []
    while len(rows := state.ask()):
        assert rows.shape[1] == 3
        assert np.all(rows >= bounds.lb) and np.all(rows <= bounds.ub)
        batches.append(len(rows))
        state.tell([sphere(row) for row in rows])
        bests.append(state.result().fun)
    result = state.result()
    assert result.nfev == sum(batches) == 509
    assert result.nit == len(batches) - 1
    assert batches[0] == 24 and state.popsize == 5 and min(batches) < 24
    assert result.fun < 0.1 and result.fun == sphere(result.x)
    assert all(a >= b for a, b in zip(bests, bests[1:]))
    assert state.ask().shape == (0, 3)
    for workers in (1, 2):
        other = solver.minimize(sphere, bounds, workers=workers,
                                rg=np.random.default_rng(seed), **options)
        np.testing.assert_array_equal(other.x, result.x)
        assert (other.fun, other.nfev, other.nit, other.status) == (
            result.fun, result.nfev, result.nit, result.status)


def test_default_population_and_hard_budget(solver):
    state = solver.Jso([(-1.0, 1.0)] * 4, max_evaluations=71, rg=np.random.default_rng(1))
    assert state.popsize == 69
    assert len(state.ask()) == 69
    state.tell(np.ones(69))
    assert len(state.ask()) == 2
    state.tell([0.0, 1.0])
    assert state.result().nfev == 71

    evaluated = []

    def objective(x):
        evaluated.append(x.copy())
        return sphere(x)

    result = solver.minimize(objective, [(0.0, 1.0)] * 4, max_evaluations=3,
                             rg=np.random.default_rng(17))
    assert result.nfev == len(evaluated) == 3 and result.nit == 0
    assert result.fun == min(map(sphere, evaluated))
    assert any(np.array_equal(x, result.x) for x in evaluated)


def test_target_stops_at_a_prefix_or_batch(solver):
    bounds = [(0.0, 1.0)] * 2
    options = dict(popsize=10, max_evaluations=17, stop_fitness=1.0)
    serial = solver.minimize(zero, bounds, workers=1, **options)
    parallel = solver.minimize(zero, bounds, workers=2, **options)
    assert (serial.nfev, serial.nit, serial.status, serial.fun) == (1, 0, 1, 0.0)
    assert (parallel.nfev, parallel.nit, parallel.status, parallel.fun) == (10, 0, 1, 0.0)
    tied = solver.minimize(zero, bounds, popsize=10, max_evaluations=17, stop_fitness=0.0)
    assert (tied.nfev, tied.nit, tied.status) == (17, 1, 0)


def test_rejected_calls_and_snapshots(solver):
    state = solver.Jso([(-2.0, 3.0)] * 2, popsize=8, max_evaluations=40)
    assert state.result().nfev == 0
    with pytest.raises(RuntimeError):
        state.tell([])
    rows = state.ask()
    first = rows[0].copy()
    rows[0] = [200.0, 300.0]
    pop = state.population()
    np.testing.assert_array_equal(pop[0], first)
    pop[0] = [100.0, 100.0]
    with pytest.raises(RuntimeError):
        state.ask()
    for values in ([], np.ones(7), np.ones(9), np.ones((8, 1)), 1.0):
        with pytest.raises(ValueError):
            state.tell(values)
        assert state.result().nfev == 0
    state.tell([0.0] + [1.0] * 7)
    snapshot = state.result()
    np.testing.assert_array_equal(snapshot.x, first)
    snapshot.x[:] = -100
    np.testing.assert_array_equal(state.result().x, first)
    del state
    np.testing.assert_array_equal(rows[0], [200.0, 300.0])


@pytest.mark.parametrize('bounds,options', [
    (None, {}), ([], {}), ([(1.0, -1.0)], {}), ([(0.0, np.inf)], {}),
    ([(-1.7e308, 1.7e308)], {}), ([(0, 1)], {'max_evaluations': 0}),
    ([(0, 1)], {'stop_fitness': np.nan}), ([(0, 1)], {'p_min': 0.5}),
    ([(0, 1)], {'p_max': np.nan}), ([(0, 1)], {'archive_rate': -1}),
])
def test_invalid_options(solver, bounds, options):
    with pytest.raises(ValueError):
        solver.Jso(bounds, **options)


def test_fixed_dimensions_and_large_finite_numbers(solver):
    state = solver.Jso([(1e308, 1.7e308), (2.0, 2.0)], max_evaluations=251,
                       popsize=12, archive_rate=0, rg=np.random.default_rng(43))
    batch = 0
    while len(xs := state.ask()):
        assert np.all(np.isfinite(xs)) and np.all(xs[:, 1] == 2.0)
        assert np.all((xs[:, 0] >= 1e308) & (xs[:, 0] <= 1.7e308))
        if batch <= 1:
            scores = [np.nan, -np.inf, np.inf] * 4
        else:
            scores = xs[:, 0] / 1e308
        state.tell(scores[:len(xs)])
        batch += 1
    result = state.result()
    assert result.success and result.nfev == 251
    assert result.fun == result.x[0] / 1e308


@pytest.mark.parametrize('workers', [1, 2])
def test_simulation_failures_count_toward_budget(solver, workers):
    result = solver.minimize(bad_score, [(0, 1)], max_evaluations=33, workers=workers)
    assert result.nfev == 33 and not result.success and result.fun == np.inf
    assert 0 <= result.x[0] <= 1


@pytest.mark.parametrize('adapter', [Jso_python, Jso_cpp])
def test_retry_adapter(adapter):
    bounds = Bounds([-4.0] * 2, [4.0] * 2)
    optimizer = adapter(max_evaluations=310, popsize=12)
    result = retry.minimize(sphere, bounds, optimizer=optimizer, num_retries=2, workers=2)
    assert result.nfev == 620
    assert result.fun == sphere(result.x)


def test_jso_cma_runs_both_stages_and_replays_incumbent():
    bounds = Bounds([-5.0] * 3, [5.0] * 3)
    points = []

    def objective(x):
        points.append(x.copy())
        return sphere(x)

    seq = jso_cma(1550, jso_popsize=20)
    assert [solver.max_evaluations for solver in seq.optimizers] == [620, 930]
    x, value, count = seq.minimize(objective, bounds, rg=np.random.default_rng(57))
    assert 620 < count == len(points) <= 1550
    assert np.all(x >= bounds.lb) and np.all(x <= bounds.ub)
    assert value == pytest.approx(sphere(x), abs=1e-15)
    assert value <= min(map(sphere, points[:620]))


def test_sequence_retains_a_point_after_nonfinite_stages():
    seq = Sequence([Jso_cpp(max_evaluations=12), Jso_python(max_evaluations=12)])
    x, value, count = seq.minimize(bad_score, Bounds([0.0], [1.0]))
    assert count == 24 and 0 <= x[0] <= 1 and value == np.inf


# The tests below fail when the search itself is broken, not only the API.
# They hold for any random stream (no recorded values), so they also hold for
# the C++ backend, whose stream depends on the C++ standard library.


def test_equal_values_replace_parents_and_worse_values_do_not(solver):
    """A trial replaces its parent when it is not worse (``<=``)."""
    state = solver.Jso([(-1.0, 1.0)] * 3, popsize=8, max_evaluations=10**9,
                       rg=np.random.default_rng(3))
    state.tell(np.ones(len(state.ask())))
    trials = state.ask()
    state.tell(np.ones(8))
    np.testing.assert_array_equal(state.population(), trials)
    state.ask()
    state.tell(np.full(8, 2.0))
    np.testing.assert_array_equal(state.population(), trials)


def test_population_follows_the_linear_reduction(solver):
    initial, budget = 30, 1000
    state = solver.Jso([(-1.0, 1.0)] * 4, popsize=initial, max_evaluations=budget,
                       rg=np.random.default_rng(5))
    rows = state.ask()
    state.tell([sphere(x) for x in rows])
    used, expected = initial, initial  # the initial batch is not reduced
    while len(rows := state.ask()):
        assert len(rows) == min(expected, budget - used)
        state.tell([sphere(x) for x in rows])
        used += len(rows)
        expected = int(np.floor(initial + (4 - initial) * used / budget + 0.5))
        assert state.popsize == expected
    assert used == budget and expected == 4


def test_out_of_box_mutants_move_halfway_to_the_bound(solver):
    """With the optimum in a corner, many mutants leave the box. The repair
    puts them halfway between the parent and the bound, so trials land on
    the bound only by rare rounding (none in 60 calibration runs); clipping
    puts more than 1,000 coordinates there."""
    # A short run: the population stays far enough from the bound (above
    # 1e-12 in 60 calibration runs) that no trial reaches it by rounding.
    state = solver.Jso([(0.0, 1.0)] * 5, popsize=20, max_evaluations=600,
                       rg=np.random.default_rng(7))
    state.tell([np.sum(x) for x in state.ask()])
    halved = on_bound = 0
    while len(rows := state.ask()):
        parents = state.population()[:len(rows)]
        assert np.all(rows >= 0.0) and np.all(rows <= 1.0)
        halved += np.count_nonzero(rows == 0.5 * parents)
        on_bound += np.count_nonzero(rows == 0.0)
        state.tell([np.sum(x) for x in rows])
    assert halved > 100 and on_bound < 50


def test_first_generation_crosses_most_coordinates(solver):
    """Early crossover rates are drawn around 0.8 and 0.9 and floored at 0.7,
    so about 80% of the coordinates of a trial come from the mutant."""
    dim = 40
    state = solver.Jso([(-1.0, 1.0)] * dim, popsize=40, max_evaluations=10**6,
                       rg=np.random.default_rng(11))
    state.tell([sphere(x) for x in state.ask()])
    parents = state.population()
    trials = state.ask()
    changed = np.mean(trials != parents)
    assert 0.7 < changed < 0.95


def rosen(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def test_solves_rosenbrock(solver):
    """10-D Rosenbrock, 20,000 evaluations: in 80 calibration runs (40 per
    backend) every run ended below 2e-4, with medians near 2e-7. Disabling
    the memory update, the archive, the population reduction or the p-best
    restriction raises the median above 1e-2."""
    values = [solver.minimize(rosen, [(-5.0, 5.0)] * 10, max_evaluations=20_000,
                              rg=np.random.default_rng(seed)).fun for seed in range(5)]
    assert np.median(values) < 1e-3
