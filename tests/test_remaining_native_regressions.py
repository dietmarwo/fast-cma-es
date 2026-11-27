"""Regression checks for the remaining native optimizer and front algorithms."""
import numpy as np
import pytest
from scipy.optimize import Bounds
from scipy.spatial.distance import cdist

from fcmaes import crfmnescpp, dacpp, jsocpp, pgpecpp
from fcmaes.pareto import ParetoArchive2D, ParetoArchiveND, igd


@pytest.mark.parametrize("module", [crfmnescpp, pgpecpp])
@pytest.mark.parametrize("budget", [1, 31, 32, 33])
@pytest.mark.parametrize("normalize", [False, True])
def test_batched_es_returns_best_evaluated_row(module, budget, normalize):
    dimensions = 8
    bounds = Bounds(np.full(dimensions, 2.), np.full(dimensions, 3.))
    observed = []

    def objective(x):
        value = -x.sum()
        observed.append((value, x.copy()))
        return value

    result = module.minimize(
        objective, bounds, x0=np.full(dimensions, 2.99),
        input_sigma=.9, popsize=32, max_evaluations=budget,
        normalize=normalize, rg=np.random.default_rng(101),
    )

    assert result.success and result.nfev == len(observed) == budget
    best, point = min(observed, key=lambda entry: entry[0])
    assert result.fun == best
    np.testing.assert_array_equal(result.x, point)
    assert np.all(result.x >= bounds.lb) and np.all(result.x <= bounds.ub)


def test_crfmnes_state_reports_physical_incumbent_with_penalty():
    bounds = Bounds(np.full(8, 2.), np.full(8, 3.))
    state = crfmnescpp.CRFMNES_C(
        8, bounds, x0=np.full(8, 2.99), input_sigma=.9,
        normalize=True, rg=np.random.default_rng(101),
    )
    observed = []
    for _ in range(2):
        population = state.ask()
        scores = -population.sum(axis=1)
        observed.extend(zip(scores, population))
        state.tell(scores)
    score, point = min(observed, key=lambda entry: entry[0])
    result = state.result()
    assert result.nfev == 64
    assert result.fun == score
    np.testing.assert_array_equal(result.x, point)


@pytest.mark.parametrize("dim,budget", [(16, 1), (16, 40), (128, 260)])
def test_annealing_charges_every_finite_difference_probe(dim, budget):
    bounds = Bounds(np.full(dim, -5.), np.full(dim, 5.))
    observed = []

    def objective(x):
        value = x @ x
        observed.append((value, x.copy()))
        return value

    result = dacpp.minimize(
        objective, bounds, x0=np.full(dim, 1.5), max_evaluations=budget,
        use_local_search=True, rg=np.random.default_rng(101),
    )
    assert result.success and result.nfev == len(observed) == budget
    best, point = min(observed, key=lambda entry: entry[0])
    assert result.fun == best
    np.testing.assert_array_equal(result.x, point)


def test_annealing_reset_respects_one_evaluation_budget():
    result = dacpp.minimize(
        lambda _: np.inf, Bounds([-1., -1.], [1., 1.]),
        x0=np.zeros(2), max_evaluations=1, rg=np.random.default_rng(2),
    )
    assert result.nfev == 1
    assert result.nit == 0


@pytest.mark.parametrize("module", [crfmnescpp, pgpecpp, jsocpp])
def test_vectorized_objective_owns_physical_batches(module):
    observed = []

    def objective(xs):
        observed.append(xs)
        return np.einsum("ij,ij->i", xs, xs)

    result = module.minimize(
        objective, Bounds([-2.] * 5, [2.] * 5), popsize=8, max_evaluations=17,
        vectorized=True, rg=np.random.default_rng(6),
    )

    xs = np.concatenate(observed)
    ys = np.einsum("ij,ij->i", xs, xs)
    assert result.success and len(xs) == result.nfev == 17
    assert len(observed) == 3
    assert result.fun == ys.min()
    np.testing.assert_array_equal(result.x, xs[np.argmin(ys)])


@pytest.mark.parametrize("module", [crfmnescpp, pgpecpp])
def test_vectorized_callback_rejects_invalid_result_shape(module):
    result = module.minimize(
        lambda xs: np.zeros((len(xs), 1)), Bounds([-2.] * 4, [2.] * 4),
        vectorized=True, max_evaluations=32, rg=np.random.default_rng(5),
    )
    assert not result.success


def test_jso_vectorized_target_counts_a_complete_batch():
    r = jsocpp.minimize(
        lambda xs: np.zeros(len(xs)), Bounds([-1.] * 3, [1.] * 3),
        max_evaluations=10, popsize=8, stop_fitness=1., vectorized=True,
        rg=np.random.default_rng(4),
    )
    assert r.success and r.status == 1 and r.nfev == 8


def _sequential_sum(values):
    answer = 0.
    for value in values:
        answer += float(value)
    return answer


def test_strict_archive_indexes_thin_front_then_handles_wide_updates():
    rng = np.random.default_rng(3)
    raw = rng.uniform(size=(512, 4))
    raw /= raw.sum(axis=1, keepdims=True)
    # The first coordinate is constant. An indexed axis must be chosen
    # from the remaining coordinates. Add two duplicates and some outliers.
    values = np.column_stack((np.full(len(raw), .25), raw))
    values = np.r_[values, values[[50, 180]], np.nextafter(values[7:9], np.inf),
                   [[.1, .14, .14, .14, .14], [.5, .2, .2, .2, .2]]]
    alive = []
    expected = []
    for i, point in enumerate(values):
        keep = all(not (np.all(values[j] <= point) and np.any(values[j] < point)) for j in alive)
        expected.append(keep)
        if keep:
            alive = [j for j in alive
                     if not (np.all(point <= values[j]) and np.any(point < values[j]))]
            alive.append(i)
    alive.sort(key=lambda i: (_sequential_sum(values[i]), i))

    archive = ParetoArchiveND(5, 1)
    np.testing.assert_array_equal(archive.add_batch(values, np.arange(len(values))[:, None]), expected)
    payload, front = archive.front()
    np.testing.assert_array_equal(payload[:, 0], alive)
    np.testing.assert_array_equal(front, values[alive])
    archive.clear()
    assert len(archive) == 0 and archive.add(values[1], [1])


def test_2d_archive_retains_valid_prefix_in_owning_mask_path():
    archive = ParetoArchive2D([1., 1.], 1)
    with pytest.raises(ValueError, match="row 2"):
        archive.add_batch([[.2, .5], [.3, .4], [np.nan, .2], [.1, .1]], [[0], [1], [2], [3]])
    payload, _ = archive.front()
    np.testing.assert_array_equal(payload[:, 0], [0, 1])
    assert archive.add_batch(np.empty((0, 2)), np.empty((0, 1))).dtype == np.bool_


@pytest.mark.parametrize("dim,rows", [(1, 256), (4, 512), (8, 2304)])
def test_igd_specialized_search_matches_exhaustive_oracle(dim, rows):
    rng = np.random.default_rng(5)
    front = rng.uniform(size=(rows, dim))
    reference = rng.uniform(size=(320, dim))
    expected = cdist(reference, front).min(axis=1).mean()
    assert igd(front, reference) == pytest.approx(expected, rel=2e-15)
    assert igd(front * 1e200, reference * 1e200) == pytest.approx(expected * 1e200, rel=2e-15)


def test_igd_on_flat_data_with_distant_references():
    rng = np.random.default_rng(7)
    front = np.zeros((2560, 8))
    front[:, 4] = rng.random(len(front))
    reference = 100. + rng.random((300, 8))
    expected = cdist(reference, front).min(axis=1).mean()
    assert igd(front, reference) == pytest.approx(expected, rel=2e-15)


def _strict_archive_by_definition(ys):
    front, mask = [], []
    for y in ys:
        if any(np.all(f <= y) and np.any(f < y) for f in front):
            mask.append(False)
            continue
        front = [f for f in front if not (np.all(y <= f) and np.any(y < f))]
        front.append(y)
        mask.append(True)
    return np.array(mask), len(front)


@pytest.mark.parametrize("nobj", [3, 4, 5, 8, 12])
@pytest.mark.parametrize("stream", ["near sums", "equal sums", "uniform", "ties", "worsening"])
def test_strict_archive_matches_the_definition_for_every_kernel(nobj, stream):
    # Three objectives use the unrolled kernel, four and more the branch-free
    # one; sums equal to rounding switch to the coordinate index after 128
    # entries, and the later wide rows switch back.
    rng = np.random.default_rng(nobj)
    u = rng.random((400, nobj))
    simplex = u / u.sum(axis=1, keepdims=True)
    ys = {
        "near sums": simplex + 1e-3*rng.random((400, nobj)),
        "equal sums": np.vstack([simplex[:300], 3*u[300:] - 1]),
        "uniform": u,
        "ties": np.round(u*3)/3,
        "worsening": simplex*np.linspace(1, 2, 400)[:, None],
    }[stream]
    archive = ParetoArchiveND(nobj)
    mask = np.concatenate([archive.add_batch(ys[:250]),
                           [archive.add(y) for y in ys[250:260]],
                           archive.add_batch(ys[260:])])
    expected, size = _strict_archive_by_definition(ys)
    np.testing.assert_array_equal(mask, expected)
    assert len(archive) == size
