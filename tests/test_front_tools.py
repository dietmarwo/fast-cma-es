import numpy as np
import pytest
from scipy.optimize import Bounds
from scipy.spatial.distance import cdist

from fcmaes import archivede
from fcmaes.pareto import ParetoArchive2D, ParetoArchiveND, igd
from fcmaes.rankindex import BordaBoxIndex, GridBoxIndex


@pytest.mark.parametrize("index_type", [BordaBoxIndex, GridBoxIndex])
@pytest.mark.parametrize("d", [1, 3, 6])
def test_box_indices_match_closed_scan_and_own_coordinates(index_type, d):
    rg = np.random.default_rng(173 + d)
    data = rg.integers(-10, 11, size=(600, 2*d)).astype(float)[:, ::2]
    original = data.copy()
    index = index_type(data) if index_type is BordaBoxIndex else index_type(data, 0.015)
    data[:] = 80
    low = rg.uniform(-12, 5, size=(32, d))
    high = low + rg.uniform(1, 14, size=low.shape)
    low[0], high[0] = -np.inf, np.inf
    low[1] = high[1] = original[0]
    low[2], high[2] = 2., -2.
    low[3, 0] = np.nan
    batch = index.query_batch(low, high)
    assert batch.offsets.dtype == np.uint64 and batch.ids.dtype == np.uint32
    assert batch.stats.shape == (32, len(index.stat_names))
    for i, (lo, hi) in enumerate(zip(low, high)):
        ids = batch.ids[batch.offsets[i]:batch.offsets[i + 1]]
        np.testing.assert_array_equal(ids, index.query(lo, hi))
        expected = np.flatnonzero(((original >= lo) & (original <= hi)).all(axis=1))
        np.testing.assert_array_equal(np.sort(ids), expected)
        if index_type is BordaBoxIndex:
            axis, band = batch.stats[i]
            assert len(ids) <= band <= axis
        else:
            touched, occupied, full, tested = batch.stats[i]
            assert 0 <= full <= occupied <= touched and tested <= len(original)
    empty = index.query_batch(np.empty((0, d)), np.empty((0, d)))
    assert empty.offsets.tolist() == [0] and empty.ids.size == 0
    with pytest.raises(ValueError):
        index.query([0.]*(d + 1), [1.]*d)
    with pytest.raises(ValueError):
        index.query_batch(low, high[:-1])


def test_index_limits_sparse_domain_and_compact_rank_boundary():
    with pytest.raises(ValueError):
        BordaBoxIndex([[np.inf]])
    with pytest.raises(ValueError):
        GridBoxIndex([[0]], 0)
    with pytest.raises((ValueError, RuntimeError)):
        GridBoxIndex(np.zeros((1, 65)), 1.0)
    for ctor in [lambda p: GridBoxIndex(p, 0.1), BordaBoxIndex]:
        assert ctor(np.empty((0, 2))).query([0, 0], [1, 1]).size == 0
    p = np.random.default_rng(114).random((40, 40))
    g = GridBoxIndex(p, 1.0)
    low, hi = np.zeros(40), np.ones(40)
    hi[-1] = np.median(p[:, -1])
    got = g.query_batch(low[None], hi[None])
    np.testing.assert_array_equal(np.sort(got.ids), np.flatnonzero(p[:, -1] <= hi[-1]))
    assert got.stats[0, 0] == 2**39
    for n in [65536, 65537]:
        x = np.arange(n, dtype=float)
        b = BordaBoxIndex(np.column_stack((x, x[::-1])))
        assert b.query([n-2, 1], [n-2, 1]).tolist() == [n-2]


def _dominates(y, z):
    return np.all(y <= z) and np.any(y < z)


@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_strict_archive_masks_payloads_and_order(d):
    rg = np.random.default_rng(375)
    pts = rg.integers(0, 12, (170, d)).astype(float)
    pts[::15] = pts[0]
    payloads = np.arange(len(pts), dtype=float)[:, None]
    expected, keep = [], []
    for i, y in enumerate(pts):
        ok = not any(_dominates(pts[j], y) for j in keep)
        expected.append(ok)
        if ok:
            keep = [j for j in keep if not _dominates(y, pts[j])] + [i]
    keep.sort(key=lambda i: (sum(pts[i]), i))
    a = ParetoArchiveND(d, 1)
    np.testing.assert_array_equal(a.add_batch(pts, payloads), expected)
    xs, ys = a.front()
    a.clear()
    np.testing.assert_array_equal(xs, payloads[keep])
    np.testing.assert_array_equal(ys, pts[keep])
    assert len(a) == 0 and a.front()[1].shape == (0, d)


def test_strict_archive_exact_line_migration_signed_zero_and_error_contract():
    a = ParetoArchiveND(2, 1)
    rows = [(i % 100, 100 - i % 100) for i in range(300)]
    np.testing.assert_array_equal(a.add_batch(rows, np.arange(300)[:, None]), True)
    assert a.add([20, 10], [300])
    assert a.add([20, 10], [301])
    assert not a.add([20, 11], [302])
    xs, ys = a.front()
    assert xs[(ys == [20, 10]).all(axis=1)].ravel().tolist() == [300, 301]
    for d in [2, 3]:
        b = ParetoArchiveND(d)
        base = [2.0**54, 0.] + [0.]*(d-2)
        worse = [2.0**54, 1.] + [0.]*(d-2)
        assert b.add(worse) and b.add(base) and not b.add(worse) and len(b) == 1
        b.clear()
        with pytest.raises(ValueError, match="row 1"):
            b.add_batch([base, [np.nan]*d])
        assert len(b) == 1
        with pytest.raises(ValueError):
            b.add_batch([base, base], [[1], [2]])
        assert len(b) == 1
    a = ParetoArchiveND(2)
    assert a.add([-0., 0.]) and a.add([0., -0.])
    assert len(a) == 2
    assert ParetoArchive2D([1, 1]).add([1, 1])


@pytest.mark.parametrize("d,n", [(1, 3), (2, 39), (2, 40), (2, 180), (3, 127), (3, 128), (5, 142)])
def test_igd_exact_search_matches_cartesian_oracle(d, n):
    rg = np.random.default_rng(d+n)
    r, f = rg.normal(size=(32, d)), rg.normal(size=(n, d))
    assert igd(f, r) == pytest.approx(cdist(r, f).min(axis=1).mean(), rel=2e-15)
    if d == 2:
        f = np.column_stack((np.arange(n), -np.arange(n)))
        assert igd(f, r) == pytest.approx(cdist(r, f).min(axis=1).mean(), rel=2e-15)
    assert np.isinf(igd(np.empty((0, d)), r))
    with pytest.raises(ValueError):
        igd(f, np.empty((0, d)))
    with pytest.raises(ValueError):
        igd(np.zeros((1, d + 1)), r)
    assert igd([[1e-250, 0.]], [[0., 0.]]) == 1e-250
    assert igd([[1e250, 0.]], [[0., 0.]]) == 1e250
    # The front comes first, as in fcmaes_rust.igd: here IGD and GD differ.
    assert igd([[0., 0.]], [[0., 0.], [3., 4.]]) == 2.5
    assert igd([[0., 0.], [3., 4.]], [[0., 0.]]) == 0.0


def objective(x):
    return np.array([x[0] + x[1], 4. - x[0] - x[1]])


def test_extreme_coordinates_in_mean_distance_and_decision_bounds():
    limit = np.finfo(float).max
    assert igd([[limit]], [[-limit], [limit]]) == limit
    assert np.isinf(igd([[limit]], [[-limit]]))
    state = archivede.ArchiveDE(1, [(-limit, limit)], popsize=4, max_evaluations=4, seed=5)
    xs = state.ask()
    assert np.isfinite(xs).all() and np.ptp(xs / limit) > 0
    state.tell(xs / limit)
    np.testing.assert_array_equal(state.population()[0], xs)


@pytest.mark.parametrize("budget", [0, 3, 8, 9, 31])
def test_ask_tell_hard_cap_and_owned_snapshots(budget):
    s = archivede.ArchiveDE(2, Bounds([-2, 3, 1], [4, 3, 2]), popsize=8,
                            max_evaluations=budget, seed=100)
    seen = []
    with pytest.raises(RuntimeError):
        s.tell(np.ones((8, 2)))
    while len(x := s.ask()):
        copy = x.copy()
        x[:] = 999
        np.testing.assert_array_equal(s.ask(), copy)
        assert np.all(copy[:, 1] == 3)
        with pytest.raises(ValueError):
            s.tell(np.zeros((len(x), 3)))
        invalid = np.ones((len(x), 2))
        invalid[-1, 0] = np.nan
        with pytest.raises(ValueError):
            s.tell(invalid)
        assert s.nfev == len(seen)
        np.testing.assert_array_equal(s.ask(), copy)
        seen.extend(copy)
        s.tell([objective(row) for row in copy])
    assert s.nfev == budget
    assert s.ask().shape == (0, 3)
    px, py = s.population()
    assert px.shape == (min(8, budget), 3) and py.shape == (len(px), 2)
    fx, fy = s.front()
    assert len(fx) <= budget
    if len(fx):
        np.testing.assert_array_equal(fy, [objective(row) for row in fx])


def test_row_batch_and_process_objectives_agree():
    kw = dict(nobj=2, bounds=[(-1, 1), (-0.5, 2)], popsize=12, max_evaluations=41, seed=19)
    row = archivede.minimize(objective, **kw)
    batch = archivede.minimize(lambda x: np.column_stack((x[:, 0] + x[:, 1], 4. - x[:, 0] - x[:, 1])),
                               vectorized=True, **kw)
    par = archivede.minimize(objective, workers=2, **kw)
    for a, b, c in zip(row, batch, par):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
    with pytest.raises(ValueError):
        archivede.minimize(objective, vectorized=True, workers=2, **kw)


@pytest.mark.parametrize("n", [40, 300])
def test_ties_keep_row_order_and_match_the_documented_order(n):
    values = np.array([[-0.0, 0.0, 1.5, -2.0, 1.0][i % 5] for i in range(n)])
    want = np.argsort(values + 0.0, kind="stable")
    lo, hi = [-np.inf], [np.inf]
    np.testing.assert_array_equal(BordaBoxIndex(values[:, None]).query(lo, hi), want)
    np.testing.assert_array_equal(GridBoxIndex(values[:, None], 1.0 / n).query(lo, hi), want)


def test_seeded_runs_repeat_and_tell_takes_ys():
    def run():
        s = archivede.ArchiveDE(2, [(0., 1.)] * 3, popsize=8, max_evaluations=40, seed=3)
        while len(xs := s.ask()):
            s.tell(ys=np.column_stack((xs[:, 0], 1. - xs[:, 0] + xs[:, 1:].sum(axis=1))))
        return s.front()
    for a, b in zip(run(), run()):
        np.testing.assert_array_equal(a, b)
