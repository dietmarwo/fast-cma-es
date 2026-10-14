import csv
import gc
from pathlib import Path

import numpy as np
import pytest
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds

from fcmaes.modecpp import MODE_C
from fcmaes.pareto import ParetoArchive2D


def oracle_add(front, point, payload):
    if any(np.all(q <= point) for q, _ in front):
        return False
    front[:] = [(q, x) for q, x in front if not np.all(point <= q)]
    front.append((point.copy(), payload.copy()))
    front.sort(key=lambda item: item[0][0])
    return True


def union_area(points, reference):
    if not len(points):
        return 0.0
    rows = np.asarray(points, dtype=np.float64)
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    best = np.minimum.accumulate(rows[:, 1])
    widths = np.diff(np.r_[rows[:, 0], reference[0]])
    return np.sum(widths.astype(np.longdouble) *
                  (np.longdouble(reference[1]) - best.astype(np.longdouble)))


def fixture_cases():
    groups = {}
    with (Path(__file__).parent / "data/pareto_archive_2d.csv").open() as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["case"], []).append(
                [float(row["a"]), float(row["b"]), int(row["id"])])
    return groups


@pytest.mark.parametrize("case", list(fixture_cases()))
def test_every_fixture_prefix(case):
    archive = ParetoArchive2D([2, 2], payload_dim=1)
    front, all_points = [], []
    for a, b, id_ in fixture_cases()[case]:
        point, payload = np.array([a, b]), np.array([id_], dtype=float)
        expected = oracle_add(front, point, payload)
        assert archive.add(point, payload) == expected
        all_points.append(point)
        xs, ys = archive.front()
        assert len(archive) == archive.size == len(front)
        np.testing.assert_array_equal(ys, [p for p, _ in front])
        np.testing.assert_array_equal(xs, [x for _, x in front])
        assert archive.hypervolume() == pytest.approx(float(union_area(all_points, [2, 2])), abs=1e-11)


def test_bulk_matches_scalar_and_handles_empty_matrices():
    rows = np.asarray(fixture_cases()["antichain"])
    ys, xs = rows[:, :2], rows[:, 2:]
    a = ParetoArchive2D([2, 2], payload_dim=1)
    b = ParetoArchive2D([2, 2], payload_dim=1)
    scalar = [b.add(y, x) for y, x in zip(ys, xs)]
    np.testing.assert_array_equal(a.add_batch(ys, xs), scalar)
    for u, v in zip(a.front(), b.front()):
        np.testing.assert_array_equal(u, v)
    assert a.add_batch(np.empty((0, 2))).shape == (0,)
    empty = ParetoArchive2D([1, 1])
    xx, yy = empty.front()
    assert xx.shape == (0, 0) and yy.shape == (0, 2)
    assert empty.add_batch(np.empty((0, 2))).dtype == np.bool_
    assert empty.add([0.2, 0.3])
    assert empty.front()[0].shape == (1, 0)


def test_shape_errors_precede_mutation_and_numeric_errors_keep_prefix():
    a = ParetoArchive2D([1, 1], payload_dim=1)
    with pytest.raises(ValueError):
        a.add_batch([[0.5, 0.5], [0.2, 0.2]], [[1]])
    assert a.size == 0
    with pytest.raises(ValueError, match="row 1"):
        a.add_batch([[0.5, 0.5], [np.nan, 0.2], [0.1, 0.1]], [[1], [2], [3]])
    np.testing.assert_array_equal(a.front()[0], [[1]])
    np.testing.assert_array_equal(a.front()[1], [[0.5, 0.5]])
    for point in [[np.nan, 0], [np.inf, 0], [0, -np.inf]]:
        with pytest.raises(ValueError):
            a.add(point, [4])
    assert a.hypervolume() == 0.25


def test_snapshot_and_input_ownership_duplicate_payload_and_clear():
    a = ParetoArchive2D([1, 1], payload_dim=2)
    x = np.array([3.0, 4.0])
    assert a.add([0.2, 0.8], x)
    x[:] = -1
    assert not a.add([0.2, 0.8], [5, 6])
    xs, ys = a.front()
    a.add([0.1, 0.1], [7, 8])
    a.clear()
    assert a.size == 0 and a.hypervolume() == 0
    del a
    gc.collect()
    np.testing.assert_array_equal(xs, [[3, 4]])
    np.testing.assert_array_equal(ys, [[0.2, 0.8]])


def test_noncontiguous_inputs_and_invalid_reference():
    a = ParetoArchive2D([1, 1], payload_dim=1)
    ys = np.array([[0.2, 9, 0.8, 9], [0.8, 9, 0.2, 9]])[:, ::2]
    assert not ys.flags.c_contiguous
    np.testing.assert_array_equal(a.add_batch(ys, [[1], [2]]), [True, True])
    for ref in [[np.nan, 1], [1, np.inf], [1, 1, 1], [[1, 1]]]:
        with pytest.raises(ValueError):
            ParetoArchive2D(ref)
    with pytest.raises(ValueError):
        ParetoArchive2D([1, 1], payload_dim=-1)


def test_overflow_rollback_and_extreme_zero_area():
    a = ParetoArchive2D([1e308, 1e308])
    assert a.add([1e308, 0])
    with pytest.raises(OverflowError):
        a.add([0, 0])
    np.testing.assert_array_equal(a.front()[1], [[1e308, 0]])
    assert a.hypervolume() == 0


def test_mode_observation_preserves_run_and_retains_all_feasible_evaluations():
    import importlib.util
    path = Path(__file__).parents[1] / "examples/pareto_mode.py"
    spec = importlib.util.spec_from_file_location("pareto_mode_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plain, _, trace0 = module.run(generations=20, track=False)
    tracked, archive, trace1 = module.run(generations=20, track=True)
    np.testing.assert_array_equal(plain, tracked)
    assert len(trace0) == len(trace1) == 640
    front = []
    for (x0, y0), (x1, y1) in zip(trace0, trace1):
        np.testing.assert_array_equal(x0, x1)
        np.testing.assert_array_equal(y0, y1)
        if y1[2] <= 0:
            oracle_add(front, y1[:2], x1)
    xs, ys = archive.front()
    np.testing.assert_array_equal(xs, [x for _, x in front])
    np.testing.assert_array_equal(ys, [y for y, _ in front])


def test_cached_map_accepts_reference_after_empty_and_clear():
    a = ParetoArchive2D([1, 1], payload_dim=1)
    assert a.add([1, 1], [1])
    assert a.hypervolume() == 0
    assert not a.add([1, 1], [2])
    np.testing.assert_array_equal(a.front()[0], [[1]])
    assert a.add([0.5, 0.5], [3])
    # Outside the reference box: not archived, no error.
    assert not a.add([2, 2], [4])
    # Finiteness is checked before the cached rejection shortcut.
    with pytest.raises(ValueError):
        a.add([np.inf, 2], [4])
    assert a.hypervolume() == 0.25
    a.clear()
    assert a.add([1, 1], [5])
    np.testing.assert_array_equal(a.front()[0], [[5]])
    assert a.hypervolume() == 0


def test_outside_points_are_skipped_and_there_is_no_lower_bound():
    a = ParetoArchive2D([1, 1], payload_dim=1)
    ys = np.array([[2.0, 0.5], [0.5, 0.5], [0.5, 1.5], [-3.0, 0.9], [0.9, -2.0]])
    mask = a.add_batch(ys, np.arange(5.0)[:, None])
    np.testing.assert_array_equal(mask, [False, True, False, True, True])
    xs, front = a.front()
    np.testing.assert_array_equal(front, [[-3.0, 0.9], [0.5, 0.5], [0.9, -2.0]])
    np.testing.assert_array_equal(xs.ravel(), [3, 1, 4])
    assert a.hypervolume() == pytest.approx(float(union_area(ys[mask], [1, 1])))
    assert a.reference == (1.0, 1.0)
    # The boundary itself is inside the box.
    b = ParetoArchive2D([1, 1])
    assert b.add([1, 1]) and b.hypervolume() == 0
    assert not b.add([1 + 1e-12, 0])


def test_zdt1_mode_run_with_tight_reference():
    """Typical use: archive every MODE generation with the usual ZDT1 reference
    (1.1, 1.1). All initial points lie outside the box."""
    dim, reference = 10, [1.1, 1.1]

    def zdt1(x):
        g = 1 + 9 * x[:, 1:].sum(axis=1) / (dim - 1)
        return np.column_stack((x[:, 0], g * (1 - np.sqrt(x[:, 0] / g))))

    solver = MODE_C(2, 0, Bounds([0.0] * dim, [1.0] * dim),
                    popsize=64, rg=Generator(PCG64DXSM(7)))
    archive = ParetoArchive2D(reference, payload_dim=dim)
    front, inside = [], []
    for generation in range(150):
        xs = solver.ask()
        ys = zdt1(xs)
        mask = archive.add_batch(ys, xs)
        if generation == 0:
            assert (ys[:, 1] > reference[1]).all() and not mask.any()
        for x, y in zip(xs, ys):
            if (y <= reference).all():
                oracle_add(front, y, x)
                inside.append(y)
        solver.tell(ys)
    assert len(front) > 0
    xs, ys = archive.front()
    np.testing.assert_array_equal(ys, [y for y, _ in front])
    np.testing.assert_array_equal(xs, [x for _, x in front])
    assert archive.hypervolume() == pytest.approx(
        float(union_area(inside, reference)), abs=1e-11)
