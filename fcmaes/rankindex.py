"""Immutable native rank-space indexes for closed, axis-aligned box queries."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from fcmaes import _fcmaes_ext as _ext


class BoxBatch(NamedTuple):
    """Query i owns ids[offsets[i]:offsets[i+1]] and one row in stats."""
    offsets: np.ndarray
    ids: np.ndarray
    stats: np.ndarray


def _array(values, ndim):
    a = np.require(values, dtype=np.float64, requirements=["C", "A"])
    if a.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array")
    return a


class _BoxIndex:
    @property
    def dim(self):
        return self._native.dim

    def query(self, low, high) -> np.ndarray:
        """Return matching input row IDs (uint32). Both endpoints are inclusive."""
        return self._native.query(_array(low, 1), _array(high, 1))

    def query_batch(self, low, high) -> BoxBatch:
        """Run a matrix of queries in C++. See stat_names for the diagnostic columns."""
        return BoxBatch(*self._native.query_batch(_array(low, 2), _array(high, 2)))


class BordaBoxIndex(_BoxIndex):
    """Static Borda rank index. Reports IDs in the shortest axis's rank order.

    Input is a finite (points, dimensions) matrix. The index owns its data.
    Ties are included in full. Within a tie the native sort determines order.
    """
    stat_names = ("axis_candidates", "score_candidates")

    def __init__(self, points):
        self._native = _ext.BordaBoxIndex(_array(points, 2))


class GridBoxIndex(_BoxIndex):
    """Static rank grid. Reports IDs by ascending cell key, then input row ID.

    query_volume is the positive nominal fraction used to choose the resolution.
    The last three diagnostic counts are exact. touched_cells is a float64
    count of theoretical cells and may round above 2**53.
    """
    stat_names = ("touched_cells", "nonempty_cells", "full_cells", "point_tests")

    def __init__(self, points, query_volume):
        self._native = _ext.GridBoxIndex(_array(points, 2), float(query_volume))
