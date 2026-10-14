# Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
"""Exact 2D minimization archive and incremental hypervolume.

One update owner; native calls retain the GIL. This archive has no capacity
limit and owns accepted payloads. It is separate from MODE's breeding
population and from the bounded multiprocessing result store.
"""
from __future__ import annotations

import operator
from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from fcmaes._fcmaes_ext import ParetoArchive2D as _Archive


class ParetoArchive2D:
    """Archive unique nondominated objective pairs with a fixed HV reference.

    Bounds apply to objectives. Equal pairs retain the first payload. Use
    ``payload_dim=0`` for HV only or the decision dimension to retain designs.
    Numeric errors leave the failing row unapplied; earlier batch rows remain.
    Allocation failures have no strong exception guarantee.
    """

    def __init__(self, lower: ArrayLike, reference: ArrayLike, payload_dim: int = 0):
        dim = operator.index(payload_dim)
        self._native = _Archive(self._vector(lower), self._vector(reference), dim)

    @staticmethod
    def _vector(values: ArrayLike) -> np.ndarray:
        result = np.ascontiguousarray(values, dtype=np.float64)
        if result.ndim != 1:
            raise ValueError("expected a one-dimensional vector")
        return result

    @property
    def size(self) -> int:
        """Number of unique nondominated objective pairs."""
        return self._native.size

    @property
    def payload_dim(self) -> int:
        """Fixed width of retained decision vectors."""
        return self._native.payload_dim

    def __len__(self) -> int:
        return self.size

    def add(self, y: ArrayLike, x: Optional[ArrayLike] = None) -> bool:
        """Insert an objective pair; copy its payload only if accepted."""
        payload = np.empty(0) if x is None else self._vector(x)
        return self._native.add(self._vector(y), payload)

    def add_batch(self, ys: ArrayLike, xs: Optional[ArrayLike] = None) -> np.ndarray:
        """Insert rows in order and return their boolean acceptance mask.

        Shapes must be ``(n, 2)`` and ``(n, payload_dim)``. For an empty batch
        pass a matrix with shape ``(0, 2)``. Noncontiguous inputs are copied.
        Shape errors precede mutation. A numeric error names its zero-based
        row; the successfully processed prefix remains in the archive.
        """
        values = np.ascontiguousarray(ys, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("objectives must have shape (rows, 2)")
        if xs is None:
            if self.payload_dim and len(values):
                raise ValueError("payloads are required when payload_dim > 0")
            payloads = np.empty((len(values), self.payload_dim), dtype=np.float64)
        else:
            payloads = np.ascontiguousarray(xs, dtype=np.float64)
            if payloads.ndim != 2:
                raise ValueError("payloads must be a matrix")
        return self._native.add_batch(values, payloads)

    def front(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return owned ``(xs, ys)`` snapshots in first-objective order.

        Snapshots remain valid after clear, later updates or archive deletion.
        Export traverses the ordered map without changing its representation.
        """
        return self._native.front()

    def hypervolume(self) -> float:
        """Current dominated union area without exporting the front."""
        return self._native.hypervolume()

    def clear(self) -> None:
        """Reset contents and HV while retaining bounds and payload width."""
        self._native.clear()
