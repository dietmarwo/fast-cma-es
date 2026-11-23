# Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
"""Pareto archives, incremental two-objective hypervolume, and front distance.

The native archive (``_fcmaescpp/include/paretofront2d.h``) is derived from
``bitehv2.h`` in Aleksey Vaneev's BiteOpt repository,
https://github.com/avaneev/biteopt (MIT license). See
``docs/pareto_archive.md`` for the semantics and a MODE example.
"""
from __future__ import annotations

import operator
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from fcmaes._fcmaes_ext import ParetoArchive2D as _Archive
from fcmaes._fcmaes_ext import ParetoArchiveND as _StrictArchive
from fcmaes._fcmaes_ext import front_igd as _front_igd


def _payload_batch(xs, rows, width):
    if xs is None:
        if rows and width:
            raise ValueError("payloads are required when payload_dim > 0")
        return np.empty((rows, width), dtype=np.float64)
    payloads = np.require(xs, dtype=np.float64, requirements=["C", "A"])
    if payloads.ndim != 2:
        raise ValueError("payloads must be a matrix")
    return payloads


class ParetoArchive2D:
    """All unique nondominated pairs of two minimized objectives, with payloads.

    Every insertion updates the hypervolume dominated within the reference
    box. Points with an objective above ``reference`` are not archived: they
    add nothing to the hypervolume, and ``add`` returns ``False`` for them.
    There is no lower bound. Equal objective pairs keep the first payload.

    Parameters
    ----------
    reference : array_like, shape (2,)
        Finite hypervolume reference point.
    payload_dim : int, optional
        Length of the payload stored with each point, usually the decision
        vector dimension. 0 (the default) stores no payload.
    """

    def __init__(self, reference: ArrayLike, payload_dim: int = 0):
        dim = operator.index(payload_dim)
        reference = self._vector(reference)
        self._native = _Archive(reference, dim)
        self._reference = tuple(float(r) for r in reference)

    @staticmethod
    def _vector(values: ArrayLike) -> np.ndarray:
        result = np.ascontiguousarray(values, dtype=np.float64)
        if result.ndim != 1:
            raise ValueError("expected a one-dimensional vector")
        return result

    @property
    def reference(self) -> Tuple[float, float]:
        """Hypervolume reference point."""
        return self._reference

    @property
    def size(self) -> int:
        """Number of archived points."""
        return self._native.size

    @property
    def payload_dim(self) -> int:
        """Length of the payload stored with each point."""
        return self._native.payload_dim

    def __len__(self) -> int:
        return self.size

    def add(self, y: ArrayLike, x: Optional[ArrayLike] = None) -> bool:
        """Insert objective pair ``y`` with payload ``x``.

        Returns ``True`` if the point was archived. The payload is copied only
        then. Raises ``ValueError`` for non-finite objectives and
        ``OverflowError`` if the area exceeds the float range; the archive is
        unchanged in both cases.
        """
        payload = np.empty(0) if x is None else self._vector(x)
        return self._native.add(self._vector(y), payload)

    def add_batch(self, ys: ArrayLike, xs: Optional[ArrayLike] = None) -> np.ndarray:
        """Insert the rows of ``ys`` (shape ``(n, 2)``) with payloads ``xs``
        (shape ``(n, payload_dim)``) in order.

        Returns a boolean mask of the archived rows. Shape errors are raised
        before any change. If a row raises, the rows before it stay applied
        and the message names the failing row.
        """
        values = np.ascontiguousarray(ys, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("objectives must have shape (rows, 2)")
        payloads = _payload_batch(xs, len(values), self.payload_dim)
        return self._native.add_batch(values, payloads)

    def front(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return copies ``(xs, ys)`` of payloads and objectives, sorted by the
        first objective."""
        return self._native.front()

    def hypervolume(self) -> float:
        """Area dominated by the archive within the reference box."""
        return self._native.hypervolume()

    def clear(self) -> None:
        """Remove all points, keeping the reference and payload length."""
        self._native.clear()


class ParetoArchiveND:
    """Strict minimization front in nobj dimensions, with optional fixed-width payloads.

    Equal nondominated rows coexist. Inputs are finite objectives with no
    reference bound. Snapshots follow objective-sum order and then arrival
    order within equal rounded sums.
    """
    def __init__(self, nobj: int, payload_dim: int = 0):
        nobj, payload_dim = operator.index(nobj), operator.index(payload_dim)
        if nobj < 1 or payload_dim < 0:
            raise ValueError("nobj must be positive and payload_dim nonnegative")
        self._native = _StrictArchive(nobj, payload_dim)

    @property
    def nobj(self) -> int:
        return self._native.nobj

    @property
    def payload_dim(self) -> int:
        return self._native.payload_dim

    @property
    def size(self) -> int:
        return self._native.size

    def __len__(self) -> int:
        return self.size

    def add(self, y, x=None) -> bool:
        """Return whether this row entered the archive at the time it was added."""
        values = np.require(y, dtype=np.float64, requirements=["C", "A"])
        payload = np.empty(0) if x is None else np.require(x, dtype=np.float64, requirements=["C", "A"])
        if values.ndim != 1 or payload.ndim != 1:
            raise ValueError("objectives and payload must be vectors")
        return self._native.add(values, payload)

    def add_batch(self, ys, xs=None) -> np.ndarray:
        """Return a boolean admission mask in arrival order. A bad numeric row retains the valid prefix."""
        values = np.require(ys, dtype=np.float64, requirements=["C", "A"])
        if values.ndim != 2 or values.shape[1] != self.nobj:
            raise ValueError("objectives must have shape (rows, nobj)")
        return self._native.add_batch(values, _payload_batch(xs, len(values), self.payload_dim))

    def front(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return independent (payloads, objectives) matrices in archive order."""
        return self._native.front()

    def clear(self) -> None:
        self._native.clear()


def igd(reference: ArrayLike, front: ArrayLike) -> float:
    """Mean Euclidean distance from each reference row to its nearest supplied front row.

    Both arguments are finite matrices of equal positive width. The reference
    must contain at least one row. An empty front has infinite IGD.
    """
    reference = np.require(reference, dtype=np.float64, requirements=["C", "A"])
    front = np.require(front, dtype=np.float64, requirements=["C", "A"])
    if reference.ndim != 2 or front.ndim != 2:
        raise ValueError("reference and front must be matrices")
    return _front_igd(reference, front)
