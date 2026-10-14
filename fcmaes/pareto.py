# Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
"""Exact 2D Pareto archive with incremental hypervolume.

Based on Aleksey Vaneev's 2D archive from BiteOpt (``bitehv2.h``). See
``docs/pareto_archive.md`` for the semantics and a MODE example.
"""
from __future__ import annotations

import operator
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from fcmaes._fcmaes_ext import ParetoArchive2D as _Archive


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
        """Return copies ``(xs, ys)`` of payloads and objectives, sorted by the
        first objective."""
        return self._native.front()

    def hypervolume(self) -> float:
        """Area dominated by the archive within the reference box."""
        return self._native.hypervolume()

    def clear(self) -> None:
        """Remove all points, keeping the reference and payload length."""
        self._native.clear()
