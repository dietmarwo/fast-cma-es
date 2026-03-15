# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""ESA GTOP test functions exposed through the nanobind extension."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
np.set_printoptions(legacy='1.25')
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds

from fcmaes._fcmaes_ext import gtop_cassini1
from fcmaes._fcmaes_ext import gtop_cassini1_minlp
from fcmaes._fcmaes_ext import gtop_cassini2
from fcmaes._fcmaes_ext import gtop_cassini2_minlp
from fcmaes._fcmaes_ext import gtop_gtoc1
from fcmaes._fcmaes_ext import gtop_messenger
from fcmaes._fcmaes_ext import gtop_messengerfull
from fcmaes._fcmaes_ext import gtop_rosetta
from fcmaes._fcmaes_ext import gtop_sagas
from fcmaes._fcmaes_ext import gtop_tandem
from fcmaes._fcmaes_ext import gtop_tandem_unconstrained

PenaltyValue = 1e10
ScalarAstroCallable = Callable[[NDArray[np.float64]], float]
TandemCallable = Callable[[NDArray[np.float64], Sequence[int]], float]

astro_map: dict[str, ScalarAstroCallable] = {
    "messengerfullC": gtop_messengerfull,
    "messengerC": gtop_messenger,
    "gtoc1C": gtop_gtoc1,
    "cassini1C": gtop_cassini1,
    "cassini1minlpC": lambda x: float(gtop_cassini1_minlp(x)[0]),
    "cassini2C": gtop_cassini2,
    "cassini2minlpC": gtop_cassini2_minlp,
    "rosettaC": gtop_rosetta,
    "sagasC": gtop_sagas,
}

_tandem_map: dict[str, TandemCallable] = {
    "tandemC": gtop_tandem,
    "tandemCu": gtop_tandem_unconstrained,
}


def _as_float_array(values: ArrayLike) -> NDArray[np.float64]:
    return np.ascontiguousarray(values, dtype=np.float64)


def _safe_scalar_eval(fun: ScalarAstroCallable, x: ArrayLike) -> float:
    try:
        value = float(fun(_as_float_array(x)))
        if not math.isfinite(value):
            return PenaltyValue
        return value
    except Exception:
        return PenaltyValue


def _safe_tandem_eval(fun: TandemCallable, x: ArrayLike, seq: Sequence[int]) -> float:
    try:
        value = float(fun(_as_float_array(x), list(seq)))
        if not math.isfinite(value):
            return PenaltyValue
        return value
    except Exception:
        return PenaltyValue


def _safe_cassini1_minlp_eval(x: Sequence[float]) -> tuple[float, float]:
    try:
        dv, launch_dv = gtop_cassini1_minlp(_as_float_array(x))
        dv = float(dv)
        launch_dv = float(launch_dv)
        if not math.isfinite(dv):
            dv = PenaltyValue
        if not math.isfinite(launch_dv):
            launch_dv = PenaltyValue
        return dv, launch_dv
    except Exception:
        return PenaltyValue, PenaltyValue


def _concat_planets(x: ArrayLike, planets: Sequence[int]) -> list[float]:
    return [float(xi) for xi in x] + [float(pi) for pi in planets]


class Astrofun:
    """Provides access to ESA GTOP optimization test functions."""

    def __init__(self, name: str, fun_c: str,
                 lower: Sequence[float], upper: Sequence[float]) -> None:
        self.name = name
        self.fun_c = fun_c
        self.bounds = Bounds(lower, upper)
        self.fun = python_fun(fun_c, self.bounds)


class MessFull:
    """see https://www.esa.int/gsp/ACT/projects/gtop/messenger_full/"""

    def __init__(self) -> None:
        Astrofun.__init__(
            self, "messenger full", "messengerfullC",
            [1900.0, 3.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
             0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.1, 1.1, 1.05, 1.05, 1.05,
             -math.pi, -math.pi, -math.pi, -math.pi, -math.pi],
            [2200.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 550.0,
             0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 6.0, 6.0, 6.0, 6.0, 6.0,
             math.pi, math.pi, math.pi, math.pi, math.pi],
        )


class Messenger:
    """see https://www.esa.int/gsp/ACT/projects/gtop/messenger_reduced/"""

    def __init__(self) -> None:
        Astrofun.__init__(
            self, "messenger reduced", "messengerC",
            [1000.0, 1.0, 0.0, 0.0, 200.0, 30.0, 30.0, 30.0, 0.01, 0.01, 0.01,
             0.01, 1.1, 1.1, 1.1, -math.pi, -math.pi, -math.pi],
            [4000.0, 5.0, 1.0, 1.0, 400.0, 400.0, 400.0, 400.0, 0.99, 0.99,
             0.99, 0.99, 6.0, 6.0, 6.0, math.pi, math.pi, math.pi],
        )


class Gtoc1:
    """see https://www.esa.int/gsp/ACT/projects/gtop/gtoc1/"""

    def __init__(self) -> None:
        Astrofun.__init__(
            self, "GTOC1", "gtoc1C",
            [3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0],
            [10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0, 9000.0],
        )
        self.gfun = self.fun
        self.fun = self.gtoc1

    def gtoc1(self, x: ArrayLike) -> float:
        return self.gfun(x) - 2000000.0


class Cassini1:
    """see https://www.esa.int/gsp/ACT/projects/gtop/cassini1/"""

    def __init__(self) -> None:
        Astrofun.__init__(
            self, "Cassini1", "cassini1C",
            [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
            [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
        )


class Cassini2:
    """see https://www.esa.int/gsp/ACT/projects/gtop/cassini2/"""

    def __init__(self) -> None:
        Astrofun.__init__(
            self, "Cassini2", "cassini2C",
            [-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0,
             0.01, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.15, 1.7,
             -math.pi, -math.pi, -math.pi, -math.pi],
            [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0,
             0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0, 6.5, 291.0,
             math.pi, math.pi, math.pi, math.pi],
        )


class Rosetta:
    """see https://www.esa.int/gsp/ACT/projects/gtop/rosetta/"""

    def __init__(self) -> None:
        Astrofun.__init__(
            self, "Rosetta", "rosettaC",
            [1460.0, 3.0, 0.0, 0.0, 300.0, 150.0, 150.0, 300.0, 700.0,
             0.01, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.05, 1.05,
             -math.pi, -math.pi, -math.pi, -math.pi],
            [1825.0, 5.0, 1.0, 1.0, 500.0, 800.0, 800.0, 800.0, 1850.0,
             0.9, 0.9, 0.9, 0.9, 0.9, 9.0, 9.0, 9.0, 9.0,
             math.pi, math.pi, math.pi, math.pi],
        )


class Sagas:
    """see https://www.esa.int/gsp/ACT/projects/gtop/sagas/"""

    def __init__(self) -> None:
        Astrofun.__init__(
            self, "Sagas", "sagasC",
            [7000.0, 0.0, 0.0, 0.0, 50.0, 300.0, 0.01, 0.01, 1.05, 8.0,
             -math.pi, -math.pi],
            [9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0, 0.9, 0.9, 7.0, 500.0,
             math.pi, math.pi],
        )


class Tandem:
    """see https://www.esa.int/gsp/ACT/projects/gtop/tandem/"""

    def __init__(self, i: int, constrained: bool = True) -> None:
        self.name = ("Tandem " if constrained else "Tandem unconstrained ") + str(i + 1)
        self.fun_c = "tandemC" if constrained else "tandemCu"
        self.fun = self.tandem
        self.bounds = Bounds(
            [5475.0, 2.5, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 0.01, 0.01, 0.01,
             0.01, 1.05, 1.05, 1.05, -math.pi, -math.pi, -math.pi],
            [9132.0, 4.9, 1.0, 1.0, 2500.0, 2500.0, 2500.0, 2500.0, 0.99, 0.99,
             0.99, 0.99, 10.0, 10.0, 10.0, math.pi, math.pi, math.pi],
        )
        self.seqs = [
            [3, 2, 2, 2, 6], [3, 2, 2, 3, 6], [3, 2, 2, 4, 6], [3, 2, 2, 5, 6],
            [3, 2, 3, 2, 6], [3, 2, 3, 3, 6], [3, 2, 3, 4, 6], [3, 2, 3, 5, 6],
            [3, 2, 4, 2, 6], [3, 2, 4, 3, 6], [3, 2, 4, 4, 6], [3, 2, 4, 5, 6],
            [3, 3, 2, 2, 6], [3, 3, 2, 3, 6], [3, 3, 2, 4, 6], [3, 3, 2, 5, 6],
            [3, 3, 3, 2, 6], [3, 3, 3, 3, 6], [3, 3, 3, 4, 6], [3, 3, 3, 5, 6],
            [3, 3, 4, 2, 6], [3, 3, 4, 3, 6], [3, 3, 4, 4, 6], [3, 3, 4, 5, 6],
        ]
        self.seq = self.seqs[i]

    def tandem(self, x: ArrayLike) -> float:
        return _safe_tandem_eval(_tandem_map[self.fun_c], x, self.seq)


class Tandem_minlp:
    """see https://www.esa.int/gsp/ACT/projects/gtop/tandem/"""

    def __init__(self, constrained: bool = True) -> None:
        self.name = "Tandem minlp " if constrained else "Tandem unconstrained minlp "
        self.fun_c = "tandemC" if constrained else "tandemCu"
        self.fun = self.tandem_minlp
        self.bounds = Bounds(
            [5475.0, 2.5, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 0.01, 0.01, 0.01,
             0.01, 1.05, 1.05, 1.05, -math.pi, -math.pi, -math.pi,
             1.51, 1.51, 1.51],
            [9132.0, 4.9, 1.0, 1.0, 2500.0, 2500.0, 2500.0, 2500.0, 0.99, 0.99,
             0.99, 0.99, 10.0, 10.0, 10.0, math.pi, math.pi, math.pi,
             3.49, 4.49, 5.49],
        )

    def tandem_minlp(self, xs: ArrayLike) -> float:
        x = list(xs[:-3])
        seq = [3] + [int(round(xi)) for xi in xs[-3:]] + [6]
        return _safe_tandem_eval(_tandem_map[self.fun_c], x, seq)


class Cassini1multi:
    """see https://www.esa.int/gsp/ACT/projects/gtop/cassini1/"""

    def __init__(self, weights: Sequence[float] = (1, 0, 0, 0),
                 planets: Sequence[int] = (2, 2, 3, 5)) -> None:
        Astrofun.__init__(
            self, "Cassini1minlp", "cassini1minlpC",
            [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
            [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
        )
        self.fun = self.cassini1
        self.weights = weights
        self.planets = planets
        self.mfun = lambda x: cassini1multi(_concat_planets(x, [2, 2, 3, 5]))

    def cassini1(self, x: ArrayLike) -> float:
        values = cassini1multi(_concat_planets(x, self.planets))
        return (
            self.weights[0] * values[0]
            + self.weights[1] * values[1]
            + self.weights[2] * values[2]
            + self.weights[3] * values[3]
        )


class Cassini1minlp:
    """see https://www.esa.int/gsp/ACT/projects/gtop/cassini1/"""

    def __init__(self, planets: Sequence[int] = (2, 2, 3, 5)) -> None:
        Astrofun.__init__(
            self, "Cassini1", "cassini1C",
            [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
            [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
        )
        self.fun = self.cassini1
        self.planets = planets

    def cassini1(self, x: ArrayLike) -> float:
        return cassini1minlp(_concat_planets(x, self.planets))


def cassini1minlp(x: Sequence[float]) -> float:
    dv, _ = _safe_cassini1_minlp_eval(x)
    return dv


def cassini1multi(x: Sequence[float]) -> list[float]:
    dv, launch_dv = _safe_cassini1_minlp_eval(x)
    tof = float(sum(x[1:6]))
    launch_time = float(x[0])
    return [dv, launch_dv, tof, launch_time]


def cassini2multi(x: Sequence[float]) -> list[float]:
    try:
        dv = float(gtop_cassini2_minlp(_as_float_array(x)))
        if not math.isfinite(dv):
            dv = PenaltyValue
    except Exception:
        dv = PenaltyValue
    tof = float(sum(x[4:9]))
    launch_time = float(x[0])
    return [dv, tof, launch_time]


class python_fun:

    def __init__(self, cfun: str, bounds: Bounds) -> None:
        self.cfun = cfun
        self.bounds = bounds

    def __call__(self, x: ArrayLike) -> float:
        return _safe_scalar_eval(astro_map[self.cfun], x)
