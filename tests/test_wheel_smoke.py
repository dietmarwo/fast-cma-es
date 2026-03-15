import numpy as np
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds

import fcmaes
from fcmaes import bitecpp, cmaescpp, dacpp, decpp


def sphere(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.dot(x, x))


def fixed_rg(seed):
    return Generator(PCG64DXSM(seed))


def test_native_extension_imports():
    assert isinstance(fcmaes.__version__, str)
    assert fcmaes.__version__
    assert callable(bitecpp.optimize_bite)
    assert callable(dacpp.optimize_da)
    assert callable(decpp.optimize_de)
    assert callable(cmaescpp.optimize_acma)


def test_native_optimizers_smoke():
    bounds = Bounds([-5.0, -5.0], [5.0, 5.0])
    x0 = np.array([3.0, -4.0], dtype=np.float64)

    runs = [
        (
            "bite",
            1.0,
            bitecpp.minimize,
            dict(
                fun=sphere,
                bounds=bounds,
                x0=x0,
                max_evaluations=320,
                rg=fixed_rg(1),
            ),
        ),
        (
            "da",
            1.0,
            dacpp.minimize,
            dict(
                fun=sphere,
                bounds=bounds,
                x0=x0,
                max_evaluations=320,
                use_local_search=True,
                rg=fixed_rg(2),
            ),
        ),
        (
            "de",
            1.0,
            decpp.minimize,
            dict(
                fun=sphere,
                dim=2,
                bounds=bounds,
                x0=x0,
                popsize=16,
                max_evaluations=320,
                rg=fixed_rg(3),
            ),
        ),
        (
            "cma",
            1.0,
            cmaescpp.minimize,
            dict(
                fun=sphere,
                bounds=bounds,
                x0=x0,
                input_sigma=0.5,
                popsize=16,
                max_evaluations=320,
                workers=1,
                rg=fixed_rg(4),
            ),
        ),
    ]

    for name, limit, fn, kwargs in runs:
        result = fn(**kwargs)
        assert result.success, name
        assert np.isfinite(result.fun), name
        assert result.nfev > 0, name
        assert result.fun < limit, name
