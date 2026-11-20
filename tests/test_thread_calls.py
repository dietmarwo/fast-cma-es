"""Optimizer calls from several Python threads complete with correct results.

Each scenario runs in a subprocess with a timeout, so a deadlock fails the test
instead of stopping the test run. Single-threaded calls (``workers=1``) are
deterministic for a fixed seed: run in threads they must give the same results
as run one after the other. With ``workers > 1`` the order of the evaluations
depends on timing, so those scenarios check that each call reports a point
and value of its own objective.
"""

import json
import subprocess
import sys
import threading

import numpy as np
import pytest
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds

TIMEOUT = 120  # seconds; generous for a loaded CI host
DIM = 6
BOUNDS = Bounds([-2.0] * DIM, [2.0] * DIM)
EVALS = 3000


class Objective:
    """Shifted sphere that counts its evaluations."""

    def __init__(self, shift):
        self.shift = shift
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return float(np.sum((np.asarray(x) - self.shift) ** 2))


def run_cma(fun, seed, workers=1):
    from fcmaes import cmaescpp
    return cmaescpp.minimize(fun, BOUNDS, np.zeros(DIM), input_sigma=0.3,
                             max_evaluations=EVALS,
                             rg=Generator(PCG64DXSM(seed)), workers=workers)


def run_de(fun, seed, workers=1):
    from fcmaes import decpp
    return decpp.minimize(fun, DIM, BOUNDS, max_evaluations=EVALS,
                          rg=Generator(PCG64DXSM(seed)), workers=workers)


def run_bite(fun, seed, workers=1):
    from fcmaes import bitecpp
    return bitecpp.minimize(fun, BOUNDS, np.zeros(DIM), max_evaluations=EVALS,
                            rg=Generator(PCG64DXSM(seed)))


def run_crfmnes(fun, seed, workers=1):
    from fcmaes import crfmnescpp
    return crfmnescpp.minimize(fun, BOUNDS, np.zeros(DIM), input_sigma=0.3,
                               popsize=16, max_evaluations=EVALS,
                               rg=Generator(PCG64DXSM(seed)))


def run_pgpe(fun, seed, workers=1):
    from fcmaes import pgpecpp
    return pgpecpp.minimize(fun, BOUNDS, np.zeros(DIM), input_sigma=0.3,
                            popsize=16, max_evaluations=EVALS,
                            rg=Generator(PCG64DXSM(seed)))


def run_da(fun, seed, workers=1):
    from fcmaes import dacpp
    return dacpp.minimize(fun, BOUNDS, np.zeros(DIM), max_evaluations=EVALS,
                          rg=Generator(PCG64DXSM(seed)))


def run_cma_nested(fun, seed, workers=1):
    """CMA-ES whose objective runs a short inner CMA-ES call."""
    from fcmaes import cmaescpp

    def outer(x):
        inner = cmaescpp.minimize(Objective(0.1), BOUNDS, np.zeros(DIM),
                                  input_sigma=0.3, max_evaluations=40,
                                  rg=Generator(PCG64DXSM(seed)))
        return fun(x) + 0.0 * inner.fun

    return cmaescpp.minimize(outer, BOUNDS, np.zeros(DIM), input_sigma=0.3,
                             max_evaluations=200,
                             rg=Generator(PCG64DXSM(seed)))


RUNNERS = {
    "cma": run_cma, "de": run_de, "bite": run_bite, "crfmnes": run_crfmnes,
    "pgpe": run_pgpe, "da": run_da, "cma_nested": run_cma_nested,
}

# scenario: list of (optimizer, seed) calls, one thread per call
DETERMINISTIC = {
    "cma": [("cma", 1), ("cma", 2)],
    "de": [("de", 1), ("de", 2)],
    "cma_de": [("cma", 1), ("de", 2), ("cma", 3), ("de", 4)],
    "cma_nested": [("cma_nested", 1), ("cma", 2)],
    "others": [("bite", 1), ("crfmnes", 2), ("pgpe", 3), ("da", 4),
               ("bite", 5), ("crfmnes", 6), ("pgpe", 7), ("da", 8)],
}
PARALLEL = {
    "cma_workers": [("cma", 1), ("cma", 2)],
    "de_workers": [("de", 1), ("de", 2)],
    "cma_de_workers": [("cma", 1), ("de", 2)],
}


def call(name, seed, index, workers=1):
    fun = Objective(0.1 * (index + 1))
    r = RUNNERS[name](fun, seed, workers)
    return {"x": [float(v) for v in r.x], "fun": float(r.fun),
            "nfev": int(r.nfev), "evals": fun.count,
            "own_value": fun(r.x)}


def in_threads(calls, workers):
    results = [None] * len(calls)
    errors = []
    barrier = threading.Barrier(len(calls))

    def run(i, name, seed):
        try:
            barrier.wait()
            results[i] = call(name, seed, i, workers)
        except BaseException as e:  # reported to the parent process
            errors.append(repr(e))

    threads = [threading.Thread(target=run, args=(i, name, seed))
               for i, (name, seed) in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    return results


def scenario_main(kind, scenario):
    if kind == "deterministic":
        calls = DETERMINISTIC[scenario]
        sequential = [call(name, seed, i) for i, (name, seed) in enumerate(calls)]
        threaded = in_threads(calls, 1)
    else:
        calls = PARALLEL[scenario]
        sequential = None
        threaded = in_threads(calls, 2)
    print(json.dumps({"sequential": sequential, "threaded": threaded}))


def run_scenario(kind, scenario):
    try:
        proc = subprocess.run([sys.executable, __file__, kind, scenario],
                              capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{scenario}: threaded calls did not finish within "
                    f"{TIMEOUT} s (deadlock)")
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("scenario", sorted(DETERMINISTIC))
def test_threaded_calls_equal_sequential_calls(scenario):
    out = run_scenario("deterministic", scenario)
    assert out["threaded"] == out["sequential"]
    for r in out["threaded"]:
        assert r["fun"] == r["own_value"]


@pytest.mark.parametrize("scenario", sorted(PARALLEL))
def test_threaded_calls_with_workers(scenario):
    out = run_scenario("parallel", scenario)
    for r in out["threaded"]:
        # each call reports a point of its own objective
        assert r["fun"] == r["own_value"]
        assert r["evals"] > 0


if __name__ == "__main__":
    scenario_main(sys.argv[1], sys.argv[2])
