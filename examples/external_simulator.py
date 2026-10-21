"""Optimize an external simulator program with fcmaes.

Many objectives are programs: a CFD or FEM solver, a circuit simulator, a
legacy executable. This example wraps such a program as an fcmaes objective:

- Each evaluation runs the program in a fresh working directory, which is
  removed afterwards (or kept on failure, for inspection), so parallel
  evaluations never share files.
- The program reads ``input.json`` (the decision vector ``x`` and a ``seed``)
  and writes ``output.json`` with ``{"value": <float>}``. Adapt
  ``write_input`` and ``read_value`` to the real program's formats.
- A timeout, a nonzero exit status, a missing or malformed output, or a
  non-finite value counts as a failed evaluation. It returns
  ``failure_value``, a finite value worse than every real result (fcmaes
  minimizes), and is counted, so the run can report how many evaluations
  failed.
- The seed is derived from ``x``, so the same point always gets the same seed:
  repeated evaluations are reproducible, and with a stochastic simulator the
  comparison of two points uses the same random numbers.
- The program's own thread pools are limited to one thread (``OMP_NUM_THREADS``
  and friends), because fcmaes owns the parallelism: one evaluation per
  worker process. Give the simulator more threads only together with fewer
  fcmaes workers, and keep their product within ``usable_cpus()``.

The objective object is picklable, so ``fcmaes.retry.minimize`` can run it in
``workers`` processes. Run this file to optimize a small demo simulator, a
Python script written to a temporary directory::

    python examples/external_simulator.py
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds

# Thread-pool variables of common native libraries; each simulator process
# gets one thread unless the caller sets them.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def usable_cpus() -> int:
    """CPUs this process may use.

    ``os.cpu_count()`` and ``multiprocessing.cpu_count()`` count the machine.
    This also honours the affinity mask (``taskset``, cpusets) and, on Linux,
    a cgroup v2 CPU quota (containers, systemd ``CPUQuota``). It is a
    starting point for ``workers``; memory per evaluation can lower it further.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cpus = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus, max(1, math.floor(quota)))
    return max(1, cpus)


def _cgroup_cpu_quota():
    """CPU quota of this process's cgroup v2 in CPUs, or None."""
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):  # the unified (v2) hierarchy
            group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            best = None
            # The most restrictive quota on the path to the root applies.
            for directory in (group, *group.parents):
                try:
                    quota, period = (directory / "cpu.max").read_text().split()
                except (OSError, ValueError):
                    pass
                else:
                    if quota != "max":
                        cpus = int(quota) / int(period)
                        best = cpus if best is None else min(best, cpus)
                if directory == Path("/sys/fs/cgroup"):
                    break
            return best
    return None


def seed_of(x) -> int:
    """A seed derived from the point, identical for identical points."""
    data = np.ascontiguousarray(x, dtype=np.float64).tobytes()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


class ExternalObjective:
    """Runs ``command`` once per evaluation; see the module docstring."""

    def __init__(self, command, timeout=60.0, failure_value=1e10,
                 work_root=None, keep_failed=False, env=None):
        self.command = list(command)
        self.timeout = timeout
        self.failure_value = failure_value
        self.work_root = work_root
        self.keep_failed = keep_failed
        self.env = dict(env or {})
        self.failures = 0  # counts the failures of this process's copy

    def write_input(self, directory: Path, x, seed: int) -> None:
        payload = {"x": [float(v) for v in x], "seed": seed}
        (directory / "input.json").write_text(json.dumps(payload), encoding="utf-8")

    def read_value(self, directory: Path) -> float:
        return float(json.loads((directory / "output.json").read_text(encoding="utf-8"))["value"])

    def __call__(self, x) -> float:
        directory = Path(tempfile.mkdtemp(prefix="eval-", dir=self.work_root))
        env = os.environ.copy()
        for name in THREAD_VARIABLES:
            env.setdefault(name, "1")
        env.update(self.env)
        ok = False
        try:
            self.write_input(directory, x, seed_of(x))
            subprocess.run(self.command, cwd=directory, env=env, timeout=self.timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            value = self.read_value(directory)
            ok = math.isfinite(value)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
                OSError, ValueError, KeyError, TypeError):
            # A timeout kills the program (subprocess.run); a simulator that
            # starts children of its own should run them in its process group.
            pass
        if not ok:
            self.failures += 1
            if self.keep_failed:
                return self.failure_value
        shutil.rmtree(directory, ignore_errors=True)
        return value if ok else self.failure_value


# A demo "simulator": a separate program that reads input.json and writes
# output.json. It fails for x[0] > 4.5, as real simulators fail in parts of
# the search space.
DEMO_SIMULATOR = """\
import json, sys
x = json.load(open("input.json"))["x"]
if x[0] > 4.5:
    sys.exit("solver diverged")
value = sum((v - 1.0) ** 2 for v in x)
json.dump({"value": value}, open("output.json", "w"))
"""


def demo(workers=None, max_evaluations=400):
    from fcmaes import retry
    from fcmaes.optimizer import Bite_cpp

    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "simulator.py"
        program.write_text(DEMO_SIMULATOR, encoding="utf-8")
        objective = ExternalObjective([sys.executable, str(program)], timeout=30.0)
        bounds = Bounds([-5.0] * 3, [5.0] * 3)
        workers = workers or usable_cpus()
        result = retry.minimize(objective, bounds, num_retries=workers, workers=workers,
                                optimizer=Bite_cpp(max_evaluations // workers + 1))
        print(f"usable CPUs {usable_cpus()}, workers {workers}")
        print(f"best value {result.fun:.3g} at {np.round(result.x, 3)}")
        # Recheck the reported best point with a fresh evaluation.
        recheck = objective(result.x)
        print(f"recheck {recheck:.3g}")
        return result, recheck


if __name__ == "__main__":
    demo()
