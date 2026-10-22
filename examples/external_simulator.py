"""Optimize an external simulator program with fcmaes.

Many objectives are programs: a CFD or FEM solver, a circuit simulator, a
legacy executable. This example wraps such a program as an fcmaes objective.

Working directories. Each simulator run gets a fresh working directory. It
is removed afterwards, or kept on failure for inspection
(``keep_failed=True``), so parallel evaluations never share files. The
program reads ``input.json`` (the decision vector ``x`` and a ``seed``) and
writes ``output.json`` with ``{"value": <float>}``. Adapt ``write_input`` and
``read_value`` to the real program's formats.

Process trees. Each run starts in a process group of its own (a new session
on POSIX). When the run ends, times out or is interrupted, the whole group is
stopped: SIGTERM first, SIGKILL after ``grace`` seconds. This includes
solver processes the program started and helpers it left running. Children
that leave the group (``setsid``, a daemon double fork) escape that
containment; for those, run the whole optimization inside a container or a
systemd scope. On Windows the run starts in a new process group and
``taskkill /T`` stops its tree while the program still runs; children of a
program that has already exited cannot be found that way. If the Python
process running the objective is itself killed, its current run is not
cleaned up.

Failures and accounting. A timeout, a nonzero exit status, a missing or
malformed output, or a non-finite value makes the evaluation fail.
``__call__`` then returns ``failure_value``, a finite value worse than every
real result, because fcmaes minimizes. Every evaluation appends one record to
a ledger directory shared by all worker processes (one file per process, so
it works with fork and spawn). ``summary()`` in the parent counts successes
and failures by kind. ``evaluate(x)`` returns an ``Evaluation`` that keeps
"the simulation succeeded" separate from the ranking value, and
``check_result`` refuses a campaign with no successful evaluation or a best
point whose fresh recheck fails.

Seeds. Without ``seeds``, each evaluation runs the program once with a seed
derived from ``x``. The same point always gets the same seed, so repeated
evaluations replay exactly. This is *not* common random numbers: different
points get unrelated seeds. With ``seeds=[...]``, every candidate runs once
per seed and the values are aggregated (mean by default). All candidates
then see the same random draws, which makes their comparison fair for a
stochastic simulator. ``replication_seeds`` returns disjoint training and
holdout seed sets. Optimize on the training seeds, then validate the result
with ``with_seeds(holdout)``, a copy of the objective (a subclass keeps its
type, its overridden methods and its state) with other seeds and a ledger of
its own. The ledger counts attempted simulator runs as well as evaluations;
an aggregate that is not finite fails the evaluation as
``invalid_aggregate``, separate from simulator failures.

Threads. The program's thread pools are set to one thread
(``OMP_NUM_THREADS`` and friends are overwritten, not inherited), because
fcmaes owns the parallelism: one evaluation per worker process. Give the
simulator more threads explicitly, ``env={"OMP_NUM_THREADS": "4"}``, together
with fewer workers, and keep workers times threads within ``usable_cpus()``,
which honours the affinity mask and the cgroup CPU quota. ``topology()``
reports the layout to record with the results.

The objective object is picklable, so ``fcmaes.retry.minimize`` can run it in
``workers`` processes. Run this file to optimize a small demo simulator, a
Python script written to a temporary directory::

    python examples/external_simulator.py
"""

import copy
import hashlib
import json
import math
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds

# Thread-pool variables of common native libraries.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def usable_cpus(cgroup_file=Path("/proc/self/cgroup"), mount=Path("/sys/fs/cgroup")) -> int:
    """CPUs this process may use.

    ``os.cpu_count()`` and ``multiprocessing.cpu_count()`` count the machine.
    This also honours the affinity mask (``taskset``, cpusets) and, on Linux,
    the CPU quota of the process's cgroup (v2 ``cpu.max`` or v1
    ``cpu.cfs_quota_us``; containers, systemd ``CPUQuota``). It is an upper
    bound for ``workers`` times the threads of each evaluation; memory per
    evaluation can lower it further.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cpus = os.cpu_count() or 1
    quota = cgroup_cpu_quota(cgroup_file, mount)
    if quota is not None:
        cpus = min(cpus, max(1, math.floor(quota)))
    return max(1, cpus)


def cgroup_cpu_quota(cgroup_file=Path("/proc/self/cgroup"), mount=Path("/sys/fs/cgroup")):
    """The CPU quota of this process's cgroup in CPUs, or None without one.

    The most restrictive quota on the path from the process's cgroup to the
    root applies. A cgroup path that is not visible under ``mount`` (a cgroup
    namespace) is read at the mount root.
    """
    try:
        lines = Path(cgroup_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    best = None

    def consider(cpus):
        nonlocal best
        if cpus is not None and cpus > 0:
            best = cpus if best is None else min(best, cpus)

    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and controllers == "":  # cgroup v2
            for directory in _ancestors(Path(mount), path):
                consider(_read_v2(directory / "cpu.max"))
        elif "cpu" in controllers.split(","):  # cgroup v1
            for base in (Path(mount) / controllers, Path(mount) / "cpu"):
                for directory in _ancestors(base, path):
                    consider(_read_v1(directory))
    return best


def _ancestors(base: Path, path: str):
    """``base/path`` and its parents up to ``base`` (``base`` alone if the
    path does not exist there)."""
    directory = base / path.lstrip("/")
    if not directory.exists():
        directory = base
    while True:
        yield directory
        if directory == base or directory.parent == directory:
            return
        directory = directory.parent


def _read_v2(file: Path):
    try:
        quota, period = file.read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def _read_v1(directory: Path):
    try:
        quota = int((directory / "cpu.cfs_quota_us").read_text())
        period = int((directory / "cpu.cfs_period_us").read_text())
        return None if quota <= 0 or period <= 0 else quota / period
    except (OSError, ValueError):
        return None


def seed_of(x) -> int:
    """A seed derived from the point: identical for identical points,
    unrelated for different points (replay, not common random numbers)."""
    data = np.ascontiguousarray(x, dtype=np.float64).tobytes()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


def replication_seeds(training: int, holdout: int, seed: int = 0):
    """Disjoint training and holdout seed lists, reproducible from ``seed``."""
    drawn = random.Random(seed).sample(range(1, 2**31), training + holdout)
    return drawn[:training], drawn[training:]


@dataclass
class Evaluation:
    """One evaluation: whether every simulator run succeeded, the value used
    for ranking (``failure_value`` on failure) and its raw values."""
    ok: bool
    value: float
    # "ok"; a simulator run's "timeout", "exit", "invalid_output" or "error";
    # or "invalid_aggregate" when every run succeeded but the aggregate of
    # their values is not a finite number.
    status: str
    seeds: list = field(default_factory=list)
    values: list = field(default_factory=list)
    seconds: float = 0.0
    message: str = ""
    runs: int = 0  # simulator runs attempted, successful or not


class NoValidResult(RuntimeError):
    """A campaign without a usable answer."""


class ExternalObjective:
    """Runs ``command`` once per seed and evaluation; see the module docstring."""

    def __init__(self, command, timeout=60.0, failure_value=1e10, work_root=None,
                 keep_failed=False, env=None, seeds=None, aggregate=statistics.fmean,
                 grace=2.0, ledger_dir=None):
        self.command = list(command)
        self.timeout = timeout
        self.failure_value = failure_value
        self.work_root = work_root
        self.keep_failed = keep_failed
        self.env = dict(env or {})
        self.seeds = None if seeds is None else [int(s) for s in seeds]
        self.aggregate = aggregate
        self.grace = grace
        # Created here, in the parent, so every worker process (fork or
        # spawn) appends to the same directory. It stays until removed
        # (shutil.rmtree(objective.ledger_dir)); put it under work_root to
        # keep it with the campaign.
        self.ledger_dir = str(ledger_dir or tempfile.mkdtemp(prefix="fcmaes-ledger-", dir=work_root))
        Path(self.ledger_dir).mkdir(parents=True, exist_ok=True)

    def with_seeds(self, seeds, ledger_dir=None):
        """The same objective on other seeds (holdout validation), with a
        ledger of its own.

        A shallow copy of this object, so a subclass keeps its type, its
        overridden ``write_input``/``read_value`` and its attributes (shared,
        not copied): training and holdout evaluate the same objective and
        differ only in the seeds and the ledger."""
        clone = copy.copy(self)
        clone.seeds = None if seeds is None else [int(s) for s in seeds]
        clone.ledger_dir = str(ledger_dir or tempfile.mkdtemp(prefix="fcmaes-ledger-", dir=self.work_root))
        Path(clone.ledger_dir).mkdir(parents=True, exist_ok=True)
        return clone

    def thread_env(self) -> dict:
        """The thread-pool variables every simulator run gets: 1, unless set
        explicitly in ``env``."""
        values = {name: "1" for name in THREAD_VARIABLES}
        values.update({k: v for k, v in self.env.items() if k in THREAD_VARIABLES})
        return values

    def topology(self, workers: int) -> dict:
        """The CPU layout to record with the results."""
        threads = max(int(v) for v in self.thread_env().values() if str(v).isdigit())
        return {"workers": workers, "threads_per_evaluation": threads,
                "cpus_needed": workers * threads, "usable_cpus": usable_cpus(),
                "thread_env": self.thread_env(), "seeds_per_evaluation": len(self.seeds or [0])}

    def write_input(self, directory: Path, x, seed: int) -> None:
        payload = {"x": [float(v) for v in x], "seed": seed}
        (directory / "input.json").write_text(json.dumps(payload), encoding="utf-8")

    def read_value(self, directory: Path) -> float:
        return float(json.loads((directory / "output.json").read_text(encoding="utf-8"))["value"])

    def evaluate(self, x) -> Evaluation:
        start = time.monotonic()
        seeds = self.seeds if self.seeds is not None else [seed_of(x)]
        values, status, message, runs = [], "ok", "", 0
        for seed in seeds:
            runs += 1
            status, value, message = self._run_once(x, seed)
            if status != "ok":
                break
            values.append(value)
        ok = status == "ok"
        value = self.failure_value
        if ok:
            # Every run succeeded; a failing aggregate is no simulator failure.
            try:
                value = float(self.aggregate(values))
            except (ArithmeticError, TypeError, ValueError) as e:
                value, message = math.nan, f"{type(e).__name__}: {e}"
            if not math.isfinite(value):
                ok, status, value = False, "invalid_aggregate", self.failure_value
                message = message or "aggregate not finite"
        evaluation = Evaluation(ok, value, status, list(seeds), values, time.monotonic() - start,
                                message, runs)
        self._record(evaluation)
        return evaluation

    def __call__(self, x) -> float:
        return self.evaluate(x).value

    def summary(self) -> dict:
        """Counts over every evaluation of every process that used this ledger."""
        counts = {"evaluations": 0, "ok": 0, "failed": 0, "timeout": 0, "exit": 0,
                  "invalid_output": 0, "error": 0, "invalid_aggregate": 0,
                  "simulator_runs": 0}  # attempted runs, successful or not
        for file in Path(self.ledger_dir).glob("*.jsonl"):
            # Complete lines only: a process may be appending right now.
            for line in file.read_text(encoding="utf-8").split("\n")[:-1]:
                record = json.loads(line)
                counts["evaluations"] += 1
                counts["simulator_runs"] += record["runs"]
                counts["ok" if record["ok"] else "failed"] += 1
                if not record["ok"]:
                    counts[record["status"]] += 1
        return counts

    def _record(self, e: Evaluation) -> None:
        line = json.dumps({"ok": e.ok, "status": e.status, "value": e.value, "seeds": e.seeds,
                           "runs": e.runs, "seconds": e.seconds})
        with open(Path(self.ledger_dir) / f"{os.getpid()}.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _run_once(self, x, seed):
        directory = Path(tempfile.mkdtemp(prefix="eval-", dir=self.work_root))
        env = os.environ.copy()
        env.update(self.env)
        env.update(self.thread_env())  # overwrite inherited pool sizes
        status, value, message, phase = "error", self.failure_value, "", "start"
        try:
            self.write_input(directory, x, seed)
            code = run_contained(self.command, directory, env, self.timeout, self.grace)
            if code != 0:
                status, message = "exit", f"exit status {code}"
            else:
                phase = "read"
                value = self.read_value(directory)
                if math.isfinite(value):
                    status = "ok"
                else:
                    status, value, message = "invalid_output", self.failure_value, "not finite"
        except subprocess.TimeoutExpired:
            status, message = "timeout", f"more than {self.timeout} s"
        except (OSError, ValueError, KeyError, TypeError) as e:
            # A missing or malformed output, or a failure to start the program.
            status = "invalid_output" if phase == "read" else "error"
            message = f"{type(e).__name__}: {e}"
        if status != "ok":
            value = self.failure_value
            if self.keep_failed:
                return status, value, message
        shutil.rmtree(directory, ignore_errors=True)
        return status, value, message


def run_contained(command, cwd, env, timeout, grace=2.0) -> int:
    """Runs ``command`` in a process group of its own and stops the whole group
    when it exits, times out (``TimeoutExpired`` is raised after the group is
    stopped) or is interrupted. Returns the exit status."""
    if os.name == "posix":
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
    else:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        return proc.wait(timeout=timeout)
    finally:
        _stop_tree(proc, grace)


def _stop_tree(proc, grace):
    if os.name != "posix":
        if proc.poll() is None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait()
        return
    group = proc.pid  # the session leader's pid is the group id
    if not _group_alive(group, proc):
        return
    with _ignore(ProcessLookupError, PermissionError):
        os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _group_alive(group, proc):
            return
        time.sleep(0.02)
    with _ignore(ProcessLookupError, PermissionError):
        os.killpg(group, signal.SIGKILL)
    proc.wait()


def _group_alive(group, proc) -> bool:
    proc.poll()  # reap the leader, so its zombie does not keep the group alive
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # members we may not signal still exist
        return True


class _ignore:
    def __init__(self, *errors):
        self.errors = errors

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        return kind is not None and issubclass(kind, self.errors)


def check_result(objective: ExternalObjective, x) -> Evaluation:
    """Refuses a campaign without a usable answer: no successful evaluation,
    or a best point whose fresh recheck fails. Returns the recheck."""
    counts = objective.summary()
    if counts["ok"] == 0:
        raise NoValidResult(f"no successful evaluation: {counts}")
    recheck = objective.evaluate(x)
    if not recheck.ok:
        raise NoValidResult(f"the best point fails its recheck ({recheck.status}: {recheck.message})")
    return recheck


# A demo "simulator": a separate program that reads input.json and writes
# output.json. It fails for x[0] > 4.5, as real simulators fail in parts of
# the search space, and adds seed-dependent noise.
DEMO_SIMULATOR = """\
import json, random, sys
data = json.load(open("input.json"))
x = data["x"]
if x[0] > 4.5:
    sys.exit("solver diverged")
noise = random.Random(data["seed"]).gauss(0.0, 0.01)
json.dump({"value": sum((v - 1.0) ** 2 for v in x) + noise}, open("output.json", "w"))
"""


def demo(workers=None, max_evaluations=400):
    from fcmaes import retry
    from fcmaes.optimizer import Bite_cpp

    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "simulator.py"
        program.write_text(DEMO_SIMULATOR, encoding="utf-8")
        training, holdout = replication_seeds(3, 3)
        objective = ExternalObjective([sys.executable, str(program)], timeout=30.0,
                                      seeds=training, work_root=tmp)
        bounds = Bounds([-5.0] * 3, [5.0] * 3)
        workers = workers or max(1, min(usable_cpus(), 8))
        print("topology", objective.topology(workers))
        result = retry.minimize(objective, bounds, num_retries=workers, workers=workers,
                                optimizer=Bite_cpp(max_evaluations // workers + 1))
        recheck = check_result(objective, result.x)
        validation = objective.with_seeds(holdout).evaluate(result.x)
        print("evaluations", objective.summary())
        print(f"best value {result.fun:.3g} at {np.round(result.x, 3)}, "
              f"recheck {recheck.value:.3g}, holdout {validation.value:.3g} ({validation.status})")
        return result, recheck, validation


if __name__ == "__main__":
    demo()
