"""examples/external_simulator.py: one working directory per run, contained
process trees, failure accounting across processes, seed modes, thread
pools, usable CPUs, picklable."""
import importlib
import json
import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import Bounds


def load_example():
    # Import through sys.path, not from a file spec: with the spawn start
    # method (macOS, Windows) worker processes unpickle the objective and
    # must import its module; they inherit the parent's sys.path.
    examples = str(Path(__file__).parents[1] / "examples")
    if examples not in sys.path:
        sys.path.insert(0, examples)
    return importlib.import_module("external_simulator")


SIMULATOR = """\
import json, os, random, sys, time
data = json.load(open("input.json"))
x = data["x"]
if x[0] > 4.5:
    sys.exit("diverged")
if x[1] > 4.5:
    time.sleep(30)
if x[2] > 4.5:
    sys.exit(0)  # no output.json
noise = random.Random(data["seed"]).gauss(0.0, 1.0) if len(x) > 3 and x[3] > 0 else 0.0
json.dump({"value": sum((v - 1.0) ** 2 for v in x[:3]) + noise,
           "threads": os.environ["OMP_NUM_THREADS"], "seed": data["seed"]}, open("output.json", "w"))
"""


def make(tmp_path, **kw):
    ex = load_example()
    program = tmp_path / "sim.py"
    program.write_text(SIMULATOR, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    ledger = tmp_path / f"ledger-{len(list(tmp_path.glob('ledger-*')))}"
    return ex, ex.ExternalObjective([sys.executable, str(program)], work_root=work,
                                    ledger_dir=ledger, **kw), work


def test_values_failures_and_cleanup(tmp_path):
    ex, f, work = make(tmp_path, timeout=2.0, failure_value=1e9)
    assert f([1.0, 2.0, 3.0]) == 5.0
    assert f([4.9, 0.0, 0.0]) == 1e9      # nonzero exit
    assert f([0.0, 4.9, 0.0]) == 1e9      # timeout
    e = f.evaluate([0.0, 0.0, 4.9])       # no output
    assert (e.ok, e.status, e.value) == (False, "invalid_output", 1e9)
    s = f.summary()
    assert (s["evaluations"], s["ok"], s["failed"]) == (4, 1, 3)
    assert (s["exit"], s["timeout"], s["invalid_output"]) == (1, 1, 1)
    assert list(work.iterdir()) == []      # every working directory removed
    missing = ex.ExternalObjective([str(tmp_path / "no-such-program")], work_root=work,
                                   ledger_dir=tmp_path / "ledger-missing")
    assert missing.evaluate([1.0, 1.0, 1.0]).status == "error"


TREE = """\
import os, subprocess, sys, time
marker = sys.argv[1]
if len(sys.argv) > 2:  # child and grandchild: write the marker later
    if sys.argv[2] == "child":
        subprocess.Popen([sys.executable, __file__, marker + "-grandchild", "grandchild"])
    time.sleep(1.0)
    open(marker, "w").write("still running")
    sys.exit(0)
subprocess.Popen([sys.executable, __file__, marker + "-child", "child"])
if os.path.exists("input.json") and '"x": [9.0' in open("input.json").read():
    open("output.json", "w").write('{"value": 1.5}')  # exits at once, leaves helpers
    sys.exit(0)
time.sleep(30)
"""


@pytest.mark.skipif(os.name != "posix", reason="process groups (POSIX); Windows uses taskkill /T")
@pytest.mark.parametrize("case", ["timeout", "normal exit"])
def test_the_whole_process_tree_stops(tmp_path, case):
    ex = load_example()
    program = tmp_path / "tree.py"
    program.write_text(TREE, encoding="utf-8")
    marker = tmp_path / "marker"
    work = tmp_path / "work"
    work.mkdir()
    f = ex.ExternalObjective([sys.executable, str(program), str(marker)], timeout=0.5, grace=0.5,
                             work_root=work, keep_failed=True, ledger_dir=tmp_path / "ledger")
    started = time.monotonic()
    e = f.evaluate([9.0] if case == "normal exit" else [0.0])
    assert time.monotonic() - started < 3.0
    if case == "timeout":
        assert e.status == "timeout"
        kept = list(work.iterdir())  # the failed run's directory stays readable
        assert len(kept) == 1 and json.loads((kept[0] / "input.json").read_text())["x"] == [0.0]
    else:
        assert e.ok and e.value == 1.5 and list(work.iterdir()) == []
    time.sleep(1.5)  # longer than the descendants would sleep
    assert not list(tmp_path.glob("marker*")), "a descendant kept running"


def evaluate_in(objective, points, method):
    with multiprocessing.get_context(method).Pool(2) as pool:
        return pool.map(objective, points)


@pytest.mark.parametrize("method", ["spawn"] + (["fork"] if os.name == "posix" else []))
def test_counts_across_worker_processes(tmp_path, method):
    ex, f, work = make(tmp_path)
    points = [[1.0, 1.0, 1.0], [4.9, 0.0, 0.0], [2.0, 1.0, 1.0], [4.8, 0.0, 0.0]]
    values = evaluate_in(f, points, method)
    assert values == [0.0, 1e10, 1.0, 1e10]
    s = f.summary()  # the parent sees every process's evaluations
    assert (s["evaluations"], s["ok"], s["failed"], s["exit"]) == (4, 2, 2, 2)
    assert ex.check_result(f, [1.0, 1.0, 1.0]).ok
    failing = ex.ExternalObjective([sys.executable, "-c", "raise SystemExit(3)"], work_root=work,
                                   ledger_dir=tmp_path / "ledger-failing")
    evaluate_in(failing, points[:2], method)
    with pytest.raises(ex.NoValidResult):  # nothing succeeded: no answer
        ex.check_result(failing, points[0])
    with pytest.raises(ex.NoValidResult):  # the best point fails its recheck
        ex.check_result(f, [4.9, 0.0, 0.0])


def test_seed_modes(tmp_path):
    ex, replay, work = make(tmp_path)
    a, b = [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0, 1.0]  # x[3] > 0: noisy
    # Replay: the same point always gets the same seed, other points others.
    assert replay.evaluate(a).seeds == replay.evaluate(a).seeds != replay.evaluate(b).seeds
    assert replay(a) == replay(a)
    training, holdout = ex.replication_seeds(4, 3, seed=7)
    assert len(set(training) | set(holdout)) == 7 and (training, holdout) == ex.replication_seeds(4, 3, seed=7)
    _, crn, _ = make(tmp_path, seeds=training)
    ea, eb = crn.evaluate(a), crn.evaluate(b)
    # Common random numbers: both candidates see the same draws, so their
    # difference is the noise-free one (1.0 here).
    assert ea.seeds == eb.seeds == training and len(ea.values) == 4
    assert abs((eb.value - ea.value) - 1.0) < 1e-9
    assert crn.evaluate(a).value == ea.value
    assert crn.summary()["simulator_runs"] == 12
    validation = crn.with_seeds(holdout)
    assert validation.evaluate(a).seeds == holdout and validation.summary()["evaluations"] == 1


def test_thread_pools_and_topology(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")  # inherited: overwritten
    ex, f, work = make(tmp_path)

    class Threads(ex.ExternalObjective):
        def read_value(self, directory):
            return float(json.loads((directory / "output.json").read_text())["threads"])

    assert Threads(f.command, work_root=work)([0.0, 0.0, 0.0]) == 1.0
    explicit = Threads(f.command, work_root=work, env={"OMP_NUM_THREADS": "3"})
    assert explicit([0.0, 0.0, 0.0]) == 3.0
    t = explicit.topology(workers=2)
    assert (t["threads_per_evaluation"], t["cpus_needed"]) == (3, 6)
    assert t["usable_cpus"] == ex.usable_cpus()


def test_usable_cpus_honours_cgroup_quotas(tmp_path):
    ex = load_example()
    v2 = tmp_path / "v2"
    (v2 / "a" / "b").mkdir(parents=True)
    (v2 / "a" / "cpu.max").write_text("800000 100000\n")      # 8 CPUs
    (v2 / "a" / "b" / "cpu.max").write_text("max 100000\n")
    cgroup = tmp_path / "cgroup-v2"
    cgroup.write_text("0::/a/b\n")
    assert ex.cgroup_cpu_quota(cgroup, v2) == 8.0
    v1 = tmp_path / "v1"
    (v1 / "cpu,cpuacct" / "c").mkdir(parents=True)
    (v1 / "cpu,cpuacct" / "c" / "cpu.cfs_quota_us").write_text("250000\n")
    (v1 / "cpu,cpuacct" / "c" / "cpu.cfs_period_us").write_text("100000\n")
    cgroup1 = tmp_path / "cgroup-v1"
    cgroup1.write_text("4:cpu,cpuacct:/c\n3:memory:/c\n")
    assert ex.cgroup_cpu_quota(cgroup1, v1) == 2.5
    (v1 / "cpu,cpuacct" / "c" / "cpu.cfs_quota_us").write_text("-1\n")
    assert ex.cgroup_cpu_quota(cgroup1, v1) is None
    assert ex.usable_cpus(cgroup, v2) <= 8 and ex.usable_cpus(tmp_path / "none", v2) >= 1


def test_picklable_and_optimizes(tmp_path):
    ex, f, work = make(tmp_path)
    clone = pickle.loads(pickle.dumps(f))
    assert clone([1.0, 1.0, 1.0]) == 0.0
    from fcmaes import retry
    from fcmaes.optimizer import Bite_cpp
    result = retry.minimize(f, Bounds([-5.0] * 3, [5.0] * 3), num_retries=1,
                            workers=1, optimizer=Bite_cpp(150))
    assert result.fun < 0.5
    assert ex.check_result(f, result.x).ok
    assert f.summary()["evaluations"] >= 150
