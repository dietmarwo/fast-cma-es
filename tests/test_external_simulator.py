"""examples/external_simulator.py: one working directory per evaluation,
failures and timeouts as failure values, reproducible seeds, picklable."""
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds


def load_example():
    path = Path(__file__).parents[1] / "examples/external_simulator.py"
    spec = importlib.util.spec_from_file_location("external_simulator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["external_simulator"] = module  # so pickle finds the class
    spec.loader.exec_module(module)
    return module


SIMULATOR = """\
import json, os, sys, time
data = json.load(open("input.json"))
x = data["x"]
if x[0] > 4.5:
    sys.exit("diverged")
if x[1] > 4.5:
    time.sleep(30)
if x[2] > 4.5:
    sys.exit(0)  # no output.json
json.dump({"value": sum((v - 1.0) ** 2 for v in x), "seed": data["seed"],
           "threads": os.environ["OMP_NUM_THREADS"]}, open("output.json", "w"))
"""


def make(tmp_path, **kw):
    ex = load_example()
    program = tmp_path / "sim.py"
    program.write_text(SIMULATOR, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    return ex, ex.ExternalObjective([sys.executable, str(program)], work_root=work, **kw), work


def test_values_failures_timeouts_and_cleanup(tmp_path):
    ex, f, work = make(tmp_path, timeout=2.0, failure_value=1e9)
    assert f([1.0, 2.0, 3.0]) == 5.0
    assert f([4.9, 0.0, 0.0]) == 1e9      # nonzero exit
    assert f([0.0, 4.9, 0.0]) == 1e9      # timeout
    assert f([0.0, 0.0, 4.9]) == 1e9      # no output
    assert f.failures == 3
    assert list(work.iterdir()) == []      # every working directory removed


def test_keep_failed_thread_variables_and_seeds(tmp_path, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    ex, f, work = make(tmp_path, keep_failed=True)
    f([4.9, 0.0, 0.0])
    kept = list(work.iterdir())
    assert len(kept) == 1 and (kept[0] / "input.json").exists()

    class Threads(ex.ExternalObjective):
        def read_value(self, directory):
            return float(json.loads((directory / "output.json").read_text())["threads"])

    assert Threads(f.command, work_root=work)([0.0, 0.0, 0.0]) == 1.0   # one thread by default
    assert Threads(f.command, work_root=work, env={"OMP_NUM_THREADS": "3"})([0.0, 0.0, 0.0]) == 3.0
    assert ex.seed_of([0.5, 1.0]) == ex.seed_of(np.array([0.5, 1.0])) != ex.seed_of([0.5, 1.0000001])


def test_picklable_and_optimizes(tmp_path):
    ex, f, work = make(tmp_path)
    clone = pickle.loads(pickle.dumps(f))
    assert clone([1.0, 1.0, 1.0]) == 0.0
    from fcmaes import retry
    from fcmaes.optimizer import Bite_cpp
    result = retry.minimize(f, Bounds([-5.0] * 3, [5.0] * 3), num_retries=1,
                            workers=1, optimizer=Bite_cpp(150))
    assert result.fun < 0.5
    assert ex.usable_cpus() >= 1
