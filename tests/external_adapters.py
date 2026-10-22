"""Simulator adapters for test_external_simulator.py, in a module of their
own so that spawned worker processes can import them."""
import sys

from external_simulator import ExternalObjective

STANDARD = [sys.executable, "-c", "import json; json.dump({'value': 2.0}, open('output.json', 'w'))"]
FLAKY = [sys.executable, "-c",
         "import json, sys; d = json.load(open('input.json'));"
         "sys.exit(1) if d['seed'] == 2 else json.dump({'value': 1.0}, open('output.json', 'w'))"]
CUSTOM_SIMULATOR = """\
x, seed = open("custom.in").read().split()
open("custom.out", "w").write(str(float(x) ** 2))
"""


class CustomIO(ExternalObjective):
    """Own file names and an extra constructor argument with state."""

    def __init__(self, command, scale, **kw):
        super().__init__(command, **kw)
        self.scale = scale

    def write_input(self, directory, x, seed):
        (directory / "custom.in").write_text(f"{x[0]} {seed}")

    def read_value(self, directory):
        return self.scale * float((directory / "custom.out").read_text())


class Transformed(ExternalObjective):
    """The standard files, a transformed objective."""

    def read_value(self, directory):
        return 100 * super().read_value(directory)


def nan_aggregate(values):
    return float("nan")
