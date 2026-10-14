"""Usage: python tests/check_pareto_cross_language.py /path/to/pareto_trace

Requires the integrated fcmaes wheel. Rust and C++ replay identical CSV rows.
Every prefix checks acceptance/size/HV; complete payload fronts are checked at
each stream boundary. Run test_pareto.py for full C++ prefix-front checks.
"""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from fcmaes.pareto import ParetoArchive2D


def main():
    fixture = Path(__file__).parent / "data/pareto_archive_2d.csv"
    replay = subprocess.run([sys.argv[1]], input=fixture.read_bytes(),
                            stdout=subprocess.PIPE, check=True).stdout.decode()
    prefixes, rust_fronts = {}, {}
    for row in csv.reader(io.StringIO(replay)):
        if row[0] == "U":
            prefixes[(row[1], int(row[2]))] = (bool(int(row[3])), float(row[4]), int(row[5]))
        else:
            rust_fronts.setdefault(row[1], []).append((float(row[2]), float(row[3]), int(row[4])))
    groups = {}
    for row in csv.DictReader(io.StringIO(fixture.read_text())):
        groups.setdefault(row["case"], []).append((float(row["a"]), float(row["b"]), int(row["id"])))
    checked, max_error = 0, 0.0
    for case, rows in groups.items():
        archive = ParetoArchive2D([-2, -2], [2, 2], payload_dim=1)
        front, points = [], []
        for a, b, id_ in rows:
            expected = not any(q[0] <= a and q[1] <= b for q in front)
            if expected:
                front = [q for q in front if not (a <= q[0] and b <= q[1])]
                front.append((a, b, id_))
                front.sort()
            accepted = archive.add([a, b], [id_])
            rust_accepted, rust_hv, rust_size = prefixes[(case, id_)]
            points.append((a, b))
            sorted_rows = np.asarray(sorted(points), dtype=np.longdouble)
            best = np.minimum.accumulate(sorted_rows[:, 1])
            widths = np.diff(np.r_[sorted_rows[:, 0], np.longdouble(2)])
            oracle = float(np.sum(widths * (np.longdouble(2) - best)))
            assert accepted == rust_accepted == expected, (case, id_, "acceptance")
            assert len(archive) == rust_size == len(front), (case, id_, "size")
            error = max(abs(oracle - rust_hv), abs(oracle - archive.hypervolume()))
            assert error <= 1e-11, (case, id_, "HV", error)
            max_error = max(max_error, error)
            checked += 1
        xs, ys = archive.front()
        cpp_front = [(a, b, int(x[0])) for (a, b), x in zip(ys, xs)]
        assert cpp_front == rust_fronts[case] == front, (case, "complete front")
    print(json.dumps({"prefixes": checked, "streams": len(groups),
                      "max_hv_absolute_error": max_error, "complete_fronts_equal": True}))


if __name__ == "__main__":
    main()
