# Exact 2D Pareto archive

`fcmaes.pareto.ParetoArchive2D` keeps every unique nondominated pair of two
minimized objectives inside a fixed reference box, each with a payload (usually
the decision vector), and updates the dominated hypervolume with every
insertion. It is independent of the optimizers: use it to record everything an
optimizer evaluated, for example every generation of MODE.

```python
from fcmaes.pareto import ParetoArchive2D

archive = ParetoArchive2D([4, 4], payload_dim=2)
mask = archive.add_batch([[1, 3], [3, 1], [2, 3], [5, 0]],
                         [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5], [1.0, 0.0]])
assert mask.tolist() == [True, True, False, False]  # dominated, outside
assert archive.hypervolume() == 5.0
xs, ys = archive.front()  # payloads and objectives, sorted by ys[:, 0]
assert len(archive) == 2
```

## API

- `ParetoArchive2D(reference, payload_dim=0)`: `reference` is the finite
  hypervolume reference point. `payload_dim` is the payload length; with 0, no
  payload is stored and `x`/`xs` can be omitted.
- `add(y, x=None) -> bool`: insert one point; `True` if it was archived.
- `add_batch(ys, xs=None) -> ndarray[bool]`: insert the rows of `ys` (shape
  `(n, 2)`) with payloads `xs` (shape `(n, payload_dim)`) in order and return
  the mask of archived rows.
- `front() -> (xs, ys)`: copies of payloads and objectives, sorted by the first
  objective.
- `hypervolume()`, `len(archive)` / `size`, `reference`, `payload_dim`, `clear()`.

## Semantics

- **Membership.** A point is archived unless an archived point weakly
  dominates it. Archiving it removes the points it dominates. Equal objective
  pairs keep the first payload.
- **Reference box.** Points with an objective above the reference are not
  archived (`add` returns `False`), since they add nothing to the hypervolume.
  There is no lower bound: objectives may have any finite value.
- **Errors.** Non-finite objectives raise `ValueError`; areas beyond the float
  range raise `OverflowError`. The archive is unchanged after an error. In
  `add_batch`, shape errors are raised before any change; a failing row is
  named in the message and the rows before it stay applied.
- **Hypervolume.** Each insertion adds the exact area of the newly dominated
  region, summed with Kahan compensation. After millions of insertions the
  value agrees with a full recomputation to about 1e-16 relative error.
- **Capacity.** The archive is unbounded. In the worst case (every point
  nondominated) it keeps every point.
- **Threads.** Calls hold the GIL. An archive is a normal Python object; it is
  not shared between processes.

## Recording a MODE run

```python
import numpy as np
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds
from fcmaes.modecpp import MODE_C
from fcmaes.pareto import ParetoArchive2D

def zdt1(x):
    g = 1 + 9 * x[:, 1:].mean(axis=1)
    return np.column_stack((x[:, 0], g * (1 - np.sqrt(x[:, 0] / g))))

dim = 10
solver = MODE_C(2, 0, Bounds([0] * dim, [1] * dim), popsize=64,
                rg=Generator(PCG64DXSM(7)))
archive = ParetoArchive2D([1.1, 1.1], payload_dim=dim)
for _ in range(200):
    xs = solver.ask()
    ys = zdt1(xs)
    archive.add_batch(ys, xs)  # points outside (1.1, 1.1) are skipped
    solver.tell(ys)
print(len(archive), archive.hypervolume())
```

The archive only observes `ask`/`tell`, so the MODE run is unchanged. For
constrained problems, pass only the feasible rows and only the objective
columns. `examples/pareto_mode.py` shows this.

## Implementation

The C++ archive (`_fcmaescpp/include/paretofront2d.h`) stores the front in a
`std::map` ordered by the first objective, so the second objective strictly
decreases along the front. A cached front point rejects most candidates in
O(1). Otherwise an insertion costs one O(log n) lookup plus O(k log n) for the
k removed points; appending after the last point, or inserting into the gap
behind the previous insertion, skips the lookup.

`paretofront2d.h` is derived from `bitehv2.h` in Aleksey Vaneev's BiteOpt
repository, <https://github.com/avaneev/biteopt> (commit 5f86f50,
2026-10-13), MIT license: the ordered front, the delta hypervolume updates,
the cached rejection point and the append/gap fast paths come from there.
fcmaes adds compensated area sums, error handling with rollback, skipping of
outside points and payloads. `fcmaes-core` (Rust) has a port of the same
archive as `ParetoArchive2D`.

Standalone C++ tests and a benchmark, with optional sanitizers:

```sh
cmake -S tests/native -B build-native -DCMAKE_BUILD_TYPE=Debug -DFCMAES_SANITIZE=ON
cmake --build build-native && ctest --test-dir build-native --output-on-failure
build-native/benchmark_pareto 1000000
```
