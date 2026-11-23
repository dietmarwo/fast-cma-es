# Native box queries and Pareto search

These tools are in the repository but not yet in a PyPI release.
fcmaes-rust has the same tools in Rust (`fcmaes_rust.BordaBoxIndex`,
`GridBoxIndex`, `ParetoArchiveND`, `ArchiveDE`, `igd`). Given the same
inputs, both packages return the same row IDs in the same order, and
`ArchiveDE` with the same seed performs the same run.

| Operation | Python entry point | C++17 entry point |
| --- | --- | --- |
| Borda box selection | `fcmaes.rankindex.BordaBoxIndex` | `fcmaes::rankbox::BordaBoxIndex` in `rankbox.hpp` |
| Sparse rank-grid selection | `fcmaes.rankindex.GridBoxIndex` | `fcmaes::rankbox::GridBoxIndex` in `rankbox.hpp` |
| Archive-gated DE and its strict Pareto archive | `fcmaes.archivede.ArchiveDE`, `fcmaes.archivede.minimize`, `fcmaes.pareto.ParetoArchiveND` | `fcmaes::archivede::State` and `Archive` in `archivede.hpp` |
| Incremental 2D hypervolume | `fcmaes.pareto.ParetoArchive2D` | `fcmaes::ParetoArchive2D` in `paretoarchive.hpp` |

## Select points in a closed box

```python
import numpy as np
from fcmaes.rankindex import BordaBoxIndex, GridBoxIndex

points = np.array([[0., 1.], [0.5, 0.5], [1., 0.], [0.5, 0.5]])
lows = np.array([[0.25, 0.25], [0., 0.]])
highs = np.array([[0.75, 0.75], [1., 1.]])
for index in (BordaBoxIndex(points), GridBoxIndex(points, query_volume=0.25)):
    result = index.query_batch(lows, highs)
    first = result.ids[result.offsets[0]:result.offsets[1]]
    assert sorted(first) == [1, 3]
    assert np.array_equal(first, index.query(lows[0], highs[0]))
    print(dict(zip(index.stat_names, result.stats[0])))
```

Build an index once over a finite `(n, d)` matrix. It owns a snapshot, so
the input can be reused. A query includes both endpoints, returns original
row IDs as `uint32` and can use infinite endpoints. An inverted interval
or a NaN endpoint produces an empty result. Tie membership includes every
matching row. Ranks sort each axis by coordinate, and equal coordinates
(including `-0.0` and `0.0`) keep ascending row order, on every platform.
Borda emits hits in that order on the axis with the shortest rank interval.
The grid emits hits by cell key and then input ID.

`query_batch` returns one flat ID array, a `uint64` offset array of length
`queries + 1`, and one row of counters per query. Borda counts candidates
on the shortest axis and in the Borda score band. Grid counts touched cells,
occupied cells, fully contained occupied cells and individually tested points.
The theoretical touched-cell count is a float and may round above `2**53`.
Index construction and searches release the Python GIL. Keep input arrays
stable for the duration of a call.

Borda stores cross-ranks for each pair of axes, using O(n d²) space.
The grid stores O(n d) ranks and occupied cells. Its positive
`query_volume` is the nominal fraction of rank space covered by a typical
query and determines the resolution. Its cell keys fit in 64 bits.

## Optimize a Python objective

```python
import numpy as np
from fcmaes import archivede
from fcmaes.pareto import ParetoArchive2D, ParetoArchiveND, igd

def objectives(xs):
    x, y = xs.T
    return np.column_stack((x*x + y*y, (x-1)*(x-1) + y*y))

bounds = [(0., 1.), (-1., 1.)]
xs, ys = archivede.minimize(objectives, nobj=2, bounds=bounds, popsize=32,
                           max_evaluations=1000, seed=7, vectorized=True)
hv = ParetoArchive2D([2.1, 2.1], payload_dim=2)
hv.add_batch(ys, xs)
t = np.linspace(0., 1., 101)
reference = np.column_stack((t*t, (1.-t)*(1.-t)))
print(len(xs), hv.hypervolume(), igd(ys, reference))

# Record rows from any other evaluator or an existing MODE ask/tell loop.
archive = ParetoArchiveND(nobj=2, payload_dim=2)
admitted = archive.add_batch(ys, xs)
decisions, objectives_on_front = archive.front()
```

`minimize` returns decision and objective matrices for the nondominated
evaluations over the whole run. A row objective is the default.
`vectorized=True` calls the objective once per generation. `workers=k`
evaluates a pickleable row objective in an ordered process pool. On platforms
using `spawn`, launch the multi-process call inside an
`if __name__ == "__main__"` guard.

An external evaluator can drive the same search through synchronous ask/tell.

```python
solver = archivede.ArchiveDE(2, bounds, popsize=32, max_evaluations=1000, seed=7)
while len(xs := solver.ask()):
    solver.tell(objectives(xs))
xs, ys = solver.front()
population_xs, population_ys = solver.population()
assert solver.nfev == 1000
```

DE/rand/1/bin uses F = 0.5 and CR = 0.3 and has no constraint handling.
By default, the archive gates
offspring admission and survivor selection combines objective champions
with rank-space spread. `gate=False` admits all offspring to selection.
`tiered=False` selects by stable Borda truncation. Bounds are finite
pairs or a SciPy `Bounds` object. The population must have at least four
members and exceed the objective count. The budget includes initialization
and can end with a partial batch. Submit one finite objective row per asked
decision, in order. Repeated `ask()` returns a copy of the same pending batch.

`ParetoArchiveND` retains equal nondominated rows and their payloads.
It orders snapshots by rounded objective sum and then insertion order. The
2D hypervolume archive uses a finite reference point, skips rows outside the
reference box, and retains the first payload for each equal pair. Both
archives have unbounded capacity. A batch mask records admission on arrival.
A later row can dominate an earlier admitted row. Snapshot arrays are
independent of later updates. See the [2D hypervolume contract](pareto_archive.md).

`igd(front, reference)` computes the mean Euclidean distance from reference
rows to their nearest front rows in objective coordinates. The approximation
front comes first, as in `fcmaes_rust.igd`; swapped arguments give the
generational distance instead. An empty front has infinite distance. Use a
nonempty reference matrix of the same positive width.

## Build and use from C++

The CMake target `fcmaes_backend` exports the source-tree header directory
`_fcmaescpp/include`. Link it for `archivede::State` and `archivede::Archive`.
The box indexes, `fcmaes::ParetoArchive2D`, and
`fcmaes::front_distance::igd` (in `frontdistance.hpp`) are header-only.
Python uses the package's existing nanobind extension and NumPy arrays.
Standalone native callers can also compile `_fcmaescpp/archivede.cpp` with
their program. All native matrices use row-major doubles. The wheels contain
the Python extension. Native headers and the static target are available from
the source tree.

```cpp
#include "rankbox.hpp"

double points[] = {0., 1., 0.5, 0.5, 1., 0.};
double low[] = {0.25, 0.25}, high[] = {0.75, 0.75};
fcmaes::rankbox::BordaBoxIndex index(points, 3, 2);
auto hits = index.query(low, high);  // hits.ids contains input row 1
```

The [standalone native test](../tests/native/test_front_tools.cpp) exercises
the state, payload snapshots, queries and front distance without Python.

## Origin

`archivede.cpp` is derived from `archive_de.cpp` in Aleksey Vaneev's
[BiteOpt](https://github.com/avaneev/biteopt) repository (commit e8d6785).
Unlike that version, it compares archive entries whose rounded objective sums
are equal, which can hide a dominance relation. `rankbox.hpp` is derived from
two C++ programs that are not part of the published repository: a sparse rank
grid by Aleksey Vaneev and a Borda rank-band selector. The headers of both
files hold the copyright and MIT license notices.
