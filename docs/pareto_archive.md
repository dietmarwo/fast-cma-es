# Exact 2D archive and incremental hypervolume

The native archive is available through `fcmaes.pareto.ParetoArchive2D`.
It collects unique nondominated minimization objective pairs and the first
payload associated with equal pairs. It is separate from MODE's population,
the bounded shared result store and MAP-Elites.

```python
from fcmaes.pareto import ParetoArchive2D

archive = ParetoArchive2D([0, 0], [4, 4], payload_dim=2)
mask = archive.add_batch([[1, 3], [3, 1], [2, 3]],
                         [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
assert mask.tolist() == [True, True, False]
assert archive.hypervolume() == 5.0
xs, ys = archive.front()
assert archive.size == len(archive) == 2
```

Bounds describe objective coordinates. The finite lower bounds and reference
are fixed for the run. Negative objectives are supported inside those bounds.
Set `payload_dim=0` for indicator-only use and omit `xs`; otherwise provide
the fixed decision-vector width and a matching payload for each row. Payloads
are opaque values; finite validation applies to objectives, not payload data.

- `add(y, x=None)`: boolean acceptance; rejected payloads are not retained.
- `add_batch(ys, xs=None)`: matrices `(n,2)` and `(n,payload_dim)`, processed
  in order; returns a boolean acceptance mask. Noncontiguous input is copied.
- `front()`: owned `(xs,ys)` arrays in first-objective order, valid after
  further updates, clear or archive deletion. Export traverses the map directly.
- `hypervolume()`: union area without exporting.
- `clear()`: reset contents and HV, retaining bounds and payload width.

Matrix shape errors precede mutation. A numeric `ValueError` or
`OverflowError` names the failing zero-based batch row. Earlier successful
rows remain; the failing row does not mutate membership or HV. Batches are
not all-or-nothing transactions. A single kernel insertion now preserves the
front, payloads, caches and HV if allocation fails. This does not make a complete
Python batch transactional: earlier rows may already have succeeded. Empty batches use an explicit `(0,2)` objective matrix.

One owner updates an archive. Native calls retain the GIL in this initial
binding. The object is not a shared multiprocessing store; do not pass native
object pointers between processes. Exact antichain retention is unbounded.

## MODE integration and validation

```sh
python examples/pareto_mode.py
python -m pytest -q tests/test_pareto.py tests/test_wheel_smoke.py
```

The example ingests every finite feasible offspring before calling MODE's
`tell`. It uses strict trailing constraints `<= 0`. Add explicitly evaluated
initial guesses in application drivers. MODE's adaptive constraint policy and
legacy reporting tolerances remain separate, unchanged policies. The archive
does not change MODE's population update or random draws.

Native sanitizers can run independently of the Python extension:

```sh
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -I_fcmaescpp/include tests/native/test_pareto_wrapper.cpp -o /tmp/pareto-test
/tmp/pareto-test
```

`test_pareto_kernel.cpp` checks the ordered map before and after front inspection,
payload retention, rollback, copies and live references. The common
CSV fixture and `check_pareto_cross_language.py` can replay the Rust
`pareto_trace` executable against this installed wheel.

## Cached acceptance and ownership (13 October 2026)

Append and interior-gap hints are adapted from BiteOpt commit
`6f3a63a02da75f55242a5d51edf8f5b7e4ff680c`. Proven no-deletion insertions
use an appropriate map hint. The successor iterator belongs to this map; copies,
moves and swaps invalidate it, and reset clears it. Moved-from kernel objects
are empty, have zero HV, retain their reference and can be reused.

Accepted payloads are copied before any source entry is erased. New-key node
allocation precedes deletion and HV changes; same-key improvements swap a
prepared payload into the existing node. Single-insertion allocation failures
preserve the old state, including borrowed archive payloads. Copy assignment
also preserves its destination on failure. These are native C++ value semantics;
no Python copy protocol was added.

`test_pareto_safety.cpp` checks copy/move/swap, destroyed sources, aliased payloads
and nine injected insertion allocation failures, under debug iterators and
ASan/UBSan. No experimental ZDT4 cost or stopping threshold is enabled.

## Implementation and provenance

The root CMake extension target compiles `pareto_bindings.cpp`; it uses the
existing nanobind ndarray conventions. The wrapper validates objective bounds
and payload width, then calls the optimized tracker. It exports through
`forEachPoint`, never `getFront`. Scalar BiteOpt and MODE bindings are unchanged.

### Simplification on 12 October 2026

The kernel now contains one `std::map` and one cached dominance witness.
Compact vectors, paged storage, page split/merge thresholds, mode switches,
compatibility materialization and the victim-iterator buffer have been removed.
`getFront()` is always an O(1) live const view, with ordinary map iterator rules.
`forEachPoint()` traverses the same map. Neither inspection method changes storage.
The public class and Python interfaces remain unchanged.

The cached rejection wrapper stays small for compiler inlining; accepted points
use the general update or a proven no-deletion shortcut. Dominated successors form a contiguous interval,
which is erased only after positive compensated HV checks and payload copying.
A separate initialized flag accepts the reference corner into an empty archive.
The fcmaes wrapper still validates finite lower bounds and payload dimensions;
all numerical error and partial-batch guarantees remain in force.

The architecture follows BiteOpt's cached-front update at commit
`02427c2f1dad87f3a146eda8b4613f01ee2eb8d8`; this is a guarded fcmaes adaptation,
not a verbatim upstream copy. Benchmarks in the delivery report quantify the
cost of removing compact storage. No universal speedup is claimed.

The current vendored `bitehv2_optimized.h` is byte-identical to the separately
delivered header (SHA-256):

```text
af763face7e3d04949d33def78d647eacca71db36a140f829a984efd73cce86a
```

The earlier integration originally referenced the BiteOpt `bitehv2.h` Git blob:

```text
f7afb72c66a859b8417ef2f7da20a756c77abdb0
```

Retain Aleksey Vaneev's full MIT notice in the header. This fcmaes adaptation
does not change the original BiteOpt repository. The header is a replacement
definition of `RankSumFrontHV`; do not include it with the original HV header
in the same translation unit. The C++ code covers 2D only.
