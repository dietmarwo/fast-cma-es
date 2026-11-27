# Changelog

All notable changes to the `fcmaes` Python package are documented here, in
the format of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow the PyPI releases; since 2.0.0, pushing a tag `vX.Y.Z`
publishes that version to PyPI.

## [Unreleased]

## [2.3.0] - 2026-11-27

### Changed behavior to expect

These follow from the fixes below. The measurements are in
[docs/performance-120/REVIEW.md](docs/performance-120/REVIEW.md).

- **CMA-ES runs longer and ends closer to the optimum.** The history of
  generation bests used by two stop criteria held copies of one value, so a
  run stopped as soon as a few consecutive bests agreed. On 225 seeded
  serial runs the fixed history used 39% more evaluations, improved the
  result in 145 and worsened none. Runs that ended at `max_evaluations`
  before still do. On objectives with many equal values the trajectory
  itself changes, because the flat-fitness step-size rule reads the same
  history.
- **MAP-Elites now runs its CMA-ES emitter.** In 2.2.0 the emitter failed in
  every iteration, silently, and only the variation phase ran. With default
  parameters one iteration of one worker now costs up to 62,200 evaluations
  instead of 200. Pass `cma_params={'cma_generations': 0}` for the former
  cost, or set `cma_generations`, `maxiters` and `popsize`.
- **Worker failures raise.** An exception in a MAP-Elites or Diversifier
  worker process, such as an unknown solver name or an objective that
  raises, now ends the call with `RuntimeError`. Before, the call returned
  the archive as far as it had come.
- **Budgets are exact.** Delayed CMA-ES, parallel DE and the Diversifier
  made up to a generation or a batch more calls than `max_evaluations` /
  `max_evals`; they now stop at the limit.
- **DE reports the best point it evaluated.** The reported value could be
  worse than a value the objective had returned (4 of 180 seeded serial
  runs, 18 of 96 parallel ones).
- **`Bite_C` with `batch_size` above 1 proposes other points** than before,
  because its first batches no longer repeat population members.
  `bitecpp.minimize` and `batch_size=1` give the same runs as 2.2.0, as do
  serial DE and MODE.
- The Diversifier's BiteOpt emitter uses `Bite_C` ask/tell, starts from the
  best point of the preceding emitter in a sequence, and returns the best
  real objective point of its stage.
- A CMA-ES result has `success=False` when the native run reports a
  negative stop code.

### Remaining native optimizers and front tools

- CR-FM-NES returns the best evaluated physical point and its objective value.
  Box-violation penalties still determine the distribution update.
- CR-FM-NES and PGPE count the final partial batch against the requested
  budget and update the result with any improvement in that batch.
- Dual Annealing charges local-search probes and initial retries to the
  evaluation budget. The result reports attempted annealing iterations.
- The C++ CR-FM-NES, PGPE and jSO wrappers accept an optional matrix
  objective with `vectorized=True` and a single worker. The matrix is
  composed of physical points. jSO counts every row of an evaluated matrix.
- The strict Pareto archive can use objective-coordinate ranges to prune
  comparisons on fronts with nearly equal sums. Exact dominance and
  duplicate semantics are preserved. IGD has indexed searches for additional
  dimensions and rescales uniformly extreme coordinates before searching.
- Native rank queries reduce work when filtering blocks of candidates and
  looking up sparse grid cells.

### Optimizer performance and correctness

- Native CMA-ES delayed updates decode points once, honor the evaluation
  budget and retain the best completed point, including partial generations.
  Fitness history now advances correctly. Supplied-point updates use the
  eigensystem directly to recover z.
- BiteOpt's batch initialization proposes each unevaluated population member
  once and feedback avoids redundant buffer copies.
- Native DE tracks the best observed objective independently of its mutation
  anchor. Delayed results retain their original candidate and obey the
  requested budget. Pass `strict_budget=True` to cap serial callbacks inside a
  generation. Batch proposal, integer-mask and pending-trial work use fewer
  allocations.
- Native MODE computes exact one- and two-objective front levels using sorted
  sweeps. Generic dominance checks stop on a proven incomparable pair.
- `cmaescpp.minimize` and `modecpp.minimize` accept `vectorized=True` for
  single-worker population objectives.
- MAP-Elites can initialize an archive and start its native CMA emitter.
  Archive updates keep the winning point, descriptor, score and counters
  consistent across processes. Grid lookup has a native fallback when Numba
  is unavailable. Pass `persistent_workers=True` to reuse worker processes
  across outer iterations.
- MAP-Elites and Diversifier accept a root `seed` and report child-process
  failures. Diversifier enforces a shared callback budget and accepts
  `inline=True` for one worker. Its native emitters accept an optional
  `batch_fitness` in `opt_params`. For BiteOpt set `batch_size` in
  `opt_params` to use batched archive feedback.


### Added

- Native Borda and grid box queries in `fcmaes.rankindex`, including batched
  row-ID results and per-query work counts. Equal coordinates keep row order,
  so results are the same on every platform and in fcmaes-rust.
- `fcmaes.archivede` for archive-gated, bounded multi-objective differential
  evolution (DE/rand/1/bin, F = 0.5, CR = 0.3, no constraints) with ask/tell
  and scalar, vectorized or process-parallel Python objectives. It returns the
  exact front of all evaluations; with the same seed, fcmaes-rust's
  `ArchiveDE` performs the same run. `fcmaes.pareto` also exposes its
  duplicate-preserving N-D archive (`ParetoArchiveND`) and
  `igd(front, reference)`, with the argument order of `fcmaes_rust.igd`.
- The native code of both is derived from Aleksey Vaneev's C++
  (`archive_de.cpp` from BiteOpt, and a rank grid and a Borda selector); the
  headers of `archivede.hpp` and `rankbox.hpp` hold the notices. Documented in
  `docs/native_front_tools.md`, `ai-context.md` and the API pages.

## [2.2.0] - 2026-11-20

Behaviour changes: PGPE (`pgpecpp`, `PGPE_C`, `Pgpe_cpp` and the
Diversifier's `'PGPE'` solver) gives different results than 2.1.0 with
`normalize=True` and bounds other than `[-1, 1]`, and returns the best
evaluated point and its objective value instead of an unevaluated point and
a rank value. Seeded C++ runs after other runs in the same process now
reproduce the first run in a fresh process. `de_cma`, `da_cma`, `bite_cma`
and `cma_bite` called directly through `Sequence.minimize` run their CMA-ES
stage again. The default `repair_z=False` keeps the CMA-ES update of 2.1.0;
`retry.minimize` and `advretry.minimize` results are unchanged except
through these fixes. See Fixed for the details.

### Added

- Stock jSO for bounded continuous scalar optimization. `fcmaes.jso` and
  `fcmaes.jsocpp` expose `minimize` and synchronous `Jso` ask/tell with a
  hard evaluation cap and optional process-parallel objective evaluation.
  `Jso_python` and `Jso_cpp` support retry, and `optimizer.jso_cma` starts
  CMA-ES from the native jSO incumbent with a nominal 40/60 budget split.
- Links to the eight CRN/CFD simulator optimization lessons, distinguishing their native Rust/command execution from this package's in-process Python API.
- [Prompt-driven optimization tutorials](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic) demonstrate the shared agent workflow with native Rust application workers. Python objectives continue to use this package in process; these tutorials do not add a worker to fast-cma-es.
- CMA-ES option `repair_z` (default `False`) for samples that the bound
  repair moved onto the bounds. The mean follows their repaired (evaluated)
  points either way. With `False`, the update of 2.1.0, the evolution paths
  and the rank-mu / active covariance update use the random vector `z` of
  the unrepaired sample; `True` recomputes `z` from the repaired point.
  Available on `cmaescpp.minimize` (single-threaded and with `workers > 1,
  delayed_update=False`), `cmaescpp.ACMA_C` and `cmaes.Cmaes` (ask and
  `tell(ys)`), `optimizer.Cma_cpp`, `optimizer.Cma_python` and the sequences
  `optimizer.de_cma`, `jso_cma`, `da_cma`, `bite_cma` and `cma_bite`, e.g.
  `advretry.minimize(..., optimizer=de_cma(1500, repair_z=True))`.
  `cmaes.minimize` (used by `Cma_python`), `tell(ys, xs)`, `tell_one` and the
  delayed-update paths have always derived `z` from the evaluated points and
  are not affected.
  - **Default:** in 2,400 matched pairs on 12 application objectives (in
    fcmaes-rust, at about 63,000 evaluations per run), `False` won more
    often for standalone bounded CMA-ES (222 wins against 149, 13 ties), as
    the second stage of DE→CMA (217 against 62) and jSO→CMA (205 against
    75), at exactly equal work (166 against 49) and in coordinated DE→CMA
    retry (74 against 43). At a quarter of that budget coordinated retry
    had leaned toward `True` (72 against 58). On a 171-variable truss
    jSO→CMA reached a median of 394.5 with `False` against 420.4 with
    `True`. `True` was clearly better where samples keep being clipped and
    the step size inflates (a 118-variable neural controller: median loss
    0.137 against 1.13).
  - **When to try `True`:** runs that stop early far from the optimum with
    the step-size stop (status 3 or 5), optima on or outside the bounds, and
    samples clipped throughout the run. On `sum(x)` and on a sphere with half
    its optimum outside `[-1,1]^20` (popsize 16, 20,000 evaluations),
    `cmaescpp.minimize` reaches -19.8..-20.0 (optimum -20) and 10.3..10.7
    (optimum 10) with `True`, -14.2..-16.0 and 14.5..16.9 with `False`.

### Changed

- The jSO tutorial and `ai-context.md` state the result of the coordinated-retry
  test in fcmaes-rust: with `advretry`, `jso_cma` was better than `de_cma` on
  none of eight GTOP problems and significantly worse on Sagas and Messenger
  full ([table](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/algorithm-portfolio.md#why-jso-is-in)).
- The READMEs and `ai-context.md` link the
  [algorithm portfolio of fcmaes-rust](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/algorithm-portfolio.md), which records how the scalar
  algorithms were selected and the measurements behind each decision.
- README: links the [overview of the 18 related repositories](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/repositories.md).
- The READMEs point to [experiment-craft](https://github.com/dietmarwo/experiment-craft),
  which runs reproducible campaigns of the Rust job workers.
- The READMEs open with the fast-cma-es banner (`tutorials/img/fast-cma-es.png`)
  instead of the animated logo; `README.md`, the PyPI description, links it by
  absolute URL. The tutorials keep `logo.gif`.

### Fixed

- CMA-ES active covariance update: it rescales the worst samples by ratios
  of their `z` norms and divided by zero when a sample was told at exactly
  the mean (`z = 0`), e.g. a sample clipped onto a mean that lies on a
  bound, whose `z` is derived from its point. In the C++ backend this put
  NaN into the covariance matrix and the run stopped (status 2): on one
  variable in `[0,1]` started at 0 (sigma 0.15, popsize 31), `ACMA_C` with
  `repair_z=True` or with `tell(ys, xs)` (which now updates from the points,
  see below) stopped after 31 evaluations, and
  `cmaescpp.minimize` with `repair_z=True` on an optimum at the start bound
  after 62 to 93. Now the negative part is skipped in such a generation
  (zero or non-finite norm); otherwise results are unchanged. The Python
  backend replaced a zero norm by 1e-30, which kept the matrix finite but
  divided by zero (RuntimeWarning) when all these norms were zero; it now
  skips the same way, which changes results only in that case.
- `cmaescpp.ACMA_C.tell(ys, xs)` right after an `ask()` updated from the
  `z` of the asked samples instead of the supplied points `xs`.
- `optimizer.Cma_cpp` with `sdevs=None` passed NaN step sizes to CMA-ES,
  which stopped after two generations (62 evaluations at popsize 31) with
  the value of a failed evaluation (1.8e308). `Sequence.minimize` passes
  `sdevs=None` unless the caller supplies step sizes, so a direct call such
  as `de_cma(...).minimize(fun, bounds)` skipped the CMA-ES stage of
  `de_cma`, `da_cma`, `bite_cma` and `cma_bite`. `Cma_cpp` uses 0.1, its
  default for `sdevs`, in that case. `retry.minimize` and
  `advretry.minimize` always pass step sizes; their results are unchanged.
  `Sequence.minimize` raised `UnboundLocalError` when its first stage
  returned an infinite or NaN value; it keeps the first stage's point and
  does not let a NaN value replace a result.
- C++ optimizers: a run with a fixed seed could give a different result
  depending on what ran before it in the same process. The normal samplers
  shared one `std::normal_distribution` object per source file (a
  namespace-scope `static` in `evaluator.h` and one in `daoptimizer.cpp`,
  present since at least 2.0.0). That object caches the second value of each
  generated pair, so after a run with an odd number of normal draws the next
  run started with the cached value of the earlier run instead of a value
  from its own seed. Ask/tell objects stepped alternately or in different
  Python threads affected each other the same way (in threads also a data
  race). Each optimizer instance now owns its normal distribution. Affected:
  CMA-ES (`cmaescpp.minimize`, `ACMA_C`), DE with `x0` and `input_sigma`
  (`decpp.minimize`, `DE_C`), CR-FM-NES (`crfmnescpp.minimize`,
  `CRFMNES_C`), PGPE (`pgpecpp.minimize`, `PGPE_C`) and the optimizer
  sequences built on them. Dual Annealing draws normal values in pairs, so
  its single-threaded results were not affected; it gets its own
  distribution too. BiteOpt and MODE do not use the shared object. The
  first run in a fresh process gives the same results as before; a later
  run with the same seed reproduces the first instead of depending on the
  runs before it.
- PGPE result: `pgpecpp.minimize` (and `optimizer.Pgpe_cpp`) returned an `x`
  that was never evaluated, usually uninitialized memory (e.g. values like
  `4.7e-315` or `6.9e-310`), and with the default `use_ranking=True` a `fun`
  that was a rank value (e.g. `-0.4375` for popsize 16), not an objective
  value. `PGPE_C.result()` with `use_ranking=True` returned the same rank
  value as `fun` and the best point of the first generation as `x`. Both now
  return the best evaluated point (decoded, inside the bounds) and its
  objective value, with and without ranking. The evaluated points and the
  number of evaluations are unchanged; for the stop criterion see the
  `stop_fitness` entry below. Present in 2.0.0 and 2.1.0.
- PGPE start point: ADAM, which updates the center of the search
  distribution, started from the raw `x0` while the center is kept in
  normalized coordinates (the bounds mapped to `[-1, 1]`). After the first
  generation the center jumped from the normalized `x0` to the raw `x0`
  read as a normalized point: with `x0 = 1` on `[-5, 5]` the center of the second
  generation lay between 4.25 and 5.75 (samples clipped to 5) instead of
  within 0.75 of 1, so the search lost its start point and could
  stay on a bound. Affected: every run of `pgpecpp.minimize`, `PGPE_C`,
  `optimizer.Pgpe_cpp` and the Diversifier's `'PGPE'` solver with
  `normalize=True` (the default) and bounds other than `[-1, 1]`, including
  the random start points of `retry` and the guesses of `advretry`. ADAM
  starts from the normalized `x0`; results of such runs differ from 2.1.0.
  Runs with `normalize=False`, without bounds or with bounds `[-1, 1]`
  evaluate the same points as before. Present in 2.0.0 and 2.1.0.
- PGPE `stop_fitness`: with `use_ranking=True` (the default of
  `pgpecpp.minimize` and `optimizer.Pgpe_cpp`) it was compared with the best
  rank value instead of the best objective value. The best rank value is
  `-(0.5 - 1/popsize)` in every generation (`-0.4375` for popsize 16,
  `-0.46875` for 32), so a `stop_fitness` above it stopped the run after
  the first generation and a lower one never stopped it. It is compared with
  the best objective value found, as with `use_ranking=False`. This fix
  does not change runs with the default `stop_fitness=-inf` or with
  `use_ranking=False`.
  `PGPE_C` has no stop test. Present in 2.0.0 and 2.1.0.
- `cmaescpp.minimize` or `decpp.minimize` called from two Python threads at
  the same time hung forever, and so did a CMA-ES objective that itself
  calls `cmaescpp.minimize` (and a DE objective calling `decpp.minimize`).
  Each of the two bindings stored the Python objective in one global
  pointer guarded by a mutex that a call took while holding the GIL; the
  running call needed the GIL for its next objective evaluation. Each call
  now keeps its objective in its own per-call context, which also reaches
  the worker threads of `workers > 1`, so such calls complete and each uses
  its own objective. Results of a single call are unchanged. Present in
  2.0.0 and 2.1.0.
- `cmaescpp.minimize` and `decpp.minimize` with `workers > 1` read freed
  memory: the queue between the optimizer and its evaluator threads
  returned a reference to an element it had just removed (a
  heap-use-after-free that AddressSanitizer reports in every such run). On
  Linux this went unnoticed; on Windows two such calls running at the same
  time crashed with an access violation or heap corruption. The queue now
  returns a copy. The evaluation counter, the terminate flag and the stop
  flag of the evaluator, written by several evaluator threads, are atomic
  (ThreadSanitizer reported data races on them). Results of
  single-threaded calls are unchanged. The queue code is the same in
  2.0.0 and 2.1.0.
- Tutorial text and figures that disagreed:
  - Scheduling: the SBM BiteOpt section showed the single-thread `Bite1.png`; it now shows `sbmBite32.png`.
  - UAV: `uav_reward.png` was a screenshot of six image-viewer windows whose legends covered the leading bars. It is redrawn from the Multi-UAV benchmark CSVs with the same means.
  - Claims corrected to match the plots and tables they describe: OneForAll MMKP timing, TopTrumps random search at 20 s, EvoJax (cartpole hard, brax, waterworld, waterworld MA and the larger-budget runs), MINLP best run and Employee solve times.
  - PAGMO notes a mislabelled axis and empty legend counters.
  - CryptoTrading no longer shows the first run's plot as the second run's.
- Section levels in `README.adoc` and seven tutorials skipped levels. Asciidoctor reported 86 "section title out of sequence" warnings, and the HTML went from `h1` straight to `h3`/`h4`. The levels now close the gaps and keep the original nesting. `scripts/build_docs.py` prints Asciidoctor warnings and fails on them (`--failure-level WARN`), so new ones stop the documentation build.
- Broken GitHub links in the tutorials and the examples API page:
  - Files that were removed from this repository (`ascent.cpp`, `gcldecpp.py`, the old `Results.adoc`) now link to their last version.
  - The missing `Filter.adoc` link points to the filtering exercise in `Hospital.adoc`.
  - VRPTW and fcmaes-java links use their current branch and path.
  - pykep links point to the 2.6.4 release that the tutorials use.
  - Moved upstream files (GillesPy2, MOBOpt, floris) and the pymoo URL are corrected.
  - Two attributions to repositories that no longer exist keep their names without a dead link.

## [2.1.0] - 2026-10-22

Behaviour changes: MODE (`modecpp` and `mode`) gives different results than
2.0.3, and `moretry` gives different results for problems with constraints
(`ncon > 0`). With the MODE fixes below (popsize 64, default settings,
10 seeds, 300k evaluations), the NSGA update solves ZDT2 in 10 of 10 runs
instead of 3 and reaches hypervolume 1.000 on ZDT4 instead of 0.125; the DE
update reaches 0.895 on ZDT4 instead of 0.000. On 23 constrained problems
(including MW1-13, C2-DTLZ2 and CEC2006 g-problems), the final population
of the native MODE is significantly better than with 2.0.3 in 29 of 46
problem/mode cells and worse in 1, and no run loses its feasible points.
Rerun MODE baselines made with 2.0.3 or earlier instead of comparing to
them.

### Added

- `examples/external_simulator.py`: an objective that runs an external
  program in a fresh working directory.
  - **Process tree:** each run gets a process group of its own. On exit,
    timeout or interruption the whole group is stopped (SIGTERM, then
    SIGKILL), so solver processes the program started do not keep running.
  - **Failures:** a timeout, a nonzero exit, a missing output or a non-finite
    value gives a finite failure value. They are counted in a ledger shared
    by the worker processes (fork and spawn): `summary()` counts by kind, and
    `check_result` refuses a campaign without a successful evaluation or with
    a best point that fails its recheck.
  - **Seeds:** the default point-derived seed replays a point. `seeds=` runs
    every candidate on the same replication seeds (common random numbers),
    and `replication_seeds` and `with_seeds` give disjoint holdout
    validation. `with_seeds` copies the objective, so a subclass's file
    formats, transformations and state carry over to the holdout. The
    ledger counts attempted simulator runs; an aggregate that is not finite
    fails as `invalid_aggregate`, apart from simulator failures.
  - **Threads:** the program's thread pools are set to one thread, which
    overwrites inherited values, unless set explicitly; `topology()` records
    the layout.
  - **Usable CPUs:** `usable_cpus()` counts the CPUs the process may use: the
    affinity mask and the cgroup v1 or v2 quota.

  The object is picklable for parallel `retry`. Tested in
  `tests/test_external_simulator.py`, which the wheel and sdist jobs run.
- `AGENTS.md`, a short entry point for coding agents. Both READMEs link it
  and `ai-context.md`. `ai-context.md` covers external programs, usable
  CPUs and memory, and running a long Python job detached.
- `fcmaes.pareto.ParetoArchive2D`: exact archive of all nondominated points
  of a two-objective minimization problem inside a reference box, each with
  an optional payload (usually the decision vector). Every insertion updates
  the dominated hypervolume exactly, with Kahan-compensated sums. Methods:
  `add`, `add_batch` (returns the mask of archived rows), `front`,
  `hypervolume`, `clear` and `len()`. Points with an objective above the
  reference are not archived (`add` returns `False`); non-finite objectives
  raise `ValueError`. The archive only observes `ask`/`tell`, so it can
  record every generation of a MODE run without changing the run. See
  `docs/pareto_archive.md` and `examples/pareto_mode.py`.
- Native kernel `_fcmaescpp/include/paretofront2d.h`
  (`fcmaes::ParetoFront2D`), bound as `fcmaes._fcmaes_ext.ParetoArchive2D`.
  It is derived from `bitehv2.h` in Aleksey Vaneev's BiteOpt repository
  (commit 5f86f50, MIT license): the ordered front, the delta hypervolume
  updates, the cached rejection point and the append/gap fast paths come
  from there; fcmaes adds compensated sums, error handling with rollback,
  skipping of outside points and payloads.
- Tests `tests/test_mode_regression.py`, `tests/test_pareto.py` (with the
  fixture `tests/data/pareto_archive_2d.csv`) and `tests/test_moretry.py`,
  and a smoke test that imports every module in `fcmaes.__all__` and runs
  `from fcmaes import *`.
- Standalone C++ tests and a benchmark for the archive in `tests/native`,
  with their own CMake project and optional sanitizers.

### Changed

- Discovery: the READMEs start with links to the shared solver-selection
  guide and the ecosystem overview, and credit OptunaHub's FastCmaesSampler
  as a third-party integration. The package description is more specific,
  with more keywords and project URLs (ecosystem, agent guide, issues,
  changelog). `RELEASING.md` covers discovery after a release.
- MODE (`modecpp` and `mode`): feasibility first. A point counts as
  feasible only if it violates no constraint (tolerance `eps = 0`). The
  adaptive tolerance, half the smoothed mean violation of the previous
  generation, is gone: once the population converged onto active
  constraints it let slightly infeasible points with better objectives
  displace every strictly feasible point, so `minimize()` could return a
  population without a feasible solution. Equality constraints now need an
  explicit tolerance, e.g. `abs(h(x)) - tol <= 0`. The multiple constraint
  ranking of infeasible points is unchanged; `mode.pareto_domination()`
  keeps its signature and ignores `last_ycon` and `last_eps`.
- `moretry`: satisfied constraints no longer enter the weighted p-norm of
  `mo_wrapper.eval`; only violations `max(c, 0)` count, plus the weight of
  each violated constraint as a penalty, as before. A satisfied constraint
  `c <= 0` added `(w*c)^p`, which penalized points deep inside the feasible
  region and made results depend strongly on the constraint weights. Median
  hypervolume of the feasible front over ten runs (64 retries of
  `Bite_cpp(1500)`) for constraint weights 10 / 1 / 0.1: Binh-Korn
  1000 / 3358 / 5074 -> 5150 / 5176 / 5159, CONSTR 0.83 / 3.45 / 4.17 ->
  3.93 / 4.08 / 4.02.
- `moretry.minimize` docstring: `weight_bounds` holds one weight per value
  `fun` returns (objectives first, then the `ncon` constraints), and
  `value_limits` limits the leading values.
- Tutorials and examples use the loguru `logger` (`logger.info(...)`) of
  1.6.x and the 2.x signatures. 18 tutorials and 6 examples still called
  `logger()` and passed `logger=` or `log=` arguments that 2.x functions do
  not accept. `multiretry.minimize` is called with keyword arguments,
  `tutorials/JobShop.adoc` passes `num_retries` and `popsize` by keyword
  (the fifth parameter of `modecpp.retry` is `guess`), and
  `tutorials/Quant.adoc` explains the DEBUG-level progress log.
- `examples/maxcut.py`: the unused `logger` parameter of `fcmaes_Optimizer`
  is removed.
- API reference pages (`docs/*.html`): the Python page describes the
  nanobind extension `fcmaes._fcmaes_ext` instead of ctypes and has a card
  for `pareto.py`. The C++ page documents the bindings in
  `_fcmaescpp/bindings` (the `optimize_*` functions and the `ACMA`, `Bite`,
  `CRFMNES`, `DE`, `MODE` and `PGPE` classes) instead of the `extern "C"`
  exports of 1.x, notes that `Bite` has ask/tell since 2.0.2 and has a card
  for `paretofront2d.h`. The examples page has a card for `pareto_mode.py`
  and no longer lists `reproblem.py` and `repython.py`.
- `ai-context.md` covers the MODE fixes, `ParetoArchive2D`, the new
  `moretry` constraint semantics, source builds of `master` (including
  `FCMAES_FETCH_EIGEN`) and related projects (fcmaes-rust,
  optimization-jobs).
- `README.md`, the PyPI project description, links fcmaes-rust and
  optimization-jobs and introduces `ParetoArchive2D`.
- Wheel CI runs the whole `tests` directory instead of only
  `tests/test_wheel_smoke.py`. A new workflow (`native-tests.yml`) runs the
  C++ archive tests with GCC plus ASan/UBSan and with Clang.
- Release workflow (`wheels.yml`): a tag must equal `v` plus the version in
  `pyproject.toml` (publishing uses `skip-existing`, so a tag without the
  version bump would have published nothing); the source distribution is
  installed and tested outside the checkout; PyPI releases get GitHub
  artifact attestations. `RELEASING.md` describes the release, and both
  READMEs describe installation from source (C++17 compiler, Eigen 3.4 from
  the system or fetched with `FCMAES_FETCH_EIGEN=ON`; no other repository of
  ours is needed). This changelog replaces the "Changes from version"
  lists in `README.adoc` as the complete record. `ai-context.md` points to
  the AI entry point of optimization-jobs, which chooses between the four
  solver repositories.

### Fixed

- MODE SBX crossover (C++ and Python): the expanding branch uses
  `(2 - 2r)^(-1/(eta+1))` (beta >= 1) instead of `(2r)^(-1/(eta+1))`
  (beta < 1), so children can leave the parents' interval.
- MODE polynomial mutation (C++): the upward branch uses `norm^(eta+1)`
  instead of `(1 - norm)^(eta+1)`. At the lower bound both branches gave a
  zero step, so a variable that touched its lower bound could never move
  again.
- MODE Pareto ranking (C++ and Python): `pareto_levels` is standard
  non-dominated sorting (higher level = better front). It counted
  sequential-elimination rounds, which ranked dominated points by the
  position of their first dominator instead of by front.
- MODE crowding distance (C++ and Python): the range-normalized NSGA-II
  distance over all objectives, and only the objectives, instead of the
  first objective only.
- `mode` (Python): in `ranks()` a satisfied constraint adds no rank, also
  for infeasible points, as in the native MODE. The arbitrary tie-order
  ranks left the Python MODE unable to reach small feasible regions
  (problem g6: feasible points in 10 of 10 runs instead of 1).
- `mode` (Python): `variation()` clips the offspring to the bounds before
  the polynomial mutation; expanding SBX can leave the box, and the
  unclamped normalization produced NaN.
- `moretry.minimize`: `value_limits` may list only the objectives of a
  constrained problem. The comparison over all `nobj + ncon` values raised
  an `IndexError` inside the retry loop, which silently dropped every
  result.
- `fcmaes.__all__`: `from fcmaes import *` raised `AttributeError`. It lists
  `crfmnes` instead of the misspelled `crmfnes`, no longer lists the removed
  `csmacpp`, and adds `pareto` and `pgpecpp`.
- `examples/__init__.py`: `__all__` had no commas between its entries (one
  concatenated string) and listed removed modules; it lists the example
  modules now.
- `tutorials/MINLP.adoc`: two errors in code snippets.

## [2.0.3] - 2026-07-26

The package code is unchanged from 2.0.2; this release updates metadata and
documentation.

### Added

- Documentation book on GitHub Pages
  (<https://dietmarwo.github.io/fast-cma-es/>): the README, the AsciiDoc
  tutorials and the Python, C++ and examples API references as one site,
  built by `scripts/build_docs.py` and the `docs.yml` workflow.
- Project URLs `Documentation` and `Tutorials` in the package metadata.
- `ai-context.md`: operational context for an AI that designs and
  implements an optimization solution with fcmaes.

### Changed

- `README.md`, the PyPI project description, links the documentation book,
  the tutorials and the Python, C++ and examples API references.
- `README.adoc` links fcmaes-rust, the native Rust implementation with a
  PyO3 API for Python users.

## [2.0.2] - 2026-04-12

### Added

- BiteOpt ask/tell interface `bitecpp.Bite_C` with `ask()`, `tell(ys)` and
  `result()`, backed by the native class `fcmaes._fcmaes_ext.Bite`
  (`_fcmaescpp/include/biteopt_asktell.h`). `ask()` returns up to
  `batch_size` candidates (default 8) generated from a frozen optimizer
  state; population updates and selector feedback happen in `tell()`. This
  parallelizes a single BiteOpt run for expensive objective functions; the
  delayed feedback can cost convergence per evaluation compared to the
  sequential run.
- Test `fcmaes.test_cma.test_rosen_ask_tell_bite`.

### Changed

- `bitecpp.minimize` runs on the same native ask/tell implementation, one
  candidate per step.

### Removed

- `fcmaes.__version__`. The version is defined in `pyproject.toml`; read it
  with `importlib.metadata.version("fcmaes")`.

## [2.0.1] - 2026-03-28

### Fixed

- The PyPI release includes the source distribution again; 2.0.0 was
  published with wheels only. The sdist excludes documentation, examples,
  tests, tutorials and images.
- `mapelites.Archive.get_qd_score`: `np.reciprocal(..., where=...)` gets an
  explicit zero-initialized `out` array.
- Examples that call native helper code (`damp.py`, `f8.py`, `hbv`, `lrgv`,
  `top_trumps.py`) load Linux shared libraries from `examples/lib`, as
  2.0.0 no longer ships `fcmaes/lib`.
- `examples/crypto.py` handles the MultiIndex columns of current `yfinance`
  downloads.

### Removed

- `examples/reproblem.py` and `examples/repython.py`.

## [2.0.0] - 2026-03-16

Includes the changes of the tag `v1.6.13` (2026-03-14), which was not
published to PyPI.

### Added

- Binary wheels for CPython 3.10 to 3.13 on Linux x86_64 (manylinux_2_28),
  Windows AMD64 and macOS x86_64 and arm64, built in CI with cibuildwheel.
  Without a matching wheel, `pip` builds from source.
- Release workflow: wheels are built on pull requests and branch pushes, a
  manual run with `publish_target=testpypi` publishes to TestPyPI, and tags
  `v*` publish to PyPI, using Trusted Publishing.
- Type stub `fcmaes/_fcmaes_ext.pyi` for the native extension.
- Wheel smoke test `tests/test_wheel_smoke.py`, run against every built
  wheel, and `scripts/smoke_nanobind.sh`.
- Optional dependency group `jit` for `numba`.
- `journal`: `journal_wrapper` marks infinite or NaN results as pruned
  trials in the Optuna journal (`state` parameter of `trial_value`,
  `Journal.value` and `journal_wrapper.store_y`).

### Changed

- The C++ optimizers are a nanobind extension module, `fcmaes._fcmaes_ext`,
  built with scikit-build-core and CMake, instead of shared libraries
  loaded with ctypes. Public Python functions and classes keep their
  signatures, except in `journal`. The GTOP space mission benchmarks of
  `fcmaes.astro` are part of the extension.
- macOS is fully supported, including ARM.
- `pyproject.toml` replaces `setup.py` and `requirements.txt`.
- Python 3.10 or newer is required (1.6.x: 3.7).
- `numba` is optional; `mapelites` works without it, without the JIT
  speedup.
- Improved type annotations, documentation and tutorials; the tutorials
  were reworked in `v1.6.13`.
- Most modules call `np.set_printoptions(legacy='1.25')` on import, which
  changes NumPy's print format for the whole process.

### Removed

- The prebuilt shared libraries in `fcmaes/lib` (`libacmalib` for Linux,
  Windows and macOS, and the libraries used by examples).
- `journal.Journal.write_x`, `write_xs`, `write_ys`, `write_starts` and
  `close`.

## [1.6.x] - 2026-01-22

Releases 1.6.0 (2023-09-13) to 1.6.12 (2026-01-22). The tag `v1.6.13` was
not published to PyPI; its changes are listed under 2.0.0.

- Logging is based on `loguru`. The `logger` parameters of the retry,
  optimizer and quality-diversity functions and `fcmaes.optimizer.logger()`
  are removed.
- New dependencies `loguru` and `numba` (`numba` is optional since 2.0.0).
- An initial population can be passed as `guess` for multi-objective
  optimization (`mode.minimize`, `modecpp.minimize`, `modecpp.retry`).
- MODE ask/tell works on whole populations: `ask()` returns all
  candidates and `tell(ys, xs=None)` takes all values. This replaces the
  per-candidate `ask()`/`tell(p, y, x)` and `ask_all()`/`tell_all()` of the
  Python `MODE`.
- New tutorial on the GECCO 2023 Space Optimization Competition,
  `tutorials/ESAChallenge.adoc`.
- 1.6.9 removes the C++ optimizers CSMA (`csmacpp`, `Csma_cpp`), GCL-DE
  (`gcldecpp`, `GCLDE_cpp`), LCL-DE (`lcldecpp`, `LCLDE_cpp`) and LDE
  (`ldecpp`, `LDe_cpp`).
- 1.6.11 adds `fcmaes.journal`, which writes an Optuna journal file for
  live monitoring of long runs with optuna-dashboard.
- 1.6.12 adds tutorials on weighted spherical t-designs, a Moran process
  simulation and an open-source AI optimization assistant, and updates the
  bundled LBFGSpp.

## [1.5.x] - 2023-09-07

Releases 1.5.0 (2022-11-18) to 1.5.11 (2023-09-07).

- Python 3.7 or newer is required; 3.6 is no longer supported (1.5.0).
- PEP 484 type hints, useful for IDEs (1.5.0).
- Pure Python versions of the algorithms (CMA-ES, CR-FM-NES, DE, MODE,
  MAP-Elites with Diversifier) work with the parallel retry and the
  advanced retry (smart boundary management) (1.5.8).
- Quality diversity: the descriptor bounds parameter of `mapelites` and
  `diversifier` is `qd_bounds` (was `desc_bounds`); per-niche statistics
  are optional (`use_stats`).
- The dependency `sklearn` is replaced by `scikit-learn` (1.5.5);
  `threadpoolctl` is a new dependency.

## [1.4.x] - 2022-11-10

Releases 1.4.0 (2022-07-14) to 1.4.9 (2022-11-10).

- Unified ask/tell interface, useful for monitoring and parallel fitness
  evaluation: added to `crfmnes`, `crfmnescpp`, `cmaescpp`, `modecpp`,
  `pgpecpp` and `decpp`, in addition to `cmaes`, `de` and `mode`.
- PGPE (Parameter Exploring Policy Gradients) in C++, `pgpecpp` (1.4.7).
- Quality diversity: CVT MAP-Elites with a CMA-ES emitter, `mapelites`
  (1.4.8), and the Diversifier meta-algorithm, `diversifier` (1.4.9), a
  generalized variant of CMA-ME for other wrapped optimizers. Also "drill
  down" for specified niches, transfer in both directions between the QD
  archive and the `advretry` store, parallel optimization on all CPU cores
  and per-niche statistics (mean, standard deviation, maximum, minimum,
  count).

## Earlier versions

Releases 0.9.5 (2020-02-18) to 1.3.37 (2022-07-11) are not described here.
Among them, 1.3.1 added the multi-objective optimizer MODE (`mode`; the C++
`modecpp` followed in 1.3.4) and 1.3.25 added CR-FM-NES (`crfmnes`,
`crfmnescpp`). See the git history and the tags `v1.0.7` to `v1.3.37`; not
every PyPI release has a tag.

[Unreleased]: https://github.com/dietmarwo/fast-cma-es/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/dietmarwo/fast-cma-es/compare/v2.0.3...v2.1.0
[2.0.3]: https://github.com/dietmarwo/fast-cma-es/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/dietmarwo/fast-cma-es/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/dietmarwo/fast-cma-es/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/dietmarwo/fast-cma-es/compare/v1.6.12...v2.0.0
[1.6.x]: https://github.com/dietmarwo/fast-cma-es/compare/v1.5.11...v1.6.12
[1.5.x]: https://github.com/dietmarwo/fast-cma-es/compare/v1.4.9...v1.5.11
[1.4.x]: https://github.com/dietmarwo/fast-cma-es/compare/v1.3.37...v1.4.9
