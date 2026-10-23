# Changelog

All notable changes to the `fcmaes` Python package are documented here, in
the format of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow the PyPI releases; since 2.0.0, pushing a tag `vX.Y.Z`
publishes that version to PyPI.

## [Unreleased]

### Added

- [Prompt-driven optimization tutorials](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic) demonstrate the shared agent workflow with native Rust application workers. Python objectives continue to use this package in process; these tutorials do not add a worker to fast-cma-es.

### Changed

- The READMEs open with the fast-cma-es banner (`tutorials/img/fast-cma-es.png`)
  instead of the animated logo; `README.md`, the PyPI description, links it by
  absolute URL. The tutorials keep `logo.gif`.

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
