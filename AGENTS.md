# AGENTS.md

This repository is **fcmaes**, a Python 3.10+ library of gradient-free
optimizers with a C++17 backend exposed through nanobind:

- scalar: BiteOpt, DE, CMA-ES, CR-FM-NES, PGPE, Dual Annealing, and parallel
  retry;
- multi-objective: MODE and the exact two-objective archive `fcmaes.pareto`;
- quality diversity: MAP-Elites and the Diversifier.

## Read first

- [`ai-context.md`](ai-context.md) shows how to choose an optimizer for a
  user's problem, encode constraints and variables, set budgets and
  parallelism, and validate and report results.
- When the solver is not chosen yet, start at the cross-repository entry
  point,
  [optimization-jobs `ai-context.md`](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md).
  Linear and mixed-integer models go to LP-IP-craft, smooth NLPs with
  derivatives to SQP-craft, and Rust objectives or durable agent jobs to
  fcmaes-rust.
- This package has no job worker or MCP server. A Python objective runs in
  process. A long run that must outlive the agent's session runs as a
  detached script (see `ai-context.md`, "Related projects").

## Build and test

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install . --config-settings=cmake.define.FCMAES_FETCH_EIGEN=ON
python -m pip install pytest                     # the tests' only extra dependency
cd .. && python -m pytest fast-cma-es/tests -q   # outside the checkout, so the installed extension is imported
```

Native archive tests (CMake, GCC or Clang):

```bash
cmake -S tests/native -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel && ctest --test-dir build-native --output-on-failure
```

Documentation site: `python scripts/build_docs.py` (needs Asciidoctor 2.0.23+).

## Conventions

- All optimizers minimize. Constraints are feasible at `g(x) <= 0`.
- The public API is the Python wrappers (`fcmaes.bitecpp`, `decpp`,
  `cmaescpp`, `modecpp`, ...). Do not call private extension symbols.
- Wheel CI runs `tests/` on every installed wheel, and the sdist job does the
  same. An example that a test imports must also be listed in
  `[tool.cibuildwheel] test-sources` in `pyproject.toml` and copied in the
  sdist job of `.github/workflows/wheels.yml`.
- Record user-visible changes in `CHANGELOG.md` under `[Unreleased]`.
  [`RELEASING.md`](RELEASING.md) describes releases. Do not tag or publish.
