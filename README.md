<p align="center">
  <img src="https://raw.githubusercontent.com/dietmarwo/fast-cma-es/master/tutorials/img/fast-cma-es.png" alt="fast-cma-es: gradient-free optimization for Python">
</p>

# fcmaes

A Python 3 gradient-free optimization library.

**Choosing an optimization tool?** Start with the
[shared solver-selection and agent guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md) or the
[optimization ecosystem overview](https://dietmarwo.github.io/optimization-jobs/).
Use `fcmaes` for Python black-box objectives, parallel retry, Pareto search
and quality diversity. Durable MCP jobs use the separate Rust workers;
this Python package does not include an MCP server.

- [Documentation book](https://dietmarwo.github.io/fast-cma-es/)
- [Full README](https://github.com/dietmarwo/fast-cma-es/blob/master/README.adoc)
- [Tutorials](https://dietmarwo.github.io/fast-cma-es/tutorials/)
- [Python API](https://dietmarwo.github.io/fast-cma-es/fcmaes_documentation.html)
- [C++ API](https://dietmarwo.github.io/fast-cma-es/cpp_documentation.html)
- [Examples API](https://dietmarwo.github.io/fast-cma-es/examples_documentation.html)
- [Native Rust implementation](https://github.com/dietmarwo/fcmaes-rust) (with PyO3 API for Python users)
- [Optimization jobs](https://github.com/dietmarwo/optimization-jobs): durable jobs and an MCP server for AI agents, with fcmaes-rust workers
- [experiment-craft](https://github.com/dietmarwo/experiment-craft): reproducible campaigns of those jobs (parameter sweeps, dependent steps, budgets and reports); it drives the Rust workers, not this package, and is in early development
- [AI agent context](https://github.com/dietmarwo/fast-cma-es/blob/master/ai-context.md): how an AI agent chooses, configures and validates fcmaes optimizers ([AGENTS.md](https://github.com/dietmarwo/fast-cma-es/blob/master/AGENTS.md) for coding agents)




[Prompt-driven optimization tutorials](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic) demonstrate the shared agent workflow with native Rust application workers. Python objectives continue to use this package in process; these tutorials do not add a worker to fast-cma-es.

## Installation

```bash
python -m pip install fcmaes
```

To build from source you need a C++17 compiler and Python 3.10 or newer;
`pip` fetches the build tools, and Eigen is fetched if it is not installed:

```bash
git clone https://github.com/dietmarwo/fast-cma-es
cd fast-cma-es
python -m pip install . --config-settings=cmake.define.FCMAES_FETCH_EIGEN=ON
```

See the [full README](https://github.com/dietmarwo/fast-cma-es/blob/master/README.adoc)
for the requirements, the [changelog](https://github.com/dietmarwo/fast-cma-es/blob/master/CHANGELOG.md) for what changed, and
[RELEASING.md](https://github.com/dietmarwo/fast-cma-es/blob/master/RELEASING.md) for the release process.

## Exact 2D Pareto archive

`fcmaes.pareto.ParetoArchive2D` keeps all nondominated evaluations of a
two-objective problem and updates their hypervolume with every insertion.
Its native code is derived from `bitehv2.h` in Aleksey Vaneev's
[BiteOpt](https://github.com/avaneev/biteopt) repository. See
[usage and a MODE example](docs/pareto_archive.md).

## Third-party integration

[OptunaHub's FastCmaesSampler](https://hub.optuna.org/samplers/fcmaes/), by
Jacob Pfeil, connects fcmaes CMA-ES to Optuna. See its example and supported
Optuna version there. This is an external integration, not a comparative
benchmark of the five-project ecosystem.
