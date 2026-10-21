# fcmaes
A Python 3 gradient-free optimization library.

- [Documentation book](https://dietmarwo.github.io/fast-cma-es/)
- [Full README](https://github.com/dietmarwo/fast-cma-es/blob/master/README.adoc)
- [Tutorials](https://dietmarwo.github.io/fast-cma-es/tutorials/)
- [Python API](https://dietmarwo.github.io/fast-cma-es/fcmaes_documentation.html)
- [C++ API](https://dietmarwo.github.io/fast-cma-es/cpp_documentation.html)
- [Examples API](https://dietmarwo.github.io/fast-cma-es/examples_documentation.html)
- [Native Rust implementation](https://github.com/dietmarwo/fcmaes-rust) (with PyO3 API for Python users)
- [Optimization jobs](https://github.com/dietmarwo/optimization-jobs): durable jobs and an MCP server for AI agents, with fcmaes-rust workers
- [AI agent context](https://github.com/dietmarwo/fast-cma-es/blob/master/ai-context.md): how an AI agent chooses, configures and validates fcmaes optimizers ([AGENTS.md](https://github.com/dietmarwo/fast-cma-es/blob/master/AGENTS.md) for coding agents)




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
