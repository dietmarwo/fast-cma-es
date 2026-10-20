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




## Exact 2D Pareto archive

`fcmaes.pareto.ParetoArchive2D` keeps all nondominated evaluations of a
two-objective problem and updates their hypervolume with every insertion.
Its native code is derived from `bitehv2.h` in Aleksey Vaneev's
[BiteOpt](https://github.com/avaneev/biteopt) repository. See
[usage and a MODE example](docs/pareto_archive.md).
