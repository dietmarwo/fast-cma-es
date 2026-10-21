# AI context for solving user optimization problems with fcmaes

Last reviewed 2026-10-20 against `master` at `a028f25` (v2.0.3 plus the MODE
fixes of #31 and the exact 2-D Pareto archive of #32; the PyPI 2.0.3 release
predates both). In another checkout, check `pyproject.toml`, the module
signatures, and `tests` before relying on a claim.

This file is operational context for an AI that must design and implement an
optimization solution with this repository. It is not a claim that one
optimizer is universally best. Use the problem structure, evaluation cost, and
the user's desired output to choose a small set of defensible candidates, then
compare them under the same evaluation and worker budget.

If the solver is not chosen yet, start with the
[entry point](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)
in optimization-jobs. It routes between fast-cma-es, fcmaes-rust,
LP-IP-craft and SQP-craft and covers installation, the job API and the MCP
server.

The repository provides the `fcmaes` Python package. It combines Python
optimizer and orchestration code with a C++17 backend exposed through nanobind.
The modules whose names end in `cpp`, and `pareto`, use that native backend;
modules such as `de.py`, `cmaes.py`, and `mode.py` contain Python
implementations. All optimizers minimize.

The public distribution and import name are the same:

```text
python -m pip install fcmaes
python -c "import fcmaes"
```

The published package requires Python 3.10 or newer. Release wheels target
CPython 3.10 through 3.13 on Linux x86-64, Windows AMD64, and current Intel and
Apple Silicon macOS runners. Installing a matching wheel needs no compiler.
A source build needs a C++17 compiler, CMake, nanobind, and Eigen 3.4; the
CMake option `FCMAES_FETCH_EIGEN=ON` downloads Eigen instead. Repository
`examples`, `tutorials`, documentation sources, scripts, and tests are
intentionally excluded from the source distribution. Until a release after
2.0.3, `fcmaes.pareto` and the corrected MODE need a source build of `master`:
`python -m pip install git+https://github.com/dietmarwo/fast-cma-es`, adding
`--config-settings=cmake.define.FCMAES_FETCH_EIGEN=ON` without a local Eigen.

Use public Python wrappers such as `fcmaes.decpp.minimize` rather than calling
private extension symbols directly. Return values from scalar minimizers are
normally `scipy.optimize.OptimizeResult` objects with `x`, `fun`, `nfev`,
`nit`, `status`, and `success` fields. Multi-objective and quality-diversity
entry points return arrays or archives as documented by their modules.

## Required workflow for the AI

Before selecting an algorithm, create a problem card containing these facts.
Inspect the user's code and data where possible. Ask only for facts that cannot
be inferred and would materially change the solution.

| Question | Why it matters |
|---|---|
| What is the decision dimension? | Population size, covariance cost, and useful budgets depend strongly on dimension. |
| What is each variable's lower and upper bound? | Most global algorithms need a finite, meaningful search box. |
| Which variables are continuous, integer, ordinal, categorical, or structured? | DE and MODE have integer support, but categorical and structured variables still need decoding or repair. |
| Is there one objective, several objectives, or a quality-diversity archive? | This selects scalar optimization, MODE/weighted retry, or MAP-Elites. |
| Which directions are optimized? | Convert maximization to minimization by negating the value. |
| What are the constraints and feasibility convention? | MODE and `moretry` require constraints after objectives and treat `g(x) <= 0` as feasible. |
| Is a good initial point known? | CMA-ES, CR-FM-NES, PGPE, and local refinement benefit from it. |
| Is the objective smooth, discontinuous, noisy, stochastic, or multimodal? | Distribution search, restart strategy, and stopping tolerances differ. |
| For a stochastic objective, can simulations accept explicit seeds? | Common random numbers make comparisons fair; disjoint seeds are needed for validation. |
| How long does one evaluation take, and is it safe to run in another process? | This determines inner population evaluation versus outer retry parallelism. |
| What is the evaluation, wall-time, memory, and core budget? | Evaluation counts are the fairest first comparison unit. |
| Is a target value known? | Set `stop_fitness`; do not confuse it with the store filter `value_limit`. |
| Does the user need one solution, a Pareto front, or diverse behavior niches? | The requested output is part of the algorithm choice. |

Do not proceed with arbitrary placeholder bounds. If bounds are genuinely
unknown, derive physically meaningful limits or transform the variables. An
unnecessarily wide search box can cost orders of magnitude more evaluations.

For every completed solution, report the problem card, algorithm and rejected
alternatives, objective and constraint convention, every non-default
parameter, evaluation and process budgets, random seeds, validation method,
and quality statistics. Code without this configuration record is not a
reproducible optimization solution.

## First decide whether fcmaes is appropriate

Do not choose a gradient-free optimizer merely because the model is a
simulation. Prefer a gradient-based optimizer when all variables are
continuous, the complete objective is differentiable, reliable analytical,
automatic-differentiation, adjoint, or sensitivity gradients are available,
and the user needs one optimum or a smooth trade-off. Those gradients usually
carry much more information per evaluation.

fcmaes is a strong candidate when one or more of these conditions holds:

- gradients are unavailable, unreliable, or dominated by simulation noise;
- events, contact, clipping, repair, routing, integer choices, or categorical
  decoding make the complete objective nonsmooth;
- the landscape is multimodal and needs global exploration or many restarts;
- objectives are cheap enough that optimizer overhead and parallel retries
  matter;
- the result must cover behaviorally different niches instead of returning
  only one optimum; or
- robust maxima, quantiles, failures, or other outer logic make an otherwise
  smooth model nonsmooth.

Route evaluation-scarce problems to Bayesian or surrogate optimization when
only tens or hundreds of evaluations are affordable and a useful surrogate
can be fitted. Route exact linear, mixed-integer linear, routing, and
scheduling formulations to dedicated mathematical-programming solvers when
their structure can be expressed faithfully. Use fcmaes for structured
problems only when a defensible continuous encoding and deterministic repair
are part of the formulation. See "Related projects" for sibling solvers.
Before choosing a generic surrogate, check whether evaluations can be made
cheap (a library call instead of a command line, Numba, parallel evaluation)
or a domain-specific approximation exists, as in `tutorials/Surrogate.adoc`.

## Fast algorithm-selection decision tree

1. If valid and economical gradients exist for the complete map from decisions
   to objectives, start with a gradient-based method.
2. If the user wants the best solution in many behavior niches, use
   `mapelites.optimize_map_elites`. Add `diversifier.minimize` when local
   improvement emitters are useful.
3. If there are two or more competing objectives:
   - Use `modecpp.minimize` for one coordinated Pareto population, especially
     with explicit constraints or integer-decoded variables.
   - Use `modecpp.retry` for cheap objectives: parallel independent MODE
     runs joined into one front scale better per core than one population.
   - Use `moretry.minimize` when many independent weighted scalar runs are
     desirable or a DE/CMA/BiteOpt scalar pipeline is already effective,
     typically for two cheap objectives.
   - With exactly two objectives, record every evaluated batch in
     `pareto.ParetoArchive2D` to keep the exact front over all evaluations.
4. For one scalar objective:
   - Start with native BiteOpt for a difficult bounded, nonsmooth,
     multimodal, or poorly characterized black box.
   - Start with native DE for robust bounded exploration, integer-decoded
     variables, and DE-to-CMA sequences.
   - Use active CMA-ES when a useful guess exists, variables are continuous,
     and correlations or ill-conditioning matter.
   - Use CR-FM-NES for higher-dimensional continuous distribution search.
   - Use PGPE for very high-dimensional mirrored sampling, noisy policy
     search, or high-throughput batch objectives.
   - Use Dual Annealing for low-to-moderate-dimensional global exploration,
     optionally followed by its local search.
5. If local optima are likely, use independent `retry`. Compare
   `advretry` only when retained elites, crossover boxes, and increasing
   per-run budgets are plausible advantages.
6. Benchmark at least two plausible choices with identical bounds, objective,
   total evaluations, random protocol, and worker resources. A useful general
   comparison is BiteOpt retry versus DE followed by CMA-ES.

## Choose the integration surface

For normal Python use, prefer these native wrappers:

- `fcmaes.bitecpp.minimize`
- `fcmaes.decpp.minimize`
- `fcmaes.cmaescpp.minimize`
- `fcmaes.crfmnescpp.minimize`
- `fcmaes.pgpecpp.minimize`
- `fcmaes.dacpp.minimize`
- `fcmaes.modecpp.minimize`

The corresponding stateful `Bite_C`, `DE_C`, `ACMA_C`, `CRFMNES_C`,
`PGPE_C`, and `MODE_C` classes provide ask/tell integration. Use ask/tell when
the application owns batching, remote execution, accelerators, or a simulator
pool. Preserve candidate order between `ask()` and `tell()`.

The Python `de`, `cmaes`, `crfmnes`, and `mode` modules remain useful for
experimentation, inspection, custom modifiers, and comparisons. Do not mix
Python and C++ APIs just because class names look similar; verify the exact
signature in the current module or generated API documentation.

Use `fcmaes.optimizer` adapters with `retry` and `advretry`. `Sequence`
passes the best point from one optimizer to the next. `Choice` selects one
configured optimizer per run. Maintained constructors include `de_cma`,
`de_cma_py`, `da_cma`, `de_crfmnes`, `crfmnes_bite`, `bite_cma`, and
`cma_bite`.

The objective callback boundary is important:

- Cheap scalar Python callbacks do not become fast merely because the search
  loop is C++; each callback still crosses into Python and needs the GIL.
- `retry`, `advretry`, `moretry`, and the evaluator helpers use Python
  processes, bypassing the GIL but requiring serializable inputs and incurring
  inter-process communication.
- NumPy, SciPy, native extensions, external programs, and simulations may
  release the GIL or own their own thread pools. Measure the complete topology.
- Ask/tell is the safest integration for a user-owned vectorized, distributed,
  or asynchronous evaluator.

## Algorithm comparison

| Algorithm | Best fit | Main limitation | Practical initial configuration |
|---|---|---|---|
| Native DE (`decpp`) | Bounded global search, discontinuities, integer decoding, restarts | Smooth local convergence can be slower than CMA-ES | Population 31, `F=0.5`, `CR=0.9`; compare a DE-to-CMA sequence |
| Active CMA-ES (`cmaescpp`) | Continuous correlated or ill-conditioned variables, refinement from a guess | Full covariance work becomes costly in high dimension | Population 31, normalized coordinates, sigma based on uncertainty |
| CR-FM-NES (`crfmnescpp`) | Medium/high-dimensional continuous search and complete batch evaluation | Needs a meaningful center and scalar sigma; not a discrete optimizer | Even population 32; parallelize expensive batches |
| PGPE (`pgpecpp`) | Very high-dimensional mirrored sampling, noisy objectives, policy search | Diagonal distribution does not learn a full covariance matrix | Even population 32; ranking enabled for one-shot `minimize` |
| Dual Annealing (`dacpp`) | Low/moderate-dimensional global exploration with optional refinement | No population ask/tell interface | Keep local search for deterministic smooth-enough objectives |
| BiteOpt (`bitecpp`) | Hard bounded black boxes, nonsmoothness, multimodality, little tuning | Constraints need penalties or repair; batched ask/tell changes feedback timing | Automatic population and `M=1` first |
| MODE (`modecpp`) | Constrained multi-objective Pareto search | Objective rows and constraint conventions must be exact | Population 64–200 first, NSGA-II update (DE update for `ints`), evaluate complete populations |
| Weighted MO retry (`moretry`) | Independent scalarizations with any scalar optimizer | Sensitive to objective scaling and weight ranges | Tens or hundreds of retries; normalize objectives |
| CVT MAP-Elites (`mapelites`) | Quality-diversity over meaningful descriptors | Requires fixed descriptor bounds and enough evaluations for coverage | Pilot with hundreds of niches before increasing capacity |
| Diversifier (`diversifier`) | CMA-ME-style local improvement of an archive | More expensive and parameter-rich than basic MAP-Elites | Seed an archive first; use a small solver set |

These are starting hypotheses. The objective landscape dominates generic
rules.

## Scalar optimizer parameters

All scalar objectives are minimized. Return a finite `float`. The wrappers
generally replace exceptions and non-finite values with a very large finite
value, but explicit, scaled rejection values are easier to diagnose.

Use `numpy.random.Generator(PCG64DXSM(seed))` for controlled runs. Avoid the
module signatures' already-constructed default generators in reproducibility-
critical code; supply a fresh generator explicitly.

### Native Differential Evolution

`decpp.minimize` defaults:

| Parameter | Default | Guidance |
|---|---:|---|
| `popsize` | 31 | Increase for broad multimodal exploration only if many generations still fit. |
| `max_evaluations` | 100,000 | Set from the real budget. |
| `keep` | 200 | Usually leave unchanged; controls temporal/history behavior. |
| `stop_fitness` | negative infinity | Set only for a genuine known target. |
| `f` / `cr` | 0.5 / 0.9 | Tune after population and budget. |
| `min_mutate` / `max_mutate` | 0.1 / 0.5 | Additional mutation range for integer-marked coordinates. |
| `min_sigma` | 0 | Usually leave at zero. |
| `workers` | 1 | More workers call Python objectives concurrently from the native path; benchmark carefully. |

Pass a Boolean `ints` mask for integer coordinates, but still round, clamp,
lookup, and repair authoritatively in the objective. Integer mutation does not
turn unrelated categorical indices into a meaningful metric space.

Without `x0`, DE initializes from the bounds. With `x0`, provide a compatible
`input_sigma` if local initialization is intended. `DE_C` exposes
`ask()`, `tell(ys)`, `population()`, and `result()`.

### Native active CMA-ES

`cmaescpp.minimize` defaults to population 31, 100,000 evaluations,
`input_sigma=0.3`, normalized coordinates, delayed update enabled,
`accuracy=1`, no finite target, and automatic history/update settings.

With `normalize=True`, the search coordinates are scaled relative to the
bounds. Choose sigma in that normalized optimizer space. A sigma around
0.05–0.3 is a useful first range only when it matches the real uncertainty in
the guess. With normalization disabled, sigma is in physical units.

Use `workers > 1` only after measuring. With `delayed_update=False`, the wrapper
uses a multiprocessing batch evaluator; with delayed update enabled, native
parallel callback behavior applies. Do not combine a full process pool with a
simulator or BLAS library already using all cores.

### CR-FM-NES

`crfmnescpp.minimize` defaults to an even population of 32, 100,000
evaluations, `input_sigma=0.3`, no finite target, bound-violation handling
enabled, `penalty_coef=1e5`, and `normalize=False`.

The `use_constraint_violation` option concerns box-bound handling, not
arbitrary user constraints. General constraints still require a penalty,
repair, MODE, or another constrained formulation. Supply a guess and use a
batch objective for expensive populations. Compare against CMA-ES around
dimension 100 rather than treating a dimension threshold as absolute.

### PGPE

`pgpecpp.minimize` defaults:

| Parameter | Default |
|---|---:|
| `popsize` | 32, expected to be even |
| `input_sigma` | 0.1 |
| `max_evaluations` | 100,000 |
| `lr_decay_steps` | 1,000 |
| `use_ranking` | true |
| `center_learning_rate` | 0.15 |
| `stdev_learning_rate` | 0.1 |
| `stdev_max_change` | 0.2 |
| `b1` / `b2` / `eps` | 0.9 / 0.999 / 1e-8 |
| `decay_coef` | 1.0 |
| `normalize` | true |

Keep ranking enabled for a robust first experiment. Increase the even
population for noisy objectives or more parallel work. Change learning rates
one at a time and validate across multiple seeds. Note that the low-level
`PGPE_C` constructor defaults `use_ranking=False`; set it explicitly when
reproducing the one-shot wrapper's behavior.

### Dual Annealing

`dacpp.minimize` defaults to 100,000 evaluations and local search enabled.
Use finite bounds. Local finite-difference refinement is inappropriate for
very noisy or sharply discontinuous objectives; disable `use_local_search` in
that case.

### BiteOpt

`bitecpp.minimize` defaults to automatic population (`popsize=0`), 100,000
evaluations, `M=1`, automatic stall criterion, and no finite target.

Start with `M=1`. Test a small deeper value only when multimodality and budget
justify multiple internal populations. `Bite_C` (since 2.0.2) provides batched
ask/tell with configurable `batch_size` (default 8); compare it against
ordinary one-at-a-time BiteOpt because delayed selector feedback can change
convergence. Use it to parallelize one run of an expensive objective; for
cheap objectives prefer sequential BiteOpt or parallel retry.

## Recommended scalar recipes

### Unknown bounded black box

Run two fair baselines:

1. `Bite_cpp` with automatic population and `M=1`.
2. A fixed-budget `Sequence([De_cpp(...), Cma_cpp(...)])`.

Set the DE and CMA evaluation allocations explicitly for a controlled
comparison. `optimizer.de_cma` randomly assigns between 10% and 50% of the
total budget to DE when explicit stage budgets are omitted, so it is
convenient for retry diversity but not a fixed-protocol benchmark.

### Known good continuous guess

Use normalized active CMA-ES first. Compare CR-FM-NES at larger dimensions and
PGPE when diagonal mirrored search matches the problem. Derive per-coordinate
uncertainty from the model rather than copying one sigma blindly.

### Expensive objective

Prefer one population with parallel or vectorized evaluation when every
generation is expensive and useful. Choose a population at least as large as
the worker count. If independent global restarts matter more, use outer retry
processes and keep inner workers at one.

If the simulator owns a process or thread pool, compare two explicit
topologies: many serial simulator evaluations under fcmaes versus fewer
fcmaes evaluations using the simulator's internal parallelism. Keep the total
core allowance equal.

For an external program (a CFD or FEM solver, a legacy executable), start
from [`examples/external_simulator.py`](examples/external_simulator.py),
which `tests/test_external_simulator.py` tests:

- **Runs.** The program runs in a fresh working directory, removed afterwards
  (kept on failure with `keep_failed=True`). It runs in a process group of its
  own: when it ends, times out or is interrupted, the whole group is stopped
  (SIGTERM, then SIGKILL), solver processes it started included. Children that
  leave the group (`setsid`, daemons) need a container or a systemd scope.
- **Failures.** A timeout, a nonzero exit, a missing output or a non-finite
  value make the evaluation fail. The objective then returns a finite
  `failure_value`, worse than every real result. Every evaluation is recorded
  in a ledger shared by all worker processes. `summary()` in the parent
  counts successes and failures by kind, and `check_result` refuses a
  campaign with no successful evaluation, or a best point whose fresh
  recheck fails. Report these counts with the result: a finite best value can
  be a failure penalty.
- **Seeds.** By default the seed is derived from the point, which replays one
  point exactly but is not common random numbers. For a stochastic simulator,
  pass `seeds=` (training seeds from `replication_seeds`): every candidate
  then runs on the same seeds, aggregated. Validate the result on the
  disjoint holdout seeds with `with_seeds(holdout)`.
- **Threads.** `OMP_NUM_THREADS` and the other thread-pool variables are set
  to 1, overwriting inherited values. Set them explicitly in `env` for a
  multithreaded simulator, with fewer workers. `topology(workers)` records
  the layout.
- **Parallel retry.** The object is picklable for
  `retry.minimize(..., workers=...)`.

### Noisy objective

Use PGPE ranking, larger populations, or repeated evaluations. Do not use a
tight `stop_fitness` based on one noisy observation. If each objective call
averages replications, report both optimizer calls and the true simulation
count.

Use the same fixed training seed set for every candidate. Re-evaluate
finalists, Pareto points, and archive elites on a disjoint holdout seed set.
For noisy QD, validate both quality and descriptor location because an elite
can move to another niche.

## Constraints and variable encoding

Use these conventions consistently:

- All objectives are minimized. Negate quantities that must be maximized.
- MODE and `moretry` rows are `[objective_0, ..., objective_m,
  constraint_0, ..., constraint_k]`.
- A constraint is feasible when `constraint <= 0`.
- Convert `h(x) = 0` to `abs(h(x)) - tolerance <= 0` when a tolerance is
  meaningful. Since #31, MODE counts a row as feasible only when every
  constraint is `<= 0`; the former adaptive tolerance for small violations is
  gone, so equality constraints need this explicit tolerance.
- For scalar optimizers, use decoding or repair for hard structure and a
  scaled penalty for remaining violations. A useful starting form is
  `objective + rho * sum(max(0, g_i(x))**2)` after normalizing terms.
  `tutorials/Constraints.adoc` instead adds a constant plus the violation for
  each violated constraint, a nonsmooth step these optimizers tolerate.
- Return a large finite rejection value for invalid simulations.
- Encode a bounded integer as a real coordinate, round it, and clamp it in the
  objective; optionally also set DE or MODE's integer mask. MODE ignores
  `ints` unless `nsga_update=False`.
- Encode a category as an index into a fixed list. Do not interpret distances
  between unrelated categories physically.
- Decode random keys or repair permutations, schedules, subsets, and mutually
  exclusive choices deterministically.

The `JobShop`, `Scheduling`, `Routing`, `Subset`, `UAV`, `FluidDynamics`, and
`MINLP` tutorials are repository-specific encoding references. Test every
decoder independently before optimization.

If a scalar penalty dominates everywhere, normalize the base objective and
constraint residuals before tuning `rho`. Too small lets infeasible points
win; too large creates a nearly flat rejection landscape.

## Multi-objective optimization

### MODE

Use `modecpp.minimize` for a coordinated Pareto population. The objective must
return exactly `nobj + ncon` finite values, with constraints trailing the
objectives. The defaults are population 64, `F=0.5`, `CR=0.9`, NSGA-II update,
`pro_c=0.5`, `dis_c=15`, `pro_m=0.9`, `dis_m=20`, and integer mutation range
0.1–0.5.

Guidance:

- Start with NSGA-II update, then compare `nsga_update=False` (DE update) in
  longer runs. Only the DE update uses the `ints` mask.
- Use population 64 to about 200 in the first experiment; try up to 512 later
  for difficult fronts, cheap objectives, or more objectives while preserving
  many generations.
- Scale objective magnitudes before judging front quality.
- Use `workers > 1` for expensive objective populations, not for tiny Python
  callbacks.
- Use `MODE_C.ask()` and `tell(ys)` when the application owns evaluation.
- `minimize` returns the last evaluated generation, not all evaluations.
  Select feasible rows, then non-dominated objective rows with
  `moretry.pareto(xs[ok], ys[ok, :nobj])`. `mode.feasible` tolerates a summed
  violation below `eps=1e-2` and returns all rows when none is feasible.
- `modecpp.retry` runs independent MODE processes and can consume
  approximately `num_retries * max_evaluations`; set both explicitly. Its
  defaults differ from `minimize`: 64 retries and `nsga_update=False`.
- #31 fixed SBX expansion, polynomial mutation (native), non-dominated
  sorting, crowding, and feasibility-first ranking in `modecpp` and `mode`.
  Rerun MODE baselines from 2.0.3 or earlier instead of comparing to them.

### Exact two-objective archive

`pareto.ParetoArchive2D(reference, payload_dim=0)` keeps every unique
non-dominated pair of two minimized objectives inside the reference box, with
an optional payload such as `x`, and updates the exact hypervolume on every
insertion. It only observes evaluations, so it runs alongside a `MODE_C`
ask/tell loop or any optimizer whose evaluations the application sees. Use it
when the report needs the front over all evaluations, not the final
population, or an exact hypervolume trace. Pass only feasible rows and the two
objective columns; points above the reference are skipped (`add` returns
`False`) and non-finite values raise `ValueError`. It is unbounded and not
shared between processes, so collect evaluations in the parent. See
`docs/pareto_archive.md` and `examples/pareto_mode.py`.

### Weighted multi-objective retry

`moretry.minimize` samples one weight per output value inside
`weight_bounds`, takes the `value_exp`-norm of the weighted objectives and
constraint violations `max(c, 0)`, adds the weight of each violated trailing
constraint, runs a scalar optimizer, and then re-evaluates retained points
using the original multi-objective function. With `ncon > 0`, `weight_bounds`
needs `nobj + ncon` entries; a satisfied constraint adds nothing, so a large
constraint weight enforces feasibility without distorting the objective
trade-off. `value_limits` may list only the objectives. MODE remains the
better choice for one coordinated, constrained Pareto population.

Use it when independent scalar runs are effective and easy to parallelize.
Set `ncon` exactly. Normalize or shift objective values to comparable,
preferably nonnegative scales before choosing weight bounds. With non-integer
`value_exp`, negative weighted values can be mathematically invalid. Report
the retained unscalarized objective vectors, not the scalar retry score.

## Quality diversity

Use MAP-Elites only when behavior descriptors express diversity the user
actually values. A descriptor is not another minimized objective.

The QD fitness contract is:

```python
fitness, descriptor = qd_fitness(x)
```

The descriptor must match `qd_bounds`. A typical workflow is:

1. Define decision and descriptor bounds.
2. Create or let `optimize_map_elites` create an archive.
3. Seed the archive with evaluated candidates.
4. Run MAP-Elites variation and optional CMA emitters.
5. Optionally pass the archive to `diversifier.minimize`.
6. Report occupied niches, coverage, best fitness, descriptor ranges, and
   representative elites.

`mapelites.optimize_map_elites` defaults to 4,000 niches, 20 CVT samples per
niche, all CPU processes, and 100 iterations. Those defaults can be expensive:
pilot with fewer niches and a measured evaluation budget. In two descriptor
dimensions, inspect whether grid indexing (`samples_per_niche=0`) or CVT
centers fit the intended geometry. Capacity controls both resolution and
memory.

`diversifier.minimize` accepts one solver configuration or a sequence through
`opt_params`. Supported maintained solver names include `CMA`, `CMA_CPP`,
`CRMFNES`, `CRMFNES_CPP`, `DE`, `DE_CPP`, `PGPE`, `BITE_CPP`, and `elites`.
Start with a small, explicit configuration rather than relying on a large
mixed default. Keep archive updates and stochastic validation reproducible.

## Retry selection and parameters

### Basic retry

`retry.minimize` defaults to 1,024 retries, one process per CPU, a retained
capacity of 500, 50,000 evaluations per run, no value filter, and no finite
stop target. Its default optimizer is a DE-to-CMA sequence.

Set these explicitly in production:

- `num_retries`: number of independent runs;
- `workers`: number of outer processes;
- the optimizer's per-run `max_evaluations`;
- `capacity`: number of retained distinct results;
- `value_limit`: strict upper filter for stored results;
- `stop_fitness`: target that stops new work;
- a reproducible random protocol; and
- `statistic_num` if improvement history is required.

Configured work is roughly `num_retries * evaluations_per_run`, reduced by
early termination. `popsize` and `max_evaluations` on `retry.minimize` are
ignored when a custom `optimizer` object supplies its own settings.

### Advanced retry

`advretry.minimize` coordinates restarts through a shared result store,
crossover boxes, and a budget that increases with retry progress. Defaults are
5,000 retries, 1,500 initial evaluations, check interval 100, retained
capacity 500, and an automatically capped final budget factor up to 50.

Use it only after a basic retry baseline. `value_limit` matters more here:
retaining many poor parents can make crossover regions unhelpful. Optimizer
adapters receive the store-derived local bounds, guess, standard deviations,
and evaluation budget through their normal `minimize` interface. A custom
adapter that ignores those values defeats advanced retry.

`datafile` can persist and reload advanced-retry state. Treat it as mutable
run state, not source data, and record the code version and problem definition
with it.

## Parallelism rules

There are three possible levels:

1. Outer process parallelism: `retry`, `advretry`, `moretry`,
   `modecpp.retry`, MAP-Elites, and Diversifier.
2. Inner population evaluation: `workers` on supported scalar/MODE wrappers,
   `evaluator.parallel`, `parallel_mo`, or application-owned ask/tell batches.
3. Parallelism inside NumPy, BLAS, a simulator, an external program, or a
   machine-learning framework.

Count the CPUs the process may actually use, not the machine:
`os.cpu_count()` and `multiprocessing.cpu_count()` ignore the affinity mask
(`taskset`, cpusets) and container or systemd CPU quotas. `usable_cpus()` in
[`examples/external_simulator.py`](examples/external_simulator.py) combines
`os.sched_getaffinity(0)` with the cgroup quota (v2 `cpu.max` or v1
`cpu.cfs_quota_us`). Workers times the threads of each evaluation should stay
within it. Then lower the worker
count until the memory of all concurrent evaluations (model data copied into
each process, simulator working sets) fits the available memory.

Do not enable every level at the full core count. For 16 cores, compare:

- 16 retry processes and one inner worker;
- one optimizer with a 16-worker expensive population;
- four retry processes with four inner workers only if both levels have
  enough work; or
- one fcmaes worker when the simulator already owns 16 threads.

Process objectives and their captured data must be serializable under the
platform's multiprocessing start method. Put process-launching code behind
`if __name__ == "__main__":`, especially on Windows and macOS. Do not rely on
fork-only global state.

The evaluator uses persistent processes and must be stopped. Use `try/finally`
or a wrapper that stops it. fcmaes limits BLAS threads in several retry worker
paths, but the application is still responsible for avoiding nested pools in
other libraries.

Multi-process scheduling can change result arrival order. Treat stochastic
parallel retry as statistically reproducible unless the exact scheduling and
seed-to-run assignment are explicitly controlled. Ask/tell with ordered,
deterministic batch results is the strongest reproducibility path.

## Budget and parameter-setting procedure

Tune in this order:

1. Validate objective values, signs, constraints, decoding, and bounds.
2. Transform variables with extreme scale differences. Use log coordinates
   for positive variables spanning orders of magnitude.
3. Choose the algorithm family from the requested result and problem
   structure.
4. Set the total evaluation and worker budget.
5. Set population size so several generations fit and workers receive work.
6. Set initialization and sigma from real uncertainty.
7. Only then tune mutation, crossover, or learning rates.

Use a staged budget:

- Smoke test: a few populations to exercise decoding, invalid paths, and
  reporting.
- Pilot: several seeds for two or three candidate algorithms.
- Production: use the measured wall-time/evaluation budget for the most robust
  configuration and retain disjoint validation seeds.

Population algorithms may check limits at generation boundaries, so actual
evaluations can exceed the requested limit by part of a population. Compare
the returned `nfev`, not only configured limits.

## Minimal implementation patterns

### Bounded scalar native DE

```python
import numpy as np
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds
from fcmaes import decpp

bounds = Bounds([-5.0] * 8, [5.0] * 8)

def objective(x):
    return float(np.dot(x, x))

result = decpp.minimize(
    objective,
    bounds=bounds,
    popsize=31,
    max_evaluations=50_000,
    rg=Generator(PCG64DXSM(1)),
)
assert result.success and np.isfinite(result.fun)
```

### Reproducible ask/tell population

```python
import numpy as np
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds
from fcmaes.decpp import DE_C

bounds = Bounds([-5.0] * 4, [5.0] * 4)
solver = DE_C(
    dim=4,
    bounds=bounds,
    popsize=32,
    rg=Generator(PCG64DXSM(2)),
)

for _ in range(100):
    xs = solver.ask()
    ys = np.asarray([np.dot(x, x) for x in xs], dtype=np.float64)
    if solver.tell(ys) != 0:
        break

result = solver.result()
```

### Parallel retry

```python
import multiprocessing as mp
import numpy as np
from scipy.optimize import Bounds
from fcmaes import retry
from fcmaes.optimizer import Bite_cpp

def objective(x):
    return float(np.dot(x, x))

if __name__ == "__main__":
    result = retry.minimize(
        objective,
        Bounds([-5.0] * 6, [5.0] * 6),
        optimizer=Bite_cpp(max_evaluations=20_000),
        num_retries=32,
        workers=min(8, mp.cpu_count()),
        capacity=32,
    )
    print(result.fun, result.x, result.nfev)
```

### Constrained native MODE

```python
import numpy as np
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds
from fcmaes import modecpp, moretry

def objectives_and_constraint(x):
    return np.asarray([
        np.dot(x, x),
        np.dot(x - 1.0, x - 1.0),
        x.sum() - 2.5,  # feasible when <= 0
    ])

xs, ys = modecpp.minimize(
    objectives_and_constraint,
    nobj=2,
    ncon=1,
    bounds=Bounds([0.0] * 4, [1.0] * 4),
    popsize=128,
    max_evaluations=50_000,
    workers=1,
    rg=Generator(PCG64DXSM(3)),
)
feasible = ys[:, -1] <= 0
front_x, front_y = moretry.pareto(xs[feasible], ys[feasible, :2])
```

For optimizer sequences and advanced retry, copy the maintained adapters from
`fcmaes/optimizer.py` instead of recreating store and budget handling ad hoc.

## Validation and reporting checklist

Before declaring success:

- Unit-test decoding, objective signs, constraint signs, bounds, and known
  reference points.
- Confirm every final point is decoded and re-evaluated independently.
- Run the tests with the selected interpreter: `python -m pytest -q tests`,
  as the wheel CI does (wheel smoke, MODE regression, and Pareto archive
  tests). Against the PyPI 2.0.3 wheel, only `tests/test_wheel_smoke.py`
  applies.
- Run relevant module and example tests; the historical `fcmaes/test_cma.py`
  suite is stochastic and comparatively expensive, so report retries or
  failures honestly.
- Record algorithm, every non-default parameter, bounds, generator seed,
  workers, configured budget, actual evaluations, and wall time.
- Compare algorithms using the same objective calls and core allowance.
- Use multiple seeds and report best, median or mean, dispersion, and
  success/feasibility rate.
- For multi-objective runs, report feasible Pareto points and an appropriate
  front-quality measure, not only one selected point. For two objectives,
  `ParetoArchive2D.hypervolume()` is exact; report its reference point.
- For QD runs, report capacity, occupied niches, coverage, best fitness,
  descriptor ranges, and holdout niche migration for noisy problems.
- Test scaling with workers 1 and N before claiming parallel speedup.
- Preserve a small deterministic smoke configuration.

## Failure diagnosis

| Symptom | Likely cause | First action |
|---|---|---|
| No improvement from initialization | Wrong sign, invalid decoding, flat penalty, or bounds too broad | Print and test decoded points and every objective term. |
| All results are infeasible | Constraint sign error or badly scaled violation | Verify feasible means `<= 0` and test a known feasible point. |
| CMA-ES immediately hits bounds | Guess or sigma uses the wrong coordinate scale | Check `normalize` and rescale sigma. |
| DE explores but does not refine | Population is too large for the budget or local smooth convergence is needed | Reduce population or hand the best point to CMA-ES. |
| Parallel execution is slower | Objective is cheap, IPC dominates, or pools are nested | Use one worker level and benchmark a larger batch. |
| Multiprocessing hangs or recursively starts | Missing main guard or unpicklable objective state | Add `if __name__ == "__main__":` and isolate process-safe data. |
| Front covers one extreme | Poor objective scaling, population, or weight bounds | Normalize objectives and increase MODE population or scalarization diversity. |
| MAP-Elites coverage stays low | Wrong descriptor bounds, too many niches, or invalid descriptors | Inspect descriptor distributions and reduce pilot capacity. |
| Noisy runs stop inconsistently | Target or tolerance is tighter than the noise | Increase replication/population and validate on disjoint seeds. |
| Native wrapper returns `success=False` with a huge value | Input shape/type error or an exception hidden by the defensive wrapper | Reproduce with a tiny call, validate contiguous numeric arrays, and inspect the underlying exception path. |

## Related projects

- [fcmaes-rust](https://github.com/dietmarwo/fcmaes-rust) is an independent
  pure-Rust reimplementation, not a wrapper of this backend, with a PyO3
  binding (`pip install fcmaes-rust`, import `fcmaes_rust`). Its README
  recommends this package for objectives written in Python and the binding for
  objectives in Rust. It adds CMA-ES IPOP/BIPOP restarts, CMA-MAE emitters, an
  experimental source-only GP surrogate for expensive simulations, and
  job-API workers. CR-FM-NES, PGPE, MODE, CVT MAP-Elites with a CMA-ES
  emitter, the CMA-ME-style `diversifier`, and `ParetoArchive2D` exist here.
- [optimization-jobs](https://github.com/dietmarwo/optimization-jobs) holds
  the shared job API, the `optimization-run` runner, and the `optimization-mcp`
  MCP server through which agents run durable jobs on fcmaes-rust,
  LP-IP-craft, and SQP-craft workers. This package has no job worker. A
  Python objective therefore runs in process. To make a long run outlive the
  agent's session, start the script in a session of its own
  (`setsid python run.py > run.log 2>&1 &`). Have it write its parameters,
  seeds and results to a file as it goes, then read that file. The job API's
  budgets, journals, cancellation and recovery need the objective as a Rust
  model in an fcmaes-rust application worker.
- LP and MIP models belong in
  [LP-IP-craft](https://github.com/dietmarwo/LP-IP-craft); smooth constrained
  NLPs with derivatives (at most 256 variables) in
  [SQP-craft](https://github.com/dietmarwo/SQP-craft).

## Repository references

- `README.adoc`: installation, algorithm overview, release behavior, and
  documentation entry points.
- `fcmaes/optimizer.py`: retry adapters, sequences, choices, and maintained
  scalar recipes.
- `fcmaes/retry.py`: independent parallel retry and result storage.
- `fcmaes/advretry.py`: coordinated retry, crossover, adaptive budgets, and
  persisted state.
- `fcmaes/moretry.py`: weighted multi-objective retry and Pareto filtering.
- `fcmaes/modecpp.py`: native constrained MODE wrapper and ask/tell state.
- `fcmaes/mode.py`: Python MODE implementation, modifiers, feasibility, and
  Pareto utilities.
- `fcmaes/pareto.py`, `docs/pareto_archive.md`, and `examples/pareto_mode.py`:
  exact 2-D Pareto archive, its semantics, and a constrained MODE recording.
- `fcmaes/mapelites.py`: CVT MAP-Elites archive and emitters.
- `fcmaes/diversifier.py`: archive improvement with scalar solvers.
- `fcmaes/evaluator.py`: persistent multiprocessing evaluators and callback
  safety behavior.
- `fcmaes/_fcmaes_ext.pyi`: native extension signatures.
- `tests/test_wheel_smoke.py`: deterministic installed-wheel smoke coverage.
- `tests/test_mode_regression.py`, `tests/test_pareto.py`, and `tests/native`:
  MODE regression, Pareto archive, and standalone C++ archive tests.
- `tutorials/README.adoc`: tutorial index and retry output explanation.
- `tutorials/Introduction.adoc`: practical optimizer selection and trajectory
  examples.
- `tutorials/Performance.adoc`: performance comparisons.
- `tutorials/Constraints.adoc`, `tutorials/MODE.adoc`, and
  `tutorials/MultiObjective.adoc`: constraint and Pareto formulations.
- `tutorials/MapElites.adoc` and `tutorials/Diversity.adoc`: quality-diversity
  workflows.
- `tutorials/FluidDynamics.adoc`, `tutorials/MINLP.adoc`,
  `tutorials/JobShop.adoc`, `tutorials/Scheduling.adoc`,
  `tutorials/Routing.adoc`, `tutorials/Subset.adoc`, and
  `tutorials/UAV.adoc`: mixed and structured encodings.
- Generated API documentation (built before #32, without `fcmaes.pareto`):
  `https://dietmarwo.github.io/fast-cma-es/fcmaes_documentation.html`.
- Local documentation build: `python scripts/build_docs.py` (needs
  Asciidoctor).

When this guide and the implementation disagree, treat the current public
Python API, native type stub, and tests as authoritative. Update this file with
the implementation and record why the behavior changed.
