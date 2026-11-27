# fast-cma-es C++ performance plan for the remaining algorithms

27 November 2026 · fcmaes 2.2.0 source at `2cea79f9223f2153655bb8a56934cb9a5452c5d2`

**Start with the Dual Annealing budget and CR-FM-NES incumbent.** Dual Annealing with local search spent 1,183–5,096 evaluations on Cassini1 with a requested cap of 1,024. CR-FM-NES returned an out-of-bounds point on a separate boundary fixture. Both affect the interpretation of fixed-budget quality. **The strongest scaling opportunity is ParetoArchiveND** on fronts where nearly every point survives. Doubling five-objective trade-off inputs from 2,048 to 4,096 raised median insertion time from 22.06 to 91.33 ms.

The plan contains **20 development iterations for each of five optimizers and each of five public support algorithms**, for 200 distinct proposals. The baseline measurements below were executed on the pinned revision. The 200 changes are experiments for a subsequent implementation campaign. [Commit](https://github.com/dietmarwo/fast-cma-es/commit/2cea79f9223f2153655bb8a56934cb9a5452c5d2) · [Prior six-algorithm campaign](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/docs/performance-120/RESULTS.md)

## Scope

| Native algorithm | Public Python entry point | Main question |
|---|---|---|
| CR-FM-NES | `crfmnescpp.minimize`, `CRFMNES_C` | How much time can be recovered in the distribution update after correcting physical incumbent reporting? |
| PGPE | `pgpecpp.minimize`, `PGPE_C` | Can mirrored sampling and gradient accumulation do less memory traffic? |
| Dual Annealing | `dacpp.minimize` | What does polishing buy when every finite-difference call is charged? |
| jSO | `jsocpp.minimize`, `Jso` | Which improvements speed mutation as the population shrinks? |
| ArchiveDE | `archivede.minimize`, `ArchiveDE` | Can rank-space survival and population copying cost less while preserving its exact archive? |
| Exact 2D Pareto front and incremental HV | `pareto.ParetoArchive2D` | Which paths pay off under high rejection versus all-admitted streams? |
| Strict N-D Pareto front | `pareto.ParetoArchiveND` | Where can dominance pruning change the growth rate? |
| Exact inverted generational distance (IGD) | `pareto.igd` | Which distance-search method wins for the given dimension and query count? |
| Borda rank-band box query | `rankindex.BordaBoxIndex` | Can the narrowest-axis scan skip more work on selective queries? |
| Sparse rank-grid box query | `rankindex.GridBoxIndex` | Can wide boxes return many IDs with less boundary scanning and copying? |

I interpreted “DE” as the standard `deoptimizer.cpp` backend. jSO and ArchiveDE belong to the DE family but expose separate native state, selection, and public APIs. The archive and index classes above are independently callable even when their ancestry is BiteOpt. The six exclusions are CMA-ES, BiteOpt, standard DE, MODE, Diversifier and MAP-Elites, including their internal-only strategies and grid lookup. ADAM is measured inside PGPE and the vendored L-BFGS-B inside Dual Annealing. GTOP mission functions serve as objective workloads. Retry, compositions and external optimizer adapters run at the Python layer. The inventory follows the root build target, module bindings and public wrappers. [Build manifest](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/CMakeLists.txt) · [Module registration](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/bindings/module.cpp) · [Front-tool bindings](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/bindings/front_bindings.cpp) · [Repository inventory](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/AGENTS.md)

## Measured baseline

The Release extension was built from this checkout with GCC 14.2, C++17, `-O3 -DNDEBUG`, Eigen 3.4.0 and nanobind 3.1.0. CPython 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 ran on a shared Linux AMD EPYC 9V74 host. Both timing drivers were pinned to logical CPU 3 and ran sequentially with one objective worker. Python BLAS/OpenMP thread limits were one. Each suite shuffled workload order per seed, warmed up on seed 100 and measured seeds 101–107. Tables show median elapsed milliseconds. Brackets are first and third quartiles across those seven runs. The spread includes both seed variation and host noise. Times below a millisecond are screening data. Repeat a fixture to reach at least 200 ms per sample before deciding a close comparison.

### Public optimizer paths

Sphere is shifted by a linear ramp from −0.75 to 0.75 on [−5,5]ᴰ. The initial guess is 1.5 on every axis where supported. Each row below made exactly 4,096 calls. Population is 32 where applicable. The public minimize path uses a scalar NumPy dot product. State uses public ask/tell with the equivalent NumPy row reduction over the whole batch. Both include setup and result retrieval. PGPE uses ranking on both paths. Its state otherwise defaults to raw scores. Final objective is the across-seed median. The time difference includes objective evaluation style.

| Optimizer | D | Scalar minimize ms [IQR] | Matrix ask/tell ms [IQR] | Final loss (minimize) |
| --- | --- | --- | --- | --- |
| CR-FM-NES | 16 | 8.17 [8.13–8.29] | 2.83 [2.75–3.00] | 9.171e-15 |
| CR-FM-NES | 128 | 17.39 [17.22–17.72] | 12.45 [12.03–13.30] | 0.1395 |
| PGPE | 16 | 7.26 [7.17–7.43] | 1.92 [1.90–1.97] | 0.4414 |
| PGPE | 128 | 14.08 [13.96–14.27] | 8.70 [8.63–8.85] | 283.9 |
| jSO | 16 | 8.21 [8.12–8.79] | 2.70 [2.62–2.77] | 5.313e-07 |
| jSO | 128 | 11.37 [11.22–11.59] | 6.10 [6.03–6.26] | 21.92 |
| DA global | 16 | 12.80 [12.74–12.95] | — | 0.03308 |
| DA global | 128 | 29.70 [29.60–29.76] | — | 13.79 |
| DA local | 16 | 12.54 [12.48–12.95] | — | 7.548e-21 |
| DA local | 128 | 26.60 [26.06–27.06] | — | 9.059e-19 |

ArchiveDE used the public state on ZDT1 (16 decisions, two objectives, P=64), with a vectorized objective, archive gating enabled and 4,096 evaluations. Timing includes its final front export. IGD uses 257 points on the analytical front and was calculated after timing. Tiered and stable Borda are different survivor policies.

| Survival policy | Total ms [IQR] | Final archive size, median | Median IGD ↓ |
| --- | --- | --- | --- |
| Tiered | 3.96 [3.92–4.04] | 108 | 0.00802 |
| Stable Borda | 3.80 [3.71–3.82] | 103 | 0.00818 |

### Native phase measurements

The C++ driver links the same Release static backend directly. The state rows include the objective in total time and time ask and tell separately. Dual Annealing uses a compiled scalar callback through its native optimize entry. Eigen populations are column-major internally. Public Python and direct C++ use different objective implementations and convert seeds differently. Use within-driver paired comparisons for attribution. Independent phase medians need not sum to the total.

| Native path | D | Total ms [IQR] | Ask ms | Tell ms | Evaluations |
| --- | --- | --- | --- | --- | --- |
| CR-FM-NES | 16 | 2.04 [2.03–2.14] | 0.875 | 1.07 | 4096 |
| CR-FM-NES | 128 | 18.08 [17.95–18.20] | 6.15 | 10.87 | 4096 |
| PGPE | 16 | 1.32 [1.29–1.36] | 0.980 | 0.264 | 4096 |
| PGPE | 128 | 8.29 [8.25–8.40] | 6.70 | 0.918 | 4096 |
| jSO | 16 | 1.33 [1.31–1.39] | 1.05 | 0.167 | 4096 |
| jSO | 128 | 4.86 [4.80–4.90] | 4.00 | 0.180 | 4096 |
| DA global | 16 | 3.40 [3.34–3.48] | — | — | 4096 |
| DA global | 128 | 21.05 [20.90–21.23] | — | — | 4096 |
| DA local | 16 | 3.44 [3.34–3.48] | — | — | 4096 |
| DA local | 128 | 17.41 [17.30–17.73] | — | — | 4096 |
| ArchiveDE tiered | 16 | 2.78 [2.74–2.85] | 1.18 | 1.52 | 4096 |
| ArchiveDE Borda | 16 | 2.48 [2.43–2.51] | 1.19 | 1.23 | 4096 |

At 128D the native CR-FM-NES tell took 10.87 ms. PGPE ask took 6.70 ms and jSO ask 4.00 ms. ArchiveDE splits time more evenly between mutation and archive/selection work. Use these phase totals to choose where to profile individual operations.

### Pareto and rank algorithms through the public API

Archive timings cover add_batch with a one-double payload. Uniform coordinates are independent on [0,1). Trade-off rows have y₂=1−y₁ in 2D, or five positive coordinates normalized to sum to one. IGD times one call including validation and index construction on 512 front and 1,024 reference rows. Rank tests use 16,384 uniform 4D points and 256 boxes, with grid query_volume equal to side⁴. Query time includes Python-visible IDs, offsets and stats. Independent checks against brute-force front/box membership and SciPy distance run outside the timer. Counts below are across-seed medians.

| Algorithm | Workload | Time ms [IQR] | Median output |
| --- | --- | --- | --- |
| 2D archive + HV | 4,096 × 2D uniform | 0.046 [0.040–0.050] | 9 survivors |
| 2D archive + HV | 4,096 × 2D trade | 0.696 [0.674–0.798] | 4,096 survivors |
| Strict ND archive | 4,096 × 2D uniform | 0.075 [0.066–0.078] | 9 survivors |
| Strict ND archive | 4,096 × 2D trade | 0.207 [0.186–0.218] | 4,096 survivors |
| Strict ND archive | 2,048 × 5D uniform | 0.596 [0.586–0.703] | 229 survivors |
| Strict ND archive | 2,048 × 5D trade | 21.18 [20.97–21.38] | 2,048 survivors |
| IGD | 512 × 1,024, 2D | 0.162 [0.151–0.180] | 0.02205 distance |
| IGD | 512 × 1,024, 3D | 0.179 [0.172–0.203] | 0.07160 distance |
| IGD | 512 × 1,024, 8D | 1.38 [1.37–1.42] | 0.42002 distance |

| Index | Box side | Build ms [IQR] | 256 queries ms [IQR] | Total IDs |
| --- | --- | --- | --- | --- |
| Borda | 0.1 | 1.27 [1.16–1.39] | 1.25 [1.23–1.27] | 416 |
| Grid | 0.1 | 1.14 [1.12–1.15] | 0.388 [0.382–0.414] | 416 |
| Borda | 0.5 | 1.17 [1.15–1.22] | 7.62 [7.53–7.65] | 263,412 |
| Grid | 0.5 | 1.08 [1.04–1.11] | 22.62 [21.68–22.67] | 263,412 |

Grid took 0.39 ms for the narrow boxes against Borda’s 1.25 ms. On wide boxes Borda took 7.62 ms against Grid’s 22.62 ms. Both returned the same memberships in this test. This crossover motivates measuring selectivity, output size, build amortization and each index’s ordering contract.

An independent five-objective scaling sweep used the same random stream per seed at successive sizes, generating a nested simplex dataset. Every row was admitted. This is a difficult input for dominance scanning and agrees with the source’s linear scans over a growing front.

| Simplex rows | Insertion ms [IQR] | Time ratio vs previous size |
| --- | --- | --- |
| 512 | 1.32 [1.30–1.39] | — |
| 1,024 | 5.33 [5.27–5.46] | 4.04× |
| 2,048 | 22.06 [21.65–22.81] | 4.14× |
| 4,096 | 91.33 [88.92–94.77] | 4.14× |

### Cassini1 application control

Public scalar minimize, six physical decision variables, 1,024 requested calls, same seven seeds. A callback recorded each returned objective. CR-FM-NES and PGPE used normalized coordinates, P=32 and midpoint initialization. DA used the midpoint, jSO its own initialization with P=32. Lower objective values are better. For this small screen all returned Cassini1 points were within bounds, and recomputing their objective reproduced the reported value. Treat the uncapped DA-local row as a budget diagnostic.

| Optimizer | Elapsed ms [IQR] | Actual calls, min–max | Median best reported |
| --- | --- | --- | --- |
| CR-FM-NES | 9.15 [8.90–9.45] | 1,024–1,024 | 18.946 |
| PGPE | 8.46 [8.41–8.93] | 1,024–1,024 | 25.135 |
| jSO | 9.25 [9.16–9.35] | 1,024–1,024 | 14.543 |
| DA global | 10.09 [9.93–10.33] | 1,024–1,024 | 21.505 |
| DA local | 22.12 [13.59–27.90] | 1,183–5,096 | 34.947 |

### Correctness probes that affect performance decisions

| Probe | Observed on the pinned source | Consequence for the campaign |
|---|---|---|
| CR-FM-NES boundary incumbent | Minimize −sum(x), 8D box [2,3], x₀=2.99, sigma=0.9, P=32, seed 101, default penalty, normalize off. At 64 calls the best observed objective was −23.419984. The result reported −21.876736, replayed at −22.209842 and had a coordinate of 3.245442. With normalization on, a coordinate still reached 3.001600. | Record evaluated physical rows independently. Preserve penalty-based adaptation and decide incumbent semantics explicitly before any fixed-call comparisons. |
| Whole-generation accounting | Requesting 33 evaluations at P=32 made 64 callbacks in both CR-FM-NES and PGPE minimize. | Use divisible budgets for baseline timing. Give partial-tail support a separately documented trial, including compatibility and quality checks. |
| Dual Annealing local budget | On unshifted Sphere, x₀=1.5, bounds [−5,5], seed 101, local search used 165 calls for a cap of 40 at D=16 and 1,285 for a cap of 260 at D=128. Global-only made exactly 40 and 260. The wrapper’s nfev agreed with the external counter, while nit was zero. | A faster run can simply have spent fewer calls. Compare against a budget-corrected reference, use wall time and callback counts, and instrument the actual number of annealing iterations. |

The screen uses one machine and short budgets. Final losses describe these workloads. They do not establish general optimizer rankings, projected speedups, multithread scaling or gains on your local Ryzen machines.

## How to execute the 200 trials

1. **Freeze the reference and the workloads.** Keep a pristine worktree and a separately built candidate. Record commit, release flags, CPU affinity, memory, dependency versions, RNG/seed mapping, objective checksum, test inputs and thread limits. First reproduce the baselines with the embedded drivers. Use the repo's existing native tests and Python suites. The measured build passed all six native CTest entries and 204 tests across `test_front_tools.py`, `test_pareto.py`, `test_jso.py`, `test_pgpe_result.py` and `test_pgpe_search.py`. These were selected suites.
2. **Respect dependencies.** Land verified correctness fixes on a separate reference before collecting performance credit. Run N01–N20 independently in the shared archive, then remeasure ArchiveDE against the same archive on both sides of each A trial. Do the same for shared rank construction in Borda and Grid. First priorities are D01, C01 and the ND archive scaling sweep. Next profile C03/C05, P01/P02 and J01 in their measured hot phases. Use the grid/Borda selectivity crossover to choose index workloads.
3. **Change one thing per iteration.** Give each row its own diff or switch. Run the indicated screen against the last accepted build with the candidate alternately first and second. Keep a fixed root comparison as well. When a proposed change depends on a rejected design, try the smallest independent version or record a concrete replacement hypothesis with the same ID and the reason. Each iteration ends with keep / opt-in / revert, benchmark evidence and the next experiment.
4. **Separate arithmetic from objectives.** For native states time ask, objective, tell, initialization and snapshot separately. Also run the public wrapper end to end. Record elapsed time, actual evaluations, throughput, peak memory and allocations. Take profiles/counters in separate instrumented runs. For DA split visiting, finite differences and L-BFGS-B line search. For archives report admission, rejection, erasure, final size and snapshots. For indices report build and query time, returned IDs and diagnostics.
5. **Hold search effort constant.** For each scalar solver run D=8/32/128, with an additional 512D stress case for the ES solvers, on shifted Sphere, rotated ellipsoid, Rosenbrock, Rastrigin and a boundary optimum. Include Cassini1 through its usual physical bounds, normalization by solver, and a deterministic simulated expensive callback. Use P=32/128 where supported, B=4,096 and 32,768, plus a final partial batch. Report target hit rate, evaluations/time to target, fixed-call best and fixed-wall best. Collect true callback counts outside the solver, independently replay x, and retain censoring for failed targets. Add controlled noise only to the separate noisy-objective suite.
6. **Exercise Pareto and rank geometry.** Use fixed evaluation/reference budgets on ZDT1/ZDT3 and DTLZ2 (3/5 objectives). Record IGD against a fixed analytic or independently generated front, and HV using a frozen per-problem reference and scaling. Verify duplicates and objective payloads. For archives scale rows geometrically, vary dimensions 2/3/5/8, rejection rates and payload widths 0/1/32. For indices vary N=256/16k/65,536/65,537/128k, D=2/4/8/16, query aspect, selectivity, correlation, ties and number of queries per build. Report build + Q × per-query time for short- and long-lived indices, plus RSS. For IGD vary front and reference sizes independently and include extreme finite inputs.
7. **Use separate selection and confirmation data.** Screen candidates with seven matched seeds/repetitions. For shortlisted changes, use 30 fresh paired seeds (e.g. 1001–1030), warm up, randomize order, and calculate a 95% paired bootstrap interval on the median reference/candidate time ratio. A provisional rollout threshold is a primary median speedup ≥1.05 with interval above one and no material regression (>3% median, confirmed on repeat) on a representative companion workload. Retain useful memory improvements separately when throughput is neutral. For sub-ms kernels time a loop of independent fixtures, charging setup only where it belongs.
8. **Gate semantics and search quality.** An exact trial must pass an identical candidate/rank/admission/ID trace on deterministic fixtures, counts and ordering included. Compare HV/IGD against independent numerical oracles with scale-aware tolerance and check special values separately. Changed reductions/RNG/search schedules require 30-seed fixed-call and fixed-time holdouts. Predeclare target and acceptable degradation per fixture before trying the change. A starting tolerance is 5% on median positive loss/IGD with a floor tied to the target accuracy, and 5 percentage points on hit rate. Report uncertainty and individual hard-case regressions. Inconclusive changes can remain experimental. Keep correctness fixes on their own merits.
9. **Check portability and composition.** On finalists run the full wheel test suite, native Debug + ASan/UBSan and both GCC and Clang release builds. Compare 1 versus 8 worker cases for changes that alter concurrency. Limit inner threading when evaluating eight independent retries. Keep seeded interleaving, GIL/callback exceptions, independent snapshots and older public defaults in regression coverage. Record changes in the changelog when public behavior changes.
10. **Record each decision.** Save raw paired rows, config, counters, corrected-reference ID and diff per trial. Include primary and holdout ratio/interval, quality, peak memory, decision and rejection reason. Rerun the integrated build against both the pristine and corrected references, remove ablations to attribute interactions and publish speedups only for measured workloads.

Minimum trial ledger:

| ID | Base / candidate commits | Case / paired seeds | Actual work | Time ratio [95% CI] | Quality / oracle | Peak bytes | Decision and reason |
|---|---|---|---:|---|---|---:|---|
| e.g. C01 | fill at execution | boundary + holdout | callback rows | report timing separately | point/value/feasibility | measured | correctness decision |

## 1. CR-FM-NES

Start with incumbent correctness, then the rank-one and diagonal updates. The native phase timings distinguish candidate generation from distribution updates. Preserve the order of mirrored samples, the penalty ranking used for adaptation and the physical points seen by the objective. Compare both normalization settings, penalties on/off, a population with nonfinite scores, and the standalone state and minimize paths. [C++ optimizer](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/crfmnes.cpp) · [Public wrapper](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/fcmaes/crfmnescpp.py)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| C01 | Track the raw best evaluated row separately from the penalty-ranked adaptation winner. Store its clipped, decoded coordinates. | Replay the boundary probe in both APIs. Match the minimum observed value to an actual callback point, including ties. Time against a corrected reference thereafter. |
| C02 | Give minimize a strict budget tail that evaluates the remaining candidates and updates the incumbent without applying a full-population distribution update. | B=1, P−1, P, P+1. Count real callbacks and return the prefix best. Document compatibility with whole-generation behavior. |
| C03 | Pass the noise matrix into sort_indices_by by const reference. | Native tell time at D=128/512, identical ranks for finite, penalized and invalid rows. |
| C04 | Make w_dist_hat accept a column expression without constructing a dynamic matrix for every column. | Count temporary allocations and compare each weight before normalizing. |
| C05 | Compute distance weights only when the path norm selects movement weights. | Count exp calls by regime. Replay both sides of the chiN transition. |
| C06 | Hoist alpha_dist(lambF) and eta_B(lambF) out of their population loops. | Compare generated weights and native tell time at P=32/256. |
| C07 | Skip distance-weight exponentials for zero rank weights on the active branch. | Count calls on the lower-ranked half and validate tied/nonfinite populations. |
| C08 | Generate Gaussian draws directly into the positive half of z, then mirror the columns. | Compare first asks bit for bit on odd/even dimensions and across independent states. |
| C09 | Store the projected row vbarᵀz once, apply the rank-one transform into a preallocated y buffer. | Ask time, allocation count, projected coordinates on ill-conditioned fixtures. |
| C10 | Compute the transform only on the sampled half of the population and obtain the other half by sign symmetry. | Compare signed zeros, both mirrored halves and fixed-budget convergence. |
| C11 | Fuse penalty accumulation with per-column physical decoding into scratch storage. | Sparse and dense bound violations, score ordering, clipping and penalty totals. |
| C12 | Keep an index permutation for sorted x and perform the weighted center sum indirectly. | Benchmark P=32/256 against gather-then-multiply. Verify original population snapshots. |
| C13 | Reuse a single column-gather buffer for y and z, eliminating duplicate full-matrix copies during permutation. | Peak bytes, native tell time, generation trace. |
| C14 | Size exY once per state and fill the final evolution-path column in place. | Count allocations over 128 generations. Include alternating independent states. |
| C15 | Replace the allocated vbar broadcast matrix with coefficient-wise broadcast expressions. | Tell time and bytes at D=512. Check each t column. |
| C16 | Build s_step1 in its destination by fusing the square and component product. | Compare a one-generation s/t reference and cache misses where available. |
| C17 | Tile the s/t updates over columns and immediately accumulate their contributions to v and D. | Sweep tile widths 4/8/16. Measure peak memory and quality under changed accumulation order. |
| C18 | Compute G_s from cached squared column norms instead of another dense squared-noise expression. | Compare step-size trajectory, especially near convergence and on rotated objectives. |
| C19 | Expose an opt-in matrix objective through the existing batch evaluator, carrying one physical row per candidate. | Time cheap NumPy and expensive Cassini1 callbacks, check exceptions, order and callback counts. |
| C20 | Transfer ownership of a contiguous decoded batch to NumPy in the state binding to remove a layout copy. | Time public ask, retain old arrays across later calls, mutate returned arrays where allowed and confirm state isolation. |

## 2. PGPE

Prioritize sampling on the 128D workload. Run both rank-shaped and raw-score variants with otherwise identical parameters. The public defaults differ between minimize and PGPE_C. Exercise the embedded ADAM update with decay steps of 1, 10 and 1,000. [C++ optimizer](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/pgpe.cpp) · [Public wrapper](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/fcmaes/pgpecpp.py)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| P01 | Generate normal values directly into scaled_noises, scaling each column in place. | Ask time, random draw order and scaled noises for scalar/vector sigma. |
| P02 | Write center ± noise straight to adjacent output columns. Remove the intermediate x1/x2 populations. | Peak bytes, native ask time and exact antithetic row ordering. |
| P03 | Decode and clip each antithetic pair into popX during generation on the state path. | Public ask time, asymmetric bounds, consistency between evaluated rows and best coordinates. |
| P04 | Reuse one population workspace across successive direct-minimize generations. | Allocations per generation, retained state snapshots, short-run startup cost. |
| P05 | Give the direct batch evaluator a contiguous output buffer for evaluated coordinates that can become popX. | End-to-end time and bytes copied under both normalization settings. |
| P06 | Represent the REINFORCE score baseline by a scalar. Broadcast only at use sites. | Peak working memory and grad_stdev against the original expression. |
| P07 | Compute grad_center as a matrix–vector product on the pairwise score differences. | Tell time at P=32/256 and fixed-budget error after rounding changes. |
| P08 | Accumulate grad_stdev one column pair at a time without building total_sigma. | Bandwidth, peak allocated bytes and sigma trace at D=128/1024. |
| P09 | Hoist reciprocal sigma and sigma squared out of the sample loop. | Per-generation divisions and tolerance near small positive sigma. |
| P10 | Read paired fitness values through strided views and fill the differences in a reusable vector. | Raw-score tell time and correct pairing of odd/even candidate rows. |
| P11 | Fuse the proposed sigma step and both allowed-change clamps into one coefficient loop. | Exact lower/upper clamp cases, zero gradient and per-coordinate sigma. |
| P12 | Compute ADAM bias correction and epsilon as scalars instead of allocating constant vectors. | Trace first step and decay-step boundaries, measure D=16/1024. |
| P13 | Cache the step-size and bias scalars while the integer ADAM time index stays the same. | Exercise repeated and changing time indices, with decay equal to and below one. |
| P14 | Fuse the two ADAM moments and center update in one coordinate pass. | Moment/center reference trace and post-update throughput. |
| P15 | Reuse rank sort indices and precompute the fixed-population rank values. | Measure tied-score and randomized-score sorts with ranking on. Preserve current tie behavior. |
| P16 | Use a read-only score view and omit the extra state-to-core copy on tell. | Public tell time, retained input buffers, ranking on/off. |
| P17 | Scan once for the lowest raw score and copy only the winning incumbent row. | Strict first-tie behavior, stop_fitness, and large populations with many successive improvements. |
| P18 | Add an opt-in vectorized objective to minimize using the existing batch trampoline. | Matrix versus row objective at equal scores, cheap/expensive cases and one caller. |
| P19 | Implement an explicit final-prefix evaluation in minimize. For an incomplete pair/generation, update only the reported incumbent. | Call caps 1/31/32/33, best-point identity and unchanged full-generation trajectory. |
| P20 | As an opt-in search trial, decay center learning rate by a measured generation schedule on high-D problems. | Compare with the current step-index schedule at equal calls and wall time on Sphere, rotated ellipsoid and noisy quadratic, using held-out seeds. |

## 3. Dual Annealing, including L-BFGS-B polishing

Time global-only and local-enabled runs separately. Count every finite-difference and line-search evaluation. For a correctness fix that reduces work, first compare outcomes at equal actual calls. Preserve callback order for stateful objectives. The source currently returns zero for the iteration count. [Annealer and finite-difference adapter](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/daoptimizer.cpp) · [Bound-constrained local solver](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/include/LBFGSB.h)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| D01 | Enforce the remaining callback budget inside the local evaluator, with controlled unwinding and propagation of the best completed probe. | Repeat budgets 40/260, B=1 and interruption during a stencil or line search. Count actual calls and recompute the best point. |
| D02 | Offer a remaining-budget trigger that skips a local solve when even one complete gradient cannot fit. | Compare with D01 at equal actual calls, especially on nonseparable objectives near the cap. |
| D03 | Cache the temperature-dependent factor in visit_fn once per strategy chain. | Count exp/log calls, compare global proposal trace and time. |
| D04 | Specialize the scalar visit path to generate the two normal draws and heavy-tailed variate without dynamic vectors. | Time the one-coordinate half of each chain. Match tail resampling and RNG order. |
| D05 | Reuse x/y Gaussian work vectors for the all-coordinate move. | Native global-only time for D=16/128/512, allocation counts. |
| D06 | Fuse absolute value, log, exponent and denominator construction for each visiting coordinate. | Compare extreme tails and zero-normal cases before timing the vector visit. |
| D07 | Replace the extra a/b vectors in periodic wrapping with an in-place component loop. | Check unit interval, near-zero nudge and exact-integer inputs. |
| D08 | Try a bounded fast path for already interior visit coordinates, preserving the near-zero rule. | Sample inputs around 0/1 and compare numerical drift versus original double-fmod arithmetic. |
| D09 | Modify one coordinate in a reusable candidate scratch vector for single-coordinate proposals. | Validate rejected moves restore state, accepted moves become current, and random traces agree. |
| D10 | Fuse normalized clipping and physical decoding into the callback argument workspace. | Native and public wall time, physical coordinates on asymmetric boxes. |
| D11 | Replace two full copies per finite-difference coordinate with one reusable stencil buffer, restoring each coordinate after its probes. | Local-enabled time on 128D, identical objective call sequence, optimum on a face. |
| D12 | Reuse unchanged center values in clipped one-sided stencils only when the caller opts into deterministic-objective semantics. | Count eliminated calls near box faces, verify gradients. Replay the default with a call-dependent objective. |
| D13 | Batch independent finite-difference probes behind an explicit matrix callback and a hard remaining-budget check. | Throughput for NumPy and simulated expensive objectives, per-row accounting and failure handling. |
| D14 | Persist normalized lower/upper arrays and initial-vector workspace across local solves. | Allocation count and time on multimodal runs with many short polishes. |
| D15 | Reuse L-BFGS-B solver storage between polish calls, resetting all quasi-Newton history on every new solve. | Same local trajectory from multiple starts, including a failed preceding solve. |
| D16 | Evaluate a cheaper one-sided gradient as an opt-in local scheme. | Objective calls per successful polish, on-bound accuracy, fixed-budget quality on Rosenbrock and Cassini1. |
| D17 | Expose L-BFGS-B history size as a polish option and try m=3/6/12. | Peak bytes and calls to target on low/high-dimensional quadratics and Rosenbrock. |
| D18 | Make the maximum local iterations depend on remaining calls and dimension. | Matched-budget quality and frequency of incomplete final gradients with D01 active. |
| D19 | Trigger a polish only after a minimum relative global improvement or a measured stagnation interval, as an opt-in policy. | Paired fixed-call and fixed-time success on multimodal objectives. Include global-only and current hybrid controls. |
| D20 | Cache the finite sequence of default chain temperatures for repeated short restarts. | Measure 1,000 short runs including setup, match restart indices and all temperature values. |

## 4. jSO

This is the separate jSO implementation with success-history adaptation and population reduction. Native ask dominates the 128D screen. Protect target stops on partial batches, tie handling, fixed coordinates, and population/archive order. Use the repository's Python implementation as an independent behavioral reference where the two RNG implementations permit comparison. [Native state](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/jsooptimizer.cpp) · [Binding](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/bindings/jso_bindings.cpp) · [Python solver](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/fcmaes/jso.py)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| J01 | Fill pending trial rows directly instead of copying the whole parent before crossing over. | Bytes written and native ask time. Preserve forced-coordinate short-circuit draws. |
| J02 | Reserve and reuse trial-row storage for failed offspring instead of freeing it every generation. | Allocation count at high rejection, verify repeated ask and archive ownership. |
| J03 | Test a flat population slab with row indices for parents. | Cache misses and elapsed time at 16/128/512D. Match population snapshots after shrinking. |
| J04 | Store displaced parents in contiguous archive slots with constant-time replacement. | Archive-enabled versus rate-zero runs. Preserve donor numbering and draw count. |
| J05 | Cache dimensionwise bound widths for initialization. | Thousands of short restarts, including fixed dimensions and near-limit valid widths. |
| J06 | Hoist pbest attraction coefficient and references to parent/pbest/donor rows outside the coordinate loop. | Time the mutation loop at high D and verify proposed points bitwise. |
| J07 | Fast-path finite, in-bound mutant coordinates before midpoint repair. | Interior and boundary-heavy workloads, NaN handling and identical bounds. |
| J08 | Separate the forced coordinate into prefix/forced/suffix loops while preserving uniform draws. | Count RNG draws and branch misses at D=16/512. |
| J09 | Replace full stable ranking for pbest with top-k selection and deterministic ordering of the selected prefix. | Compare prefix IDs for ties and all p schedules. Time at P=32/256/1024. |
| J10 | Reuse the order scratch and exploit the already-ranked survivors immediately after population reduction. | Duplicate rank-sort count per generation and identical reproduction on equal fitness. |
| J11 | Reuse survivor scratch across shrink events, moving row ownership in sorted survivor order. | Shrink-heavy budgets, allocations and post-shrink parent IDs. |
| J12 | Preallocate the successes buffer and reset its length each generation. | All-success and low-success tells, memory stability across retries. |
| J13 | Fold the maximum improvement scale into the success-collection pass. | Match finite/infinite improvement weighting and Lehmer updates on pathological scores. |
| J14 | Fuse invalid-score sanitization and first-minimum discovery in accept. | Total tell time, all-invalid input, target hit within a valid prefix. |
| J15 | Retain current population and archive capacities when compacting after shrink. | Short versus long runs, peak RSS and random archive eviction parity. |
| J16 | Add a native contiguous ask-output path for the binding, avoiding row-by-row intermediate object traversal. | Public ask time, independent arrays and empty final batch shape. |
| J17 | Pass read-only contiguous score spans through the binding and allocate owned values only on initialization when required. | Public tell time. Python arrays may be modified by their owner after return. |
| J18 | Add opt-in batch objective evaluation to minimize. Define full-batch target accounting separately from serial prefix stopping. | Compare same physical rows and count actual work on target reached in the middle of a batch. |
| J19 | Specialize tiny active populations near the end of the run with a small fixed sorting network and original-index tie keys. | Tail-only and total runtime, including population sizes 4–8 and repeated scores. |
| J20 | Try a slower linear population-reduction schedule as a named search option. | Fixed calls and fixed time on rotated/multimodal functions plus jSO→CMA as an unchanged downstream control. Record jSO's share separately. |

## 5. ArchiveDE

Benchmark its mandatory exact archive and its breeding selector separately. An offspring can enter the archive then disappear later in the same batch. The selection gate depends on admission at insertion time. Test all four gate/tiered combinations and final partial generations. Exact trials must reproduce the Rust-compatible seed stream, parent order and duplicate handling. [State and selection](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/archivede.cpp) · [Public wrapper](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/fcmaes/archivede.py)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| A01 | Decode each unit-space trial directly into the final physical buffer. | Native ask time and physical rows for wide/fixed bounds, including overflow-safe interpolation. |
| A02 | Precompute dimension widths and flags selecting the normal or extreme-width decode path. | High-D ask time and bitwise physical output for both paths. |
| A03 | Reuse capacity for pending unit rows across asks. | Count allocations and simulate interrupted allocation. A retried ask must consume the same RNG stream. |
| A04 | Use one row-major candidate buffer instead of a vector allocation per pending row. | Bytes/ask and seeded population/front traces, including budget shorter than P. |
| A05 | Delay DE difference arithmetic until crossover selects that coordinate. | D=128/512, CR=0.3, identical draws including the forced coordinate. |
| A06 | Keep a per-batch candidate-index pool referencing parents and offspring. | Remove deep copies of rejected vectors. Exercise gate on/off and no admitted offspring. |
| A07 | Reserve next-parent slots and copy only selected rows after selection commits. | Crash/retry at allocation boundaries, memory high-water mark, duplicate payload identity. |
| A08 | Add a Borda-only selection path that never materializes the unused coordinate-rank matrix. | Time non-tiered tell. Verify exactly the same stable Borda order on tied rows. |
| A09 | Replace tiered floating rank rows by contiguous 32-bit ranks with checked wide arithmetic for distances. | Time P=64/512 and objective counts 2/8. Compare nearest-spread ties and overflow fallback. |
| A10 | Record objective champions during per-objective ranking, respecting the “unused champion” rule. | Verify repeated champion and all-tie cases, count eliminated scans. |
| A11 | Reuse stable-sort index and score buffers for every generation. | Allocation counts and time at small P, including ranks after invalid tell/retry. |
| A12 | Try stable integer-key sorting of Borda sums above a measured population cutoff. | Scalar/tiered, ties, rank order and wall time above/below crossover. |
| A13 | Compact spread candidates with a live mask instead of erasing the chosen element and shifting the tail. | Order-sensitive ties, distance count and tell time on wide tiered pools. |
| A14 | Specialize two-objective rank distance with exact checked arithmetic. | Match chosen IDs against generic selector and exercise large-count fallback. |
| A15 | Block the farthest-first nearest-distance update across candidate rows. | Vectorization report and native tiered tell time. Retain the first maximum tie. |
| A16 | Cache the deterministic parent-only selection result keyed by exact ordered parents and policy. | Hit rate with low admissions. Check the potentially nonidentity permutation on repeated parent-only rounds. |
| A17 | Add an internal validated-batch archive insertion path after the whole tell matrix has passed finiteness checks. | Time gated dominated streams, same failure/retry semantics and exactly-once admissions. |
| A18 | Remove one copy in the owned ask export while retaining the canonical pending data for repeated ask. | Public ask time and independent arrays when a caller mutates or retains the first output. |
| A19 | Reserve final population snapshot storage from rows, dimensions and objectives. | Snapshot-heavy users, checked size multiplication, result independent of later calls. |
| A20 | Try a configurable spread-candidate fraction in the tiered selector. | Keep defaults as controls. Compare HV, IGD, unique front size and time on ZDT1/3 and DTLZ2, 30 paired holdouts. |

## 6. ParetoArchive2D and incremental hypervolume

The same 4,096 insertions exercise two different paths. Most independent uniform pairs are rejected. All trade-off pairs survive. Measure admission, removal, copying and snapshot costs separately. The C++ front exposes a live map view, supports copying/moving, caches an insertion gap and a dominance witness, and compensates area increments. Keep prefix masks, first duplicate payload, reference-edge behavior, exception safety and the live-view contract. [Front and area updates](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/include/paretofront2d.h) · [Public archive adapter](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/include/paretoarchive.hpp)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| H01 | Add an explicit rightmost append fast path for ascending-x, decreasing-y streams. | Sorted, reverse and shuffled fronts. Compare every admission and each HV prefix. |
| H02 | Add the analogous leftmost insertion hint for decreasing-x arrival. | Reverse-front throughput and one-point or equal-x replacement. |
| H03 | Broaden the existing gap hint to a bounded walk over adjacent successors. | Sweep neighbor distance 1–8 on clustered streams. Invalidate hints consistently after erase, copy and move. |
| H04 | Keep a second dominance witness selected from a different region of the front. | Witness hit rate and cycles/row on clustered dominated loads. Validate every rejected row. |
| H05 | Use the last rejecting predecessor as a witness for subsequent nearby insertions. | Ordered bursts versus shuffled rows, and comparison with H04 independently. |
| H06 | Precompute finite/reference/witness rejection masks for small blocks under a frozen archive prefix. Commit in original order. | Check NaN/error row and prefix masks. Pre-rejections need valid dominance certificates throughout the block. |
| H07 | Hoist payload-shape validation out of the per-row internal batch insertion. | Batch sizes 1–4096. Preserve shape-before-mutation and row-specific errors. |
| H08 | Specialize the common zero-payload insertion path in the adapter. | Compare W=0,1,32. Inspect allocation and branch counts without changing public map types. |
| H09 | Pre-size the admission-mask buffer and write by row index. | Fast-reject workload, exception at rows 0/middle/last. |
| H10 | Fill the Python output mask directly to remove its intermediate native representation. | End-to-end add_batch time, bool dtype, ownership, and failing-prefix behavior. |
| H11 | Give the no-dominated-successor case a checked rectangle-only area-update path after normal map lookup. | Random antichains outside the cached gap. Compare compensated totals and overflow behavior. |
| H12 | Specialize same-x replacement when it removes no successor. | Same-x improving and worsening streams. First duplicate payload retained. |
| H13 | Try erasing successors individually using the returned iterator after the area delta has been validated. | Long deletion runs versus single deletion. Valid hints, live iterators and allocation-failure rollback. |
| H14 | Stage payload materialization only once for same-key replacement, with a small fixed-width copy specialization. | Payload widths 1/4/64, self-aliasing native payloads and allocation failure. |
| H15 | Reuse an already allocated payload buffer on eligible same-key updates after all throwing validations. | Count allocations for W=32, verify strong guarantee and aliased-source semantics. |
| H16 | Use a cached witness block to reject repeated identical rows in add_batch. | Duplicate bursts, first payload ownership and exact insertion masks. |
| H17 | Add ordered-input detection that feeds successive map insertion hints for an ascending subrun. | Runs of 4/32/4096 intermixed with reversals. Online masks and area after every prefix. |
| H18 | Size snapshot output once and write coordinates and payloads by offset. | Time front() at 10/4,096/65,536 survivors, sorted order and independent snapshots. |
| H19 | Fill owned NumPy snapshot arrays directly in a single map traversal. | Public front time, peak bytes, GC and later mutation of the archive. |
| H20 | For the public wrapper only, prototype a small sorted-vector front with promotion to the map above a measured size. | Sweep front sizes 1–256. Match all prefix masks/HVs, failing insertions and promotion. Preserve the existing native live-map API. |

## 7. ParetoArchiveND, the strict archive shared with ArchiveDE

Profile five-objective trade-off rows first. All 2,048 survived and the measured insertion cost was about 35 times the uniform stream. Two dimensions already have an exact-sum-line shortcut and an ordered skyline. Preserve strict dominance, coexistence of duplicates, objective-sum then arrival ordering, and the public batch's valid-prefix behavior. Supplement uniform and simplex input with rounded equal sums, cancellation, asymmetric coordinate scales and late transitions off an exact line. [Archive implementation](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/archivede.cpp) · [State and storage](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/include/archivede.hpp)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| N01 | Specialize strict dominance for three objectives. | Scalar comparison count on random/DTLZ fronts, including equal records. |
| N02 | Try an unrolled five-objective dominance test with early rejection. | Both five-objective screen fixtures, branch misses and membership oracle. |
| N03 | Compare vectorized all-less-or-equal/any-strict tests to scalar early exit for 8–32 objectives. | Vary correlation and rejection rate. Select a dimension crossover on held-out data. |
| N04 | Probe likely discriminating objective axes first, keeping stored objective order intact. | Count comparisons per call on anisotropic streams and the cost of adapting the probe order. |
| N05 | Introduce a small cache of recent valid dominance witnesses. | Measure hit rate, memory and cost on front-wide trade-offs at cache widths 2/4/8. |
| N06 | Add a strict min/max bounding box per block of entries to skip impossible dominators. | Nodes/rows visited, maintained bounds after removals, agreement with exhaustive filtering. |
| N07 | Use a separate bounding-box rejection for the suffix scan that removes entries dominated by a newcomer. | Dense deletion and zero-deletion streams, survivors including ties, update overhead. |
| N08 | Store objectives in padded fixed-size inline arrays for common widths 2–5. | Heap allocations per admitted row and RSS, still supporting arbitrary nobj. |
| N09 | Put decision payloads in reusable slots referenced by entries. | W=0/16/128 and deletion churn. Distinguish duplicate payloads and handle allocation failure. |
| N10 | Keep a compact vector of sorted entry handles, moving handles instead of full entries on insertion/erase. | Prefix/suffix shift costs, sum/serial snapshot order and memory overhead. |
| N11 | Fuse survivor marking with compaction so insert-before-remove does not shift a long suffix twice. | Random deletions and adversarial equal-rounded-sum cases. Commit only after allocations succeed. |
| N12 | Test blocked sorted storage with local splits under many nondominated insertions. | Target 2k–64k survivors, compare with contiguous storage on tiny fronts. |
| N13 | Add a 3D skyline dominance index above a measured threshold while retaining duplicate multiplicities and arrival ordering. | Exact masks, total insert and snapshot time for spherical and uniform fronts. |
| N14 | Prototype an ND-tree with per-node bounds as a side index for large fronts. | Ignored-node fraction, insert/remove time and bytes versus the scan on 3/5/8 objectives. |
| N15 | For nobj=2, accelerate exact-line migration using a sorted temporary set of references and hinted map insertion. | Migration from 1/1k/64k rows, peak memory and strong guarantee on failure. |
| N16 | Coalesce identical 2D skyline keys into groups with per-arrival payloads. | Duplicate-heavy traces. Exact front order and per-insertion admission remain observable. |
| N17 | Cache sorted snapshot handles with invalidation on any accepted or erasing insertion. | Repeated front() reads versus write-heavy workloads, snapshot independence. |
| N18 | Add exact, overflow-safe certificates for an equal-weight hyperplane in 3–5D before admitting a row without a scan. | Validate near-coplanar rounded sums against an exact arithmetic oracle. Measure certificate cost on simplex and uniform data. |
| N19 | Vectorize input validation across blocks while committing the valid prefix in order. | Error row index and retained contents on NaN/Inf, plus throughput on cheap rejected rows. |
| N20 | Try size-based dispatch between a compact scan and the best validated index, with hysteresis around the crossover. | Holdout varying front growth/shrink and ArchiveDE runtime. Charge conversion cost, RSS and snapshots. |

## 8. Exact inverted generational distance

The existing source scans small/general-dimensional fronts, sorts by x for larger 2D fronts and builds a kd-tree for 3D. Extreme magnitudes have a scaled hypot fallback. Measure total time including index construction, and measure repeated-query time separately whenever an index is cached. Compare against exhaustive Euclidean nearest neighbors in the same direction, and a high-precision oracle for extreme magnitudes. [Distance kernels](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/include/frontdistance.hpp)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| I01 | Add a sorted scalar search for d=1, seeded by lower_bound. | Small/large fronts, duplicates, exact matches and empty-front behavior. |
| I02 | Choose the existing 2D scan/sorted crossover using both front and reference counts. | Sweep 8–4096 front rows and 1–4096 queries. Include sorting cost. |
| I03 | Detect already x-sorted input before copying/sorting the 2D front. | Sorted nondominated, sorted nonmonotone and shuffled inputs with ties. |
| I04 | Initialize the 2D best distance from the pivot neighbor before the leftward scan. | Candidate visits on random clouds, far-away queries and both end pivots. |
| I05 | Process references in x order using a moving pivot, restoring per-row results to original reduction order. | Sort overhead versus savings, exact mean with repeated x values. |
| I06 | Seed each coherent 2D query with the previous query's nearest front point as a valid upper bound. | Ordered trajectory fronts and shuffled controls, exact nearest distances. |
| I07 | Sweep 3D kd-tree leaf capacities 8/16/32/64. | Report build and query time, node visits and independent holdouts. |
| I08 | Split 3D nodes on greatest coordinate spread in place of a rotating axis. | Skewed/coplanar versus isotropic clouds, build cost and tree depth. |
| I09 | Store subtree boxes for tighter pruning in the native 3D tree. | Node visits versus preprocessing bytes and end-to-end time. |
| I10 | Try an iterative 3D nearest traversal with a bounded local stack and safe spill. | Deep and degenerate trees, instructions/query and identical minima. |
| I11 | Group 3D leaf coordinates by axis for vectorized squared-distance evaluation. | Leaf-only and total time, near-duplicate distances and tail sizes. |
| I12 | Use dimension-specialized brute-force kernels at 4 and 8 objectives. | D=4/8, fronts of 16/512/8192, tolerance against exhaustive double precision. |
| I13 | Tile reference/front matrices to keep front coordinate blocks hot in cache. | Throughput versus block sizes and auxiliary O(nref) best distances. |
| I14 | Cut off a scalar high-dimensional distance once the partial squared sum reaches the current best. | Dimensions 8–64 with clustered and uncorrelated data. Include branch cost. |
| I15 | Try a 4–8D kd-tree with data-driven fallback when query pruning is ineffective. | Total build-plus-query time and worst-case slowdown on broad clouds. |
| I16 | Add blocked SIMD finiteness/magnitude checks before dispatch. | Ordinary and subnormal/huge coordinate sets. Retain invalid-input error precedence. |
| I17 | Prototype a power-of-two rescaling path for uniform extreme magnitudes. Retain robust fallback for mixed scales. | High-precision oracle for opposite-sign maxima, underflow and representable means. |
| I18 | Expose an immutable prepared-front option for repeated IGD calls against different references. | Break-even number of calls. Enforce ownership when the input array is modified. |
| I19 | Parallelize independent reference queries above a size threshold, then reduce in original row order. | 1/2/4/8 workers, overhead on small jobs and reproducible mean. |
| I20 | Seed search using the nearest coarse cell representative on spatially clustered high-D fronts. Finish with exact pruning/scan. | Include construction cost, candidate visits and agreement on outliers and ties. |

## 9. BordaBoxIndex

The implementation selects the narrowest axis, keeps cross-ranks for every axis pair, counts rank-sum candidates and tests active ranks in 64-row blocks. The public stats are observable. A single query returns IDs. The batch also returns offsets and diagnostic counters. Preserve output order along the chosen axis, inclusive ties, immutable ownership, independent concurrent reads, and the 65,536-row rank-width boundary. [Ranks and Borda index](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/include/rankbox.hpp) · [Batch binding](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/bindings/front_bindings.cpp)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| B01 | Split the IDs-only single-query kernel from the stats-producing batch kernel to omit the unused score-count pass. | Time single queries and compare IDs against batch. Batch counters remain exact. |
| B02 | Fuse score-candidate counting into the same block loop as cross-axis filtering. | Batch time, cache traffic and score_candidates agreement. |
| B03 | Use the rank-sum filter to seed the hit mask, allowing exact rejection before reading cross-ranks. | Measure pruning on correlated and uniform points. Prove sum-band necessity with inclusive endpoints. |
| B04 | Skip remaining rank columns for blocks whose hit mask is already empty. | Narrow high-D boxes, wide-box branch cost and exact counters. |
| B05 | Check active axes in order of increasing rank-span width. | Candidate tests per query on anisotropic input. Original reporting axis and tie order preserved. |
| B06 | Replace byte hit flags by a 64-bit bitset and enumerate set bits in ascending position. | Sparse and dense output. Last partial block and exact ID order. |
| B07 | Tune scan block widths 16/32/128 according to rank width and dimension. | Query p50/p95 above and below the 16-bit rank cutoff. |
| B08 | Specialize one- and two-active-axis kernels. | Slabs and effectively unbounded axes. Preserve query counters and closed boundaries. |
| B09 | Reserve the output vector adaptively from observed intersection widths. | Sparse queries and high-output p95, peak bytes per thread. |
| B10 | Append IDs to the batch destination through a sink instead of building one vector per query. | Count allocations for 1/256/4096 queries. Check CSR offsets and lifetime. |
| B11 | Reuse spans and active-axis scratch per batch caller. | Many tiny queries and simultaneous read-only callers. Verify zero cross-thread state leaks. |
| B12 | Try branchless lower/upper bound searching over large sorted coordinate arrays. | Sample narrow boxes and heavy coordinate ties. Compare closed range endpoints. |
| B13 | Store cross-ranks in contiguous per-axis slabs to remove per-column allocation and pointer chasing. | Build time, RSS and cache misses across 4/8/16 dimensions. |
| B14 | Switch to on-demand cross-rank lookup at high dimensions to reduce quadratic storage. | Total time for 1/100/10k queries after build and peak memory. |
| B15 | Derive 32-bit score storage when the maximum rank sum provably fits. Keep wide query arithmetic. | Score-band counters at threshold/overflow boundaries and bandwidth. |
| B16 | Build axis rank orders independently in parallel with per-worker radix scratch. | Construction time plus memory at 1/2/4/8 workers, exact row-ID tie ordering. |
| B17 | Change radix digit width for the shared stable rank constructor. | Timings over tiny/tied/random columns and signed zero. Rerun GridBoxIndex as a shared-code control. |
| B18 | Build cross-rank slabs in cache-sized row tiles. | Cold build time and peak workspace for N=16k/128k, d=4/16. |
| B19 | Reuse previously found rank spans for repeated coordinate endpoints within a batch. | Repeated slabs versus random endpoints. Measure lookup overhead and handle infinities/NaN per contract. |
| B20 | Process independent queries in parallel with private buffers and ordered concatenation. | 1/2/4/8 threads, exact per-query IDs/counters and memory on wide boxes. |

## 10. GridBoxIndex

This is the sparse rank grid over a fixed point cloud. Its public query order follows the grid key then input row ID. Changing the nominal resolution can change that order and the theoretical-cell counters. Treat the existing resolution rule as part of the contract for a given query_volume. The current code already handles full coverage, has a dense first-occupied table when cheap, and switches to scanning occupied keys for many empty prefixes. [Grid construction and query](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/_fcmaescpp/include/rankbox.hpp) · [Public API](https://github.com/dietmarwo/fast-cma-es/blob/2cea79f9223f2153655bb8a56934cb9a5452c5d2/fcmaes/rankindex.py)

| Trial | Change to try | Deciding measurement or check |
|---|---|---|
| G01 | Reuse per-axis query scratch across a batch. | 256 and 4,096 small boxes, concurrent readers, zero/one result cases. |
| G02 | Emit IDs directly into the binding's batch destination. | Wide-box throughput, bytes copied and exact CSR offsets/stats. |
| G03 | Reserve output capacity using a bounded estimate from rank spans. | Compare narrow/wide and clustered data. Track memory per thread. |
| G04 | Specialize the partial-cell inner test for d=2 and d=4. | Run the 4D wide screen plus generic-dimensional controls, preserving inclusive endpoints. |
| G05 | Test the narrowest constrained axis first in partial cells. | Axes tested per point and time on flat slabs, keep public key/ID order. |
| G06 | Store cell ranks in 16 bits when N ≤ 65,536. | Construction/query time and bytes per point across 65,535/65,536/65,537. Widen before subtraction. |
| G07 | Lay out ranks by axis within cell blocks for vectorized boundary tests. | Boundary-dominated wide queries, sparse-cell overhead, exact diagnostic pointTests. |
| G08 | Coalesce consecutive fully covered cells into one contiguous range copy. | Full and partially covered adjacent cells. Same full/nonempty counts and ID sequence. |
| G09 | Replace recursive prefix enumeration with an iterative mixed-radix counter. | Prefix count, stack cost and query time for d=2/4/8, including holes. |
| G10 | Skip empty prefix ranges using binary search on occupied keys. | Sparse grids with many empty prefixes. Compute theoretical touchedCells exactly as specified. |
| G11 | Select occupied-key scan versus prefix enumeration from calibrated cost estimates. | Compare both routes on the same grid, broad versus thin slabs, identical counters and keys. |
| G12 | Cache per-cell decoded coordinates for the occupied-key scan. | Division count versus memory and build time in high dimensions. |
| G13 | Decode key digits using shifts and masks for power-of-two resolutions. | Bit-identical key order. Pair power-of-two and odd-resolution workloads. |
| G14 | Replace sparse comparison sort of point IDs with stable radix sorting by cell key. | Sparse-build time, memory and exact row order within each cell. |
| G15 | Try a two-level directory for moderately sparse key spaces. | Compare with firstOccupied and lower_bound. Include directory construction and RSS. |
| G16 | Calibrate the dense first-occupied table cutoff by measured occupancy and memory budget. | Threshold sweep around both current limits on multiple N. Measure query/build break-even. |
| G17 | Reuse first/last rank-cell boundaries in a per-resolution table for repeated queries. | Short-query latency and table overhead for extremely sparse, high-resolution grids. |
| G18 | Populate cellRanks in cache-sized output blocks during construction. | Cold build throughput at N=16k/128k, d=4/8. Verify ranks after stable reorder. |
| G19 | Parallelize independent cell-rank gathering during construction after stable cell order is established. | 1/2/4/8 threads, memory traffic and deterministic contents. |
| G20 | Parallelize batch queries with per-worker ID buffers and stable prefix-sum assembly. | Wide and narrow batches, p95 latency, peak RSS, IDs, offsets and all four stats. |

## Reproduce the measured screen

The four drivers used for the tables and diagnostic probes are embedded below. Save the named code blocks next to a checkout named `repo`. Set EIGEN_INC to the directory containing Eigen/Core. Supply Eigen 3.4 via its normal CMake package or pkg-config. For this run the pkg-config entry pointed to the installed Eigen headers. Run Python from outside the checkout so the installed extension is imported. On another machine choose an allowed logical CPU in place of 3.

```bash
git clone https://github.com/dietmarwo/fast-cma-es.git repo
git -C repo checkout --detach 2cea79f9223f2153655bb8a56934cb9a5452c5d2
python -m venv .venv
source .venv/bin/activate
python -m pip install cmake ninja nanobind==3.1.0 scikit-build-core==1.1.2 numpy==2.3.5 scipy==1.17.0 scikit-learn threadpoolctl loguru pytest
CXX=g++ CMAKE_BUILD_PARALLEL_LEVEL=3 python -m pip install --no-build-isolation --no-deps --config-settings=build-dir=build/perf-gcc ./repo
g++ -std=c++17 -O3 -DNDEBUG -Irepo/_fcmaescpp/include -I"${EIGEN_INC}" native.cpp repo/build/perf-gcc/libfcmaes_backend.a -pthread -o native
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 3 python screen.py > public.jsonl
taskset -c 3 ./native > native.csv
taskset -c 3 python controls.py > controls.jsonl
python probes.py
cmake -S repo/tests/native -B native-tests -DCMAKE_BUILD_TYPE=Release
cmake --build native-tests --parallel 3
ctest --test-dir native-tests --output-on-failure
python -m pytest -q repo/tests/test_front_tools.py repo/tests/test_pareto.py repo/tests/test_jso.py repo/tests/test_pgpe_result.py repo/tests/test_pgpe_search.py
```

<details>
<summary>screen.py</summary>

```python
import json
from time import perf_counter_ns as clock

import numpy as np
from scipy.optimize import Bounds
from scipy.spatial.distance import cdist
from fcmaes import archivede, crfmnescpp, dacpp, jsocpp, pareto, pgpecpp, rankindex

def timed(f):
    start = clock()
    value = f()
    return (clock() - start) / 1e6, value

def sphere(X):
    return np.einsum("...i,...i->...", X, X)

def zdt(X):
    g = 1 + 9 * X[:, 1:].mean(axis=1)
    return np.column_stack((X[:, 0], g * (1 - np.sqrt(X[:, 0] / g))))

def measure(name, case, seed):
    rng = np.random.default_rng(seed)
    budget, pop = 4096, 32
    if name in ("crfmnes", "pgpe", "jso", "da"):
        d, mode = case
        bounds = Bounds(np.full(d, -5.), np.full(d, 5.))
        kw = dict(bounds=bounds, rg=rng)
        shift = np.linspace(-.75, .75, d)
        x0 = np.full(d, 1.5)
        calls = 0
        def row(x):
            nonlocal calls
            calls += 1
            u = x - shift
            return u @ u
        module = dict(crfmnes=crfmnescpp, pgpe=pgpecpp, jso=jsocpp, da=dacpp)[name]
        if name == "da" or mode == "min":
            args = dict(max_evaluations=budget, **kw)
            if name != "jso":
                args["x0"] = x0
            if name == "da":
                args["use_local_search"] = mode == "local"
            else:
                args["popsize"] = pop
            ms, r = timed(lambda: module.minimize(row, **args))
            assert r.success and r.nfev == calls and np.isclose(sphere(r.x - shift), r.fun)
            return dict(ms=ms, n=calls, value=r.fun)
        def loop():
            nonlocal calls
            if name == "jso":
                state = module.Jso(**kw, popsize=pop, max_evaluations=budget)
            else:
                cls = crfmnescpp.CRFMNES_C if name == "crfmnes" else pgpecpp.PGPE_C
                if name == "pgpe":
                    kw["use_ranking"] = True
                state = cls(d, **kw, popsize=pop, x0=x0)
            while calls < budget:
                X = state.ask()
                assert len(X)
                y = sphere(X - shift)
                flag = state.tell(y)
                assert flag == 0
                calls += len(X)
            return state.result()
        ms, r = timed(loop)
        assert r.nfev == calls == budget and np.isclose(sphere(r.x - shift), r.fun)
        return dict(ms=ms, n=calls, value=r.fun)
    if name == "archivede":
        d, tier = case
        def loop():
            state = archivede.ArchiveDE(2, Bounds(np.zeros(d), np.ones(d)), popsize=64,
                        max_evaluations=budget, seed=seed, tiered=tier)
            while len(X := state.ask()):
                state.tell(zdt(X))
            assert state.nfev == budget
            return state.front()
        ms, (X, Y) = timed(loop)
        assert np.allclose(Y, zdt(X))
        u = np.linspace(0, 1, 257)
        quality = cdist(np.column_stack((u, 1-np.sqrt(u))), Y).min(axis=1).mean()
        return dict(ms=ms, n=budget, value=quality, front=len(Y))
    if name in ("archive2", "archivend"):
        m, shape = case
        n = 4096 if m == 2 else 2048
        Y = rng.random((n, m))
        if shape == "trade":
            if m == 2:
                Y[:, 1] = 1-Y[:, 0]
            else:
                Y /= Y.sum(axis=1, keepdims=True)
        payload = np.arange(n, dtype=float)[:, None]
        a = pareto.ParetoArchive2D([1,1], 1) if name == "archive2" else pareto.ParetoArchiveND(m, 1)
        ms, mask = timed(lambda: a.add_batch(Y, payload))
        X, front = a.front()
        assert np.array_equal(Y[X[:,0].astype(int)], front) and mask.sum() >= len(front)
        # Independent streaming front for two objectives, and exhaustive membership in higher dimensions.
        if m == 2:
            order = np.argsort(Y[:,0],kind="stable")
            ys = Y[order,1]
            selected = order[ys < np.minimum.accumulate(np.r_[np.inf,ys[:-1]])]
            if name == "archive2":
                f = Y[selected]
                hv = np.sum(np.diff(np.r_[f[:,0],1.])*(1-f[:,1]))
                assert np.isclose(a.hypervolume(), hv)
        else:
            selected = [i for i,y in enumerate(Y) if not np.any(np.all(Y<=y,axis=1) & np.any(Y<y,axis=1))]
        assert np.array_equal(np.sort(X[:,0].astype(int)),np.sort(selected))
        return dict(ms=ms, n=n, value=len(front))
    if name == "igd":
        m, = case
        F, R = rng.random((512,m)), rng.random((1024,m))
        ms, dist = timed(lambda: pareto.igd(F,R))
        assert np.isclose(dist,cdist(R,F).min(axis=1).mean(),rtol=1e-12,atol=1e-14)
        return dict(ms=ms,n=len(R),value=dist)
    d, side = case
    X = rng.random((16384,d))
    lo = (1-side)*rng.random((256,d))
    hi = lo + side
    ctor = (lambda: rankindex.BordaBoxIndex(X)) if name == "borda" else (lambda: rankindex.GridBoxIndex(X,side**d))
    build, index = timed(ctor)
    ms, out = timed(lambda: index.query_batch(lo,hi))
    assert out.offsets[-1] == len(out.ids)
    for i in range(len(lo)):
        ids = out.ids[out.offsets[i]:out.offsets[i+1]]
        brute = np.flatnonzero(np.all((X>=lo[i]) & (X<=hi[i]),axis=1))
        assert np.array_equal(np.sort(ids),brute)
    return dict(ms=ms, n=len(lo), value=len(out.ids), build=build)

if __name__ == "__main__":
    cases = {}
    for name in ("crfmnes","pgpe","jso"):
        cases[name] = [(d,mode) for d in (16,128) for mode in ("min","state")]
    cases["da"] = [(d,mode) for d in (16,128) for mode in ("global","local")]
    cases["archivede"] = [(16,t) for t in (True,False)]
    cases["archive2"] = [(2,s) for s in ("uniform","trade")]
    cases["archivend"] = [(m,s) for m in (2,5) for s in ("uniform","trade")]
    cases["igd"] = [(m,) for m in (2,3,8)]
    for name in ("borda","grid"):
        cases[name] = [(4,s) for s in (.1,.5)]
    jobs = [(k,c) for k,cs in cases.items() for c in cs]
    order_rng = np.random.default_rng(77)
    for seed in (100,101,102,103,104,105,106,107):
        for i in order_rng.permutation(len(jobs)):
            k,c = jobs[i]
            r = measure(k,c,seed)
            if seed != 100:
                print(json.dumps(dict(name=k,case=c,seed=seed,**r)),flush=True)
```

</details>

<details>
<summary>native.cpp</summary>

```cpp
#include "archivede.hpp"
#include "crfmnesoptimizer.hpp"
#include "daoptimizer.hpp"
#include "jsooptimizer.hpp"
#include "pgpeoptimizer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Vec = std::vector<double>;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
static int calls;
static double sphere(int d, const double* x) {
    ++calls;
    double value = 0;
    for (int j=0; j<d; ++j) {
        const double u = x[j] - (-.75 + 1.5*j/(d-1));
        value += u*u;
    }
    return value;
}
int main() {
    constexpr int budget=4096, pop=32;
    std::mt19937 shuffle(77);
    std::vector<std::pair<std::string,int>> jobs;
    for (int d : {16,128})
        for (auto key : {"crfmnes","pgpe","jso","da-global","da-local"})
            jobs.emplace_back(key,d);
    for (auto key : {"tiered","borda"})
        jobs.emplace_back(key,16);
    std::cout<<"name,d,seed,ms,ask_ms,tell_ms,n\n";
    for (int seed=100; seed<=107; ++seed) {
        std::shuffle(jobs.begin(),jobs.end(),shuffle);
        for (const auto& [key,d] : jobs) {
            Vec lo(d,-5), hi(d,5), x0(d,1.5), sigma(d,.1);
            int n=0;
            double ask=0,tell=0;
            calls=0;
            auto start=Clock::now();
            auto run_es=[&](auto& state) {
                while (n<budget) {
                    auto tick=Clock::now();
                    auto X=state.ask();
                    ask+=elapsed(tick);
                    Eigen::VectorXd y(X.cols());
                    for (int p=0; p<X.cols(); ++p) y[p]=sphere(d,X.col(p).data());
                    tick=Clock::now();
                    if (state.tell(y)) throw std::runtime_error("early stop");
                    tell+=elapsed(tick);
                    n+=X.cols();
                }
                if (state.result().evaluations!=n) throw std::runtime_error("count");
            };
            if (key=="crfmnes") {
                crmfnes::CrfmnesState state(0,d,x0.data(),lo.data(),hi.data(),.3,pop,seed,1e5,true,false);
                run_es(state);
            } else if (key=="pgpe") {
                pgpe::PgpeState state(0,d,x0.data(),lo.data(),hi.data(),sigma.data(),pop,seed,1000,true,.15,.1,.2,.9,.999,1e-8,1.,true);
                run_es(state);
            } else if (key=="jso") {
                fcmaes::jso::Parameters params;
                params.popsize=pop;
                params.max_evaluations=budget;
                fcmaes::jso::State state(lo,hi,params,seed);
                while (n<budget) {
                    auto tick=Clock::now();
                    const auto& X=state.ask();
                    ask+=elapsed(tick);
                    Vec y;
                    y.reserve(X.size());
                    for (const auto& x:X) y.push_back(sphere(d,x.data()));
                    n+=y.size();
                    tick=Clock::now();
                    state.tell(std::move(y));
                    tell+=elapsed(tick);
                }
            } else if (key.rfind("da-",0)==0) {
                auto r=dual_annealing::optimize_da(0,sphere,d,seed,x0.data(),lo.data(),hi.data(),budget,key=="da-local");
                n=r.evaluations;
            } else {
                fcmaes::archivede::State state(Vec(d,0),Vec(d,1),2,64,budget,seed,true,key=="tiered");
                while (n<budget) {
                    auto tick=Clock::now();
                    const auto& X=state.ask();
                    ask+=elapsed(tick);
                    const auto rows=X.size()/d;
                    Vec ys(2*rows);
                    for (size_t i=0; i<rows; ++i) {
                        const auto* x=X.data()+i*d;
                        double g=1+9*std::accumulate(x+1,x+d,0.)/(d-1);
                        ys[2*i]=x[0];
                        ys[2*i+1]=g*(1-std::sqrt(x[0]/g));
                    }
                    tick=Clock::now();
                    state.tell(ys.data(),rows,2);
                    tell+=elapsed(tick);
                    n+=rows;
                }
                if (state.front().rows==0 || state.nfev()!=size_t(n)) throw std::runtime_error("front");
            }
            const auto ms=elapsed(start);
            if ((calls && calls!=n) || n<budget) throw std::runtime_error("count");
            if (seed>100) std::cout<<key<<','<<d<<','<<seed<<','<<ms<<','<<ask<<','<<tell<<','<<n<<'\n';
        }
    }
}
```

</details>

<details>
<summary>controls.py</summary>

```python
import json
from time import perf_counter_ns
import numpy as np
from fcmaes import astro, crfmnescpp, dacpp, jsocpp, pareto, pgpecpp

p = astro.Cassini1()
center = (p.bounds.lb + p.bounds.ub)/2
shuffle = np.random.default_rng(77)
for seed in range(100,108):
    jobs = ["CR-FM-NES","PGPE","jSO","DA global","DA local",512,1024,2048,4096]
    for j in shuffle.permutation(len(jobs)):
        job = jobs[j]
        rng = np.random.default_rng(seed)
        if isinstance(job,int):
            Y = rng.random((job,5))
            Y /= Y.sum(axis=1,keepdims=True)
            a = pareto.ParetoArchiveND(5,1)
            payload = np.arange(job,dtype=float)[:,None]
            t = perf_counter_ns()
            mask = a.add_batch(Y,payload)
            ms = (perf_counter_ns()-t)/1e6
            assert np.all(mask) and len(a)==job
            result = dict(job=job,ms=ms,n=job)
        else:
            obs = []
            def fun(x):
                v = p.fun(x)
                obs.append(v)
                return v
            opts = dict(bounds=p.bounds,rg=rng,max_evaluations=1024)
            if job != "jSO":
                opts["x0"] = center.copy()
            if job.startswith("DA"):
                mod = dacpp
                opts["use_local_search"] = job.endswith("local")
            else:
                mod = {"jSO":jsocpp,"CR-FM-NES":crfmnescpp,"PGPE":pgpecpp}[job]
                opts["popsize"] = 32
                if job != "jSO":
                    opts["normalize"] = True
            t = perf_counter_ns()
            r = mod.minimize(fun,**opts)
            ms = (perf_counter_ns()-t)/1e6
            assert r.success and len(obs)==r.nfev
            result = dict(job=job,ms=ms,n=len(obs),value=r.fun,best=min(obs),
                          replay=p.fun(r.x),feasible=bool(np.all((r.x>=p.bounds.lb)&(r.x<=p.bounds.ub))))
        if seed > 100:
            print(json.dumps(dict(seed=seed,**result)),flush=True)
```

</details>

<details>
<summary>probes.py</summary>

```python
import numpy as np
from scipy.optimize import Bounds
from fcmaes import crfmnescpp, pgpecpp, dacpp
for d, budget in [(16,40),(128,260)]:
    for mode in [True,False]:
        obs=[]
        def fun(x):
            y=float(x@x)
            obs.append(y)
            return y
        r=dacpp.minimize(fun,Bounds(np.full(d,-5.),np.full(d,5.)),x0=np.full(d,1.5),
                         max_evaluations=budget,use_local_search=mode,rg=np.random.default_rng(101))
        print("da",d,budget,mode,r.nfev,len(obs),r.fun,float(r.x@r.x),r.nit,min(obs))
for module in [crfmnescpp, pgpecpp]:
    d=8
    for norm in [False,True]:
        for budget in [33,64]:
            observed=[]
            def fun(x):
                y=float(-x.sum())
                observed.append(y)
                return y
            r=module.minimize(fun,Bounds(np.full(d,2.),np.full(d,3.)),x0=np.full(d,2.99),
                input_sigma=.9,popsize=32,normalize=norm,max_evaluations=budget,
                rg=np.random.default_rng(101))
            print(module.__name__,norm,budget,r.nfev,len(observed),r.fun,float(-r.x.sum()),np.min(r.x),np.max(r.x),min(observed))
```

</details>
