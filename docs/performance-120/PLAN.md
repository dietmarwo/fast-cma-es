<!-- {% raw %} -->
# fast-cma-es performance baseline and 120-experiment plan

**Benchmark date:** 27 November 2026. **Source:** `master` at `361904d747bbaebfe0d065978fd4e285fdc0fbc0`, package metadata 2.2.0. This report contains **20 proposed experiments for each of six tracks**. The measurements below are from the unchanged checkout. A Release build, 427 recorded workload runs, 61 warm-up runs and separate correctness probes underpin the plan. [Pinned source](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/README.md) · [build configuration](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/CMakeLists.txt)

## Where to start

| Track | Measured result | First action |
|---|---|---|
| CMA-ES | At 128D/P64, native ask/tell took 164.53 ms per 4,096 evaluations. Tell took 34.09 µs per evaluation. The delayed callback probe also exposed an initial coordinate mismatch. | Fix the delayed coordinate and accounting cases, then profile covariance updates and eigensolver refreshes (C01–C09). |
| BiteOpt | At 96D, B=32 reduced median elapsed time from 13.70 to 11.51 ms for 8,192 calls. At 12D/B16, seven seeds each produced just one distinct point in the first 16 proposals. | Fix initial batch generation (B01), then remeasure batching and quality. |
| DE | The native serial loop took 0.78 ms for roughly 4,100 cheap calls. A controlled ask/tell batch returned the maximum finite double after receiving a valid minimum of zero. | Fix best-observed accounting (D01) before using the optimizer’s returned fitness in speed/quality comparisons. |
| MODE | For 4,096 ZDT1 evaluations at P=256, native SBX took 17.50 ms. Tell used 4.23 µs per evaluation and ask used 0.023 µs. | Split tell profiling into ranking, survivor gathering and offspring generation. Try exact two-objective ranking first (M01). |
| Diversifier | With 2,048 cheap QD calls and one worker, CMA_CPP took 36.45 ms, DE_CPP 32.78 ms and BITE_CPP 71.56 ms. Two-worker CMA exceeded its 2,048-call budget in five of seven runs. | Correct shared accounting (V03), then compare scalar and batched Bite archive feedback (V05–V06). |
| MAP-Elites | A supplied 256-cell archive with SBX took 39.80 ms for 2,048 calls. On the default fresh path and the optional CMA path, independent probes reproduced failures. | Restore both public paths and surface worker failure (Q01–Q03). Then try shared-buffer sorting and archive updates (Q06–Q11). |

CMA-ES, BiteOpt, DE and MODE have C++ cores in this repository. Quality-diversity (QD) methods retain elites across descriptor niches. Diversifier and MAP-Elites combine Python orchestration, compiled numerical libraries, and selectable native emitters. For those two tracks the timings identify the exact layer exercised. [Native build targets](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/CMakeLists.txt) · [Diversifier](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/fcmaes/diversifier.py) · [MAP-Elites](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/fcmaes/mapelites.py)

## Baseline measurements

Times are medians across seven runs. Brackets show the 25th–75th percentiles, in milliseconds. A workload run includes the timed operations specified for that table. Seed or repetition 99 is warm-up. Native runs and seedable drivers use 101–107 for measurement. Stage times and total time are summarized separately. All reported rates use actual calls.

### C++ ask/tell

Each timer includes native state construction, proposals, callback arithmetic, tells and the small harness overhead. Ask/tell columns time only those member calls and include one timer pair per batch. Result extraction, MODE population reevaluation and hypervolume calculation run outside the timed interval. P means population size for CMA/DE/MODE. B means requested batch width for BiteOpt with M=1 and its default internal population. Scalar quality is the best callback value observed. Each scalar coordinate has the same shift 0.23 and bounds [−5,5], an easy structured landscape. MODE uses 16D ZDT1 in [0,1], with the additional constraint 0.25−x₀ ≤ 0 in MODE-C. MODE quality is feasible 2D hypervolume (HV) of the selected population with reference (1.1, 6), higher is better.

| Core | d | P or B | Actual calls | Wall ms [IQR] | Ask µs/call | Tell µs/call | Median quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CMA | 16 | 32 | 4,096 | 4.99 [4.77–5.08] | 0.463 | 0.739 | 4.69e-10 |
| CMA | 64 | 32 | 4,096 | 52.87 [51.69–53.46] | 1.991 | 10.782 | 0.000198 |
| CMA | 128 | 64 | 4,096 | 164.53 [158.75–168.73] | 5.143 | 34.088 | 0.234 |
| Bite | 16 | 1 | 8,192 | 4.69 [4.61–5.82] | 0.264 | 0.227 | 2.77e-30 |
| Bite | 16 | 16 | 8,192 | 4.12 [3.17–5.77] | 0.202 | 0.230 | 1.66e-20 |
| Bite | 96 | 1 | 8,192 | 13.70 [13.61–16.10] | 0.649 | 0.714 | 1.16e-21 |
| Bite | 96 | 32 | 8,192 | 11.51 [11.10–13.26] | 0.573 | 0.648 | 1.54e-20 |
| DE | 16 | 32 | 8,192 | 1.83 [1.80–1.93] | 0.150 | 0.065 | 4.93e-13 |
| DE | 16 | 128 | 8,192 | 2.01 [1.94–2.19] | 0.175 | 0.058 | 4.73e-05 |
| DE | 96 | 32 | 8,192 | 5.99 [5.81–6.28] | 0.510 | 0.155 | 77.3 |
| MODE-SBX | 16 | 32 | 4,096 | 5.33 [5.26–6.11] | 0.032 | 1.257 | 6.24 |
| MODE-SBX | 16 | 128 | 4,096 | 11.18 [10.84–11.40] | 0.032 | 2.679 | 5.97 |
| MODE-SBX | 16 | 256 | 4,096 | 17.50 [17.20–17.74] | 0.023 | 4.229 | 5.47 |
| MODE-DE | 16 | 256 | 4,096 | 16.65 [16.63–17.18] | 0.353 | 3.704 | 5.76 |
| MODE-C | 16 | 128 | 4,096 | 11.82 [11.64–12.02] | 0.029 | 2.841 | 4.72 |

### Native minimize and public Python minimize

Release C++ functions are linked directly to native callbacks. Public scalar wrappers use a counted Python callable, which invokes NumPy for shifted Sphere or the repository's native trajectory objective for Cassini1. Each row gives the median best value actually observed. Requested budgets are 4,096 for Sphere (16D) and 512 for Cassini1 (6D). DE can finish a generation above budget and CMA can stop early. Wrapper RNGs transform the Python generator seed before passing it to native code. The two columns include their respective objective implementations, callback costs and seed mappings. Use a matched-trace microbenchmark to attribute a difference specifically to binding overhead.

| Optimizer | Objective | Native calls | Native ms [IQR] | Native best | Python calls | Python ms [IQR] | Python best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bite | Cassini1 | 512 | 4.14 [3.92–4.31] | 20 | 512 | 5.34 [5.18–5.39] | 17.5 |
| Bite | Sphere | 4,096 | 1.94 [1.93–1.98] | 1.37e-22 | 4,096 | 13.15 [12.71–13.24] | 3.55e-21 |
| CMA | Cassini1 | 512 | 3.95 [3.92–4.05] | 29.8 | 512 | 5.27 [5.13–5.50] | 25.8 |
| CMA | Sphere | 4,096 | 4.66 [4.55–4.80] | 5.94e-10 | 3,872–4,096 | 14.42 [14.12–15.12] | 1.02e-09 |
| DE | Cassini1 | 513–548 | 3.84 [3.83–3.94] | 17.1 | 519–547 | 5.13 [4.97–5.71] | 18.5 |
| DE | Sphere | 4,100–4,147 | 0.78 [0.76–0.81] | 9.43e-11 | 4,101–4,152 | 11.16 [10.63–12.05] | 6.66e-11 |

### Public population and MODE interfaces

The Python state loops use one vectorized NumPy evaluation per asked batch. Bounds and objectives match the native state cases. Budget 4,096, dimension 16, scalar P=32, Bite B=16, MODE P=256. Public `modecpp.minimize` uses one objective call per row. In the state loop HV comes from `population()` after tell. In public minimize it comes from the returned evaluated batch. Record both outputs when testing future changes.

| Public entry | Actual calls | Wall ms [IQR] | Median best loss or HV |
| --- | --- | --- | --- |
| Bite-state | 4,096 | 7.97 [7.92–8.00] | 1.73e-18 |
| CMA-state | 4,096 | 14.00 [13.97–14.05] | 5.7e-09 |
| DE-state | 4,096 | 1.99 [1.95–2.15] | 6.53e-06 |
| MODE-minimize | 4,096 | 94.80 [93.54–96.60] | 5.37 |
| MODE-state | 4,096 | 37.19 [33.54–37.52] | 5.39 |

### Diversifier and MAP-Elites end to end

Each public QD run starts with a supplied, initialized archive. The timer includes worker creation/join and all generations. The cheap 12D objective is 1+Σ(xᵢ−0.23)² in [−1,1], with descriptors (x₀,x₁). Positive costs make QD score the sum of reciprocals over finite occupied elites. Coverage is recomputed from finite cells. These cases use 256 niches, one worker process and default Python grid lookup without Numba unless the label specifies otherwise. CVT uses a prebuilt KDTree with 256 fixed normalized reference centers drawn using seed 42. The separate setup benchmark exercises real KMeans. MAP uses 64 generations × 32 divided among workers and disables CMA drill-down. Diversifier uses 256-call restarts, population 32 for CMA/DE and the scalar BITE_CPP path. Real operating-system workers seed themselves from entropy, so these rows are independent repetitions, despite the seeded initial decision matrices.

| Public QD configuration | Calls observed | Wall ms [IQR] | Coverage | QD score | Best elite cost |
| --- | --- | --- | --- | --- | --- |
| DIV-BITE_CPP-grid | 2,048 | 71.56 [67.82–80.32] | 99.2% | 86.28 | 1.001 |
| DIV-CMA_CPP-CVT | 2,048 | 42.76 [38.49–43.57] | 87.5% | 63.35 | 1.746 |
| DIV-CMA_CPP-grid | 2,048 | 36.45 [36.00–40.15] | 86.7% | 59.88 | 1.703 |
| DIV-CMA_CPP-grid-w2 | 2,048–2,144 | 36.61 [31.86–46.31] | 83.2% | 67.17 | 1.692 |
| DIV-DE_CPP-grid | 2,048 | 32.78 [32.30–34.14] | 96.1% | 65.19 | 1.965 |
| MAP-CVT | 2,048 | 45.80 [44.07–54.02] | 88.3% | 101.56 | 1.293 |
| MAP-Iso-grid | 2,048 | 37.83 [34.89–38.18] | 98.8% | 95.64 | 1.448 |
| MAP-grid | 2,048 | 39.80 [38.78–43.45] | 94.9% | 109.17 | 1.306 |
| MAP-grid-w2 | 2,048 | 51.22 [41.73–53.82] | 94.9% | 101.19 | 1.335 |
| MAP-grid4096 | 2,048 | 57.87 [53.81–70.16] | 22.8% | 242.06 | 1.640 |

Two-worker CMA Diversifier used 2,048–2,144 objective calls. In one run the archive’s occupied counter was one below the number of finite elite values. For this comparison the driver verified saved elite costs against their decision vectors. Shared values and descriptors must also be checked under intentionally colliding submissions in V03/Q10.

### QD kernels, setup and application

The kernel measurements process 80 batches of 64 descriptors or submissions on a 4,096-cell archive. The sort row is 80 complete sorts of a dense 4,096-value archive. Kernel timing excludes input generation and later scoring. The JIT run warms Numba before timing. The CVT lookup here uses 4,096 fixed centers. In the public JIT cases, archive geometry, callback and workload configuration match the JIT-free public cases above. Public process runs across environments have independently seeded workers.

| Case | JIT-free ms [IQR] | Warm Numba ms [IQR] |
| --- | --- | --- |
| kernel-grid-lookup | 6.19 [6.01–6.27] | 0.18 [0.14–0.18] |
| kernel-CVT-lookup | 5.89 [5.63–6.31] | — |
| kernel-grid-update | 18.22 [16.67–21.35] | 10.31 [10.27–10.61] |
| kernel-grid-update-stats | 74.38 [70.46–75.71] | 65.20 [63.62–67.25] |
| kernel-sort | 24.44 [24.20–25.91] | — |
| MAP-grid | 39.80 [38.78–43.45] | 38.47 [37.92–40.82] |
| DIV-CMA_CPP-grid | 36.45 [36.00–40.15] | 33.80 [32.80–34.32] |
| DIV-BITE_CPP-grid | 71.56 [67.82–80.32] | 58.47 [57.16–60.17] |

Pure grid lookup fell from 6.19 to 0.18 ms for 5,120 queries, while the full one-worker MAP median moved from 39.80 to 38.47 ms. Timer scope changes the engineering priority. For example, enabling statistics raised the JIT-free batch-update median from 18.22 to 74.38 ms.

The following seeded controls run existing worker functions in one process. In-process QD rows use the cheap objective, 2,048 actual calls and a 256-cell grid. The CVT rows time only niche initialization for 128 centers, 8 samples per center and 2D descriptors in a fresh temporary working directory, followed immediately by a cached read. The application calls the repository’s unmodified 20-joint planar-arm objective, which minimizes the standard deviation of joint angles at each reachable end position. It uses 1,024 calls, 256 grid cells and a final reevaluation of the returned elite coordinates and descriptors. [Arm objective](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/examples/arm.py)

| Component or application | Work units | Wall ms [IQR] | Coverage | QD score |
| --- | --- | --- | --- | --- |
| CVT-cached | 128 centers | 0.56 [0.51–0.57] | — | — |
| CVT-cold | 128 centers | 11.01 [10.66–12.18] | — | — |
| InProcess-BITE_CPP | 2,048 calls | 47.72 [47.05–48.12] | 99.2% | 89.50 |
| InProcess-CMA_CPP | 2,048 calls | 16.53 [15.71–18.01] | 89.8% | 72.70 |
| InProcess-DE_CPP | 2,048 calls | 13.60 [12.77–14.32] | 94.9% | 67.97 |
| InProcess-MAP | 2,048 calls | 22.44 [21.22–24.56] | 91.4% | 104.57 |
| Arm / DIV-CMA_CPP | 1,024 calls | 159.33 [158.07–183.60] | 25.4% | 47.28 |
| Arm / MAP-SBX | 1,024 calls | 164.37 [163.11–167.25] | 46.1% | 146.63 |

### Reproduced correctness and accounting findings

| Probe on the pinned build | Observed outcome | Experiment |
|---|---|---|
| Delayed CMA, 4D bounds [10,20], x0 all 15, two workers | For seeds 101–103 the first two callbacks each saw [20,20,20,20]. Initial dispatch decodes a point before the worker decodes it again. In a separate 60-budget probe these seeds produced 61 callbacks and reported 60. | C01–C02 |
| BiteOpt, 12D/B16, seeds 101–107 | All seven first batches had exactly one unique decision row. Ask repeatedly reads the current initialization slot before feedback advances it. | B01 |
| DE_C, P=8, first tell with costs 0 through 7 | After eight calls the result was 1.7976931348623157e308. Row zero had cost zero. Updating the slot before comparing it with the best slot can turn the comparison into a comparison with itself. | D01 |
| MAP, no supplied archive, grid mode | The public entry raised `NameError` at `self.xs.view()` before any objective calls. | Q01 |
| MAP, supplied archive, one 8-row generation followed by one CMA emitter | The worker raised `AttributeError` in `random_xs_one` because `Archive.get_x` does not exist. The caller received a partial archive after only the eight variation callbacks. | Q02–Q03 |
| MAP factory and fresh Diversifier with `use_stats=True` | Both produced `archive.name == True` and `archive.use_stats == False`. The fourth positional constructor argument is the name. Stats-on kernel timings used the keyword directly. | Q01, V01 |
| Diversifier, two workers, 2,048-call request | Independent per-worker tallies exceeded the budget in five of seven runs. Batched stopping and shared unsynchronized increments both need isolated tests. | V03 |

Five selected baseline test files passed **69 tests** (wheel smoke, MODE regressions, CMA bounds, seeded reproducibility and threaded calls). These probes cover cases outside that selection. Multithreaded counts depend on scheduling. The native stage timings identify which phase is expensive. Attributing costs to particular lines, allocator traffic or CPU-cache behavior is part of the proposed experiments.

### Benchmark environment

Linux x86-64, AMD EPYC 9V74, GCC 14.2, C++17, Release `-O3 -DNDEBUG`, Eigen 3.4.1, CMake 3.31.10, nanobind 3.1.0. Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, optional Numba 0.63.1 with llvmlite 0.46.0. BLAS and OpenMP were limited to one thread. The C++ drivers and seeded controls used CPU 3. Public QD/scalar runs used CPUs 4–5, the optional JIT suite used CPU 6, and the public state suite used CPU 3. Results are observations from one shared host. Small differences on cheap callbacks and process timings need paired confirmation. No hardware counters, broad convergence study or large-worker scaling study were collected in this baseline.

## How to run the 120 experiments

Treat each numbered row as one candidate change with its own control and result. Correctness experiments come first. Record the original failure, then establish a corrected reference before timing later changes that depend on the fix. For example, B01 precedes V06, D01 precedes D18, and Q01/Q02/Q03 precede timing a default MAP run. For shared archive kernels, implement a change in one place, exercise it through both QD clients, and credit its speedup once.

1. **Freeze the comparison.** Archive the base SHA, compiler flags, Python/Eigen/nanobind versions, CPU affinity, actual objective code, initial populations, CVT centers and cache state. Record whether Numba is present and include one cold run plus steady-state runs separately. Use recorded objective tables to validate archive/ranking kernels independently of live search.
2. **Screen a single change.** Use a clean worktree for each trial against its declared parent. Alternate timing order A/B then B/A, warm each binary, use at least seven paired repetitions with fixed seeds 101–107, and match CPU sets. Record total wall time, calls/s, per-stage time, evaluations actually entered, reported evaluations, stop code and quality. Profile suspected kernels to distinguish computation from allocator and binding costs. Measure heap allocations, peak RSS and hardware counters where available, in a separate profiling run.
3. **Use the right quality test.** For purportedly exact work, check the full candidate/feedback trace, candidate order, dimensions, stop code, status and returned samples on deterministic fixtures. Eigen reorderings that alter floating-point rounding require residual/ULP checks and the search holdout below if trajectories diverge. Changes to donors, RNG streams, feedback order, covariance schedules, parent sampling and asynchronous visibility are search experiments.
4. **Validate search on held-out seeds.** After parameter selection, use 30 unused seeds (1001–1030) and repeated independent runs for unseedable public process paths until V02/Q04 exist. Use the same draw/initialization inputs when possible. Test both equal actual calls and equal wall time. Freeze a baseline wall-time allowance per workload and record final quality plus time/evaluations to target, including failures. For parallel cases, fix worker counts and report worker-seconds as well as elapsed time.
5. **Keep, revise or reject.** A speed change should deliver at least 5% median improvement on its nominated end-to-end workload, a paired bootstrap 95% interval for baseline/variant wall time above 1, and no repeatable greater-than-3% regression on a companion workload. Resample whole seed/run pairs, report intervals and flag inconclusive small runs. Treat a correctness fix as successful when its regression passes, document any runtime cost, and score subsequent speed changes against it. A search change needs better primary quality or faster target attainment on holdout, with no material degradation on the second workload or feasibility/coverage metric. Before opening holdout results, set companion tolerances per workload. Suggested starting values are 2 percentage points for coverage or success rate and 3% relative for HV or positive QD score. Report uncertainty and keep unresolved trade-offs opt-in. These thresholds are proposed decision criteria.
6. **Verify the combination.** Rebuild only the accepted candidates. Compare the integrated result against the pinned and corrected references on a second host and on the installed wheel. Remove each winning change in turn to check whether gains survive interaction. Run the full Python wheel suite and native CTest suite, race/ownership checks for threaded changes, and archive save/load/resume checks for QD.

| Track | Holdout workload and output to record |
|---|---|
| CMA-ES | Shifted Sphere, rotated 10⁶-conditioned ellipsoid, Rosenbrock, the repository’s active-bound tests and Cassini1. Use d=8/16/64/128/256 where applicable, P=16/32/64/128, synchronous and delayed W=2/8. Record best observed and reevaluated loss, targets and stop reason. |
| BiteOpt | Heterogeneous per-coordinate shifts, rotations, Rastrigin, rounded plateaus, noisy observations and Cassini1. Test depth 1/2/8 and batch 1/8/32/64, including P−1 and P+1 initialization boundaries. Record distinct evaluated points, calls to target, loss and per-depth working set. |
| DE | Bounded continuous and integer masks, narrow-box/high-repair cases, smooth/multimodal controls and Cassini1. Test serial, batch, delayed W=2/8 and one straggler. Log every callback ID and cost, best-ever result, actual budget, repair/restart counts and worker lifetimes. |
| MODE | ZDT1/ZDT3, 3- and 5-objective DTLZ2, the constrained regression fixtures and objective duplicates. Sweep P=32/128/256/1024. Compute feasible hypervolume (HV) with a preregistered reference point and inverted generational distance (IGD) against frozen analytic front samples. Report nondominated count, extremes, feasibility and front ranks. |
| Diversifier | Projected Sphere and 20-joint arm, grid and identical frozen CVT centers, single and mixed native emitters. Use W=1/2/8 and cheap/expensive callbacks. Record actual calls, wall and worker time, score/coverage over time, restart counts and per-emitter contributions. |
| MAP-Elites | Shared QD fixtures plus sparse/dense archives, collisions and restored archives. Compare grid/CVT, warm/cold starts, optional Numba, stats, both variation operators and native CMA. Record coverage from finite cells, elite score distribution, QD score, consistent x/d/y, counts and disk/setup time. |

For scalar target reporting, preregister Sphere at 1e−6, a normalized rotated ellipsoid at 1e−4 and Cassini1 at 20 and 10 in the package’s objective units. For longer quality runs, start at 4,096/32,768 actual calls on synthetic problems, 512/4,096 on Cassini1 and 2,048/16,384 on QD. Include a nontrivial external evaluator or a GIL-releasing callback with controlled latency (0.1 and 5 ms) so that an improvement on cheap callbacks has a meaningful application check. For QD compare the same objective sign/offset, bounds, capacity, geometry and feasibility policy. The repository computes a reciprocal sum if all occupied costs are positive and a negative-cost sum otherwise. Log occupied count, median occupied-elite cost and best elite alongside that score.

Record one line per trial and workload with `id, base_sha, variant_sha, entry, objective, d, P_or_B, seed, workers, calls_requested, calls_observed, wall_ms, ask_us, tell_us, peak_bytes, quality, stop, speed_ratio, interval, decision, reason`. Preserve raw rows, failing seeds and the reference used for each decision.

## 1. Active CMA-ES

At 128D, `tell` cost 34.09 µs per evaluation. Time covariance formation, eigendecomposition and the supplied-points inverse separately before choosing among C04–C11. Exercise synchronous `minimize`, delayed `minimize`, ordinary `tell` and `tell(ys, xs)` independently. [C++ core](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/acmaesoptimizer.cpp) · [evaluator](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/include/evaluator.h) · [binding](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/bindings/acma_bindings.cpp) · [bounds regressions](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/tests/test_cmaes_bounds.py)

| ID | Proposed change to try | Deciding measurement or invariant |
|---|---|---|
| C01 | Queue encoded coordinates consistently in the delayed initial fill. The worker already decodes and clips each request. | Log the first worker window with asymmetric bounds and noncentral x0. Compare objective coordinates with the points used in the update and reevaluate the incumbent. |
| C02 | Reserve delayed evaluation slots before dispatch, retire outstanding slots on stop, and report the callbacks that actually ran. | Budgets 1, W−1, W and 60 with workers 2/8 and uneven latencies. Check target-stop behavior, counts and latency. |
| C03 | Shift fitness history from back to front or use a circular buffer. The forward loop propagates the previous first entry through the history. | Feed alternating best values into a reference history. Verify extrema and stop codes 4/5, then rerun flat and noisy objectives at equal calls. |
| C04 | Use one matrix product for `BD * arz` across a generation in `ask_all`, preserving Gaussian draw order. | Ask time at dimensions 8–256 and populations 16–128. Check repaired flags, feasibility and near-bound results. |
| C05 | Reuse a dimensioned `SelfAdjointEigenSolver` in `updateBD`. | Allocations per refresh and tell time at 64/128/256D. Check eigen residuals and condition stop. |
| C06 | Mirror the upper triangle of C into the lower in place before the solve. | Bytes written and refresh time. Require symmetry and matching eigenvalues on nearly singular input. |
| C07 | Accumulate the positive rank-one and rank-µ covariance terms with Eigen self-adjoint operations. | Hot update time and relative covariance error. Include the zero-norm negative-update fallback. |
| C08 | Fuse column weighting and subtraction for the active negative update, eliminating the full Cneg temporary. | Peak bytes and covariance time at 128D. Verify negative coefficient cap, positive definiteness and best histories. |
| C09 | Replace explicit `BD.inverse()` on supplied points with the structured transform using B transpose and inverse diagonal D. | Test permuted external tells, delayed updates and ill-conditioned B/D. Check reconstruction residual and stop reason. |
| C10 | Compact repaired columns into one dense block before `repairArz` when many samples clip. | Bound-hit rates 0%, 10%, 90%. Compare recalculated z for changed columns and unchanged z elsewhere. |
| C11 | Gather sorted parents once into reusable workspaces for weighted means and covariance. | Allocation count, tell time and tie ordering at odd/even population sizes. |
| C12 | Decode and clip directly into the batch callback buffer in `Fitness::values`. | Synchronous cheap Python callback and Cassini1, normalize on/off. Compare callback shape, coordinates and order. |
| C13 | Transfer ownership of the native ask result to the NumPy matrix capsule with compatible strides. | Public ask time and bytes copied. Keep earlier arrays alive across later asks, solver deletion and Python mutation attempts. |
| C14 | Fill all fitness and point columns at once in the ordinary asked-batch tell path. | Compare final and per-generation results, penalties and stop transitions with scalar tell at P=16/128. |
| C15 | Cache invariant path coefficients and recompute diagonal extrema only when the eigensystem changes. | Small-D tell microbenchmark and plateau stop tests. Verify iteration-dependent terms still update. |
| C16 | Expose a public vectorized synchronous objective using the existing batch trampoline. | NumPy batch, scalar callback and multiprocessing controls. Check wrong lengths, nonfinite rows and ordered results. |
| C17 | Sweep explicit covariance update intervals around the existing lazy schedule. | At 64–256D record refresh count, equal-call loss and time to target on rotated ellipsoid and Rosenbrock. |
| C18 | Trial an initial diagonal-only adaptation phase with an explicit transition to full covariance. | Test diagonal and rotated objectives with anisotropic bounds. Measure time saved versus delayed correlation learning. |
| C19 | Trial clipped-sample z repair as a selectable workload policy, using the existing `repair_z` switch as the control. | Pair existing boundary tests, interior Sphere and narrowed Cassini1 retry boxes. Record hit rates and premature stops. |
| C20 | Trial a larger restart population after stagnation under a fixed total budget. | Compare with fixed-pop independent restarts on Rastrigin and Cassini1. Record success probability, evaluations and wall time to target. |

## 2. BiteOpt

Batch width and search population are different controls. The default population is 9 + 3d. First recover the wasted initialization calls, then compare batch policies from the corrected baseline. Population insertion already uses binary search and pointer `memmove`. [Wrapper](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/biteoptimizer.cpp) · [ask/tell engine](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/include/biteopt_asktell.h) · [population storage](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/include/biteaux.h) · [generators](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/include/biteopt.h)

| ID | Proposed change to try | Deciding measurement or invariant |
|---|---|---|
| B01 | Snapshot a different unevaluated initialized population member for each pending batch slot and commit it against its own saved data. | Compare the initial candidate multiset with serial initialization, replay descending/equal costs, and verify unique rows, including the final short batch and M>1. |
| B02 | Bypass the batch sort for a singleton while retaining single-candidate tell feedback. | Scalar native throughput, stall counts, NaN sanitization and fixed-seed trace equality. |
| B03 | Pass the contiguous Eigen cost vector directly into the deep tell routine. | Allocation count and B=1/8/64 tell time. Check validation on empty and wrong-length costs. |
| B04 | Store the proposing optimizer once per pending batch. All proposals are generated by the same current optimizer before feedback. | M=1/2/8. Verify proposer attribution through mid-tell switches, ties and stalls. |
| B05 | Allocate ask scratch lazily for participating deep optimizers. The current growth path sizes all levels at once. | Peak memory and startup time at d=128, M=1/8/32, B=64. Check level re-entry. |
| B06 | Write real-valued candidates to their saved output slots directly during `askOne`. | Bytes copied per ask, identical returned rows and intact best parameters after rejection. |
| B07 | Defer restoring a real-vector copy on tell until a path actually needs the real-valued candidate. | High-rejection replay. Verify best vectors, initialization, deep pushes and selector feedback. |
| B08 | Use a lightweight synchronous candidate path with one saved proposal, keeping the existing ask-compatible generator set. | Compare serial native and Python minimization with B=1. Require identical generator choices and call counts. |
| B09 | Give the NumPy binding an owned candidate slab filled directly from saved ask rows. | Public B=8/64 ask time, capsule lifetime, short-batch shape and read/write contract. |
| B10 | Keep a contiguous sorted cost index beside population pointers to reduce binary-search pointer chasing. | Insertion time at population 57/297/777. Confirm equal-cost order and consistency on population resize and copy. |
| B11 | Short-circuit centroid-distance calculations in parallel-population selection when a partial distance cannot win. | Record visited coordinates and chosen population at 96D. Preserve tie direction and accumulator arithmetic. |
| B12 | Split the random boundary-wrap pass from the contiguous integer-to-real decode pass to permit vectorization with the existing scale arrays. | Break out decode cost at 16/96D. Check extreme ranges, wrap paths and exact bounds. |
| B13 | Specialize feedback sorting for tiny batches using the same sanitized cost and original-index ordering. | Cross over against std::sort at B=2/4/8/16 with equal, NaN and infinite costs. |
| B14 | Replace sparse selector-pointer snapshots with compact stable selector IDs translated on feedback. | Workspace bytes and tell time with deep M. Replay selector increments/decrements and RNG sequence. |
| B15 | Try an explicit population cap at high dimension. | Test caps 96/160/256 versus 9+3d at d=32/96/256. Compare warmup fraction, final loss and success at equal calls/time. |
| B16 | Vary batch width by search phase, using small batches until the population fills. | After B01, test grow-to-8/32 versus constant width. Compare candidate diversity, expensive batched objective utilization and quality. |
| B17 | Trial depth M=2/4 as a restart allocation policy with a fixed total budget. | Match M=1 independent restarts on Rastrigin, rotated ellipsoid and Cassini1. Record success and memory. |
| B18 | Stop and restart according to rolling improvement per evaluation in place of a fixed stall horizon in an opt-in policy. | Paired plateaus, smooth and noisy losses. Charge all initialization calls after each restart. |
| B19 | Trial rejection of exact duplicate population costs after the population fills. | Tie-heavy quantized problems and continuous controls. Compare retained genotype diversity, time to target and stagnation. |
| B20 | Trial feeding back a completed batch in ask order as an alternative to best-first feedback. | B=1 parity, B=8/32 quality under both fixed budgets, tied/invalid costs and deep-level behavior. |

## 3. Differential evolution

The cheap native DE loop makes callback and state-accounting overhead visible. Separate serial, delayed and batch paths throughout. The serial path resamples coordinates outside the box and immediately evaluates a temporal-locality trial on an improvement. The batch path queues that trial. [C++ DE](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/deoptimizer.cpp) · [worker queues](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/include/evaluator.h) · [Python binding](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/bindings/de_bindings.cpp)

| ID | Proposed change to try | Deciding measurement or invariant |
|---|---|---|
| D01 | Update the best observed finite pair independently of the population slot used as mutation anchor. | Replay the row-zero minimum, later improvements to the active best, restarts, nonfinite and tied scores. Check result.x, result.fun and stop target in all three paths. |
| D02 | Add a strict callback-budget mode that also counts temporal-locality evaluations inside the serial generation. | Budgets 1, P−1, P and 512. Audit actual calls, last partial generation and best result. Benchmark legacy counting separately. |
| D03 | In delayed DE, recycle slot IDs on completion and keep their candidate and population index together. | Delay one callback for more than 10P later completions. Assert every returned cost belongs to the correct candidate and population index. |
| D04 | Move completed worker tickets through the bounded queues and reclaim tickets still queued at shutdown. | Count allocations, live tickets and latency at W=2/8 with early stop, a slow tail and callback exceptions. |
| D05 | Give scalar `Fitness::eval` a stack scalar output path for DE. | Native Sphere and Cassini1 callback cost, finite-penalty conversion and termination semantics. |
| D06 | Read DE donor and target columns as Eigen views in serial mutation. | 16/96/256D timing and candidate traces, including coordinate resampling. |
| D07 | Write batch mutations straight into the available asked-matrix column. | Ask time at P=32/128. Verify pending index mapping when temporal trials precede ordinary proposals. |
| D08 | Fuse batch crossover and bound clipping in a single pass over the chosen target. | Measure cheap ask time at d=16/96. Preserve forced crossover dimension and conditional random draws. |
| D09 | Precompute sorted integer coordinate indices and their count for `modify`. | Integer densities 0/1/50/100%. Replay random draws and validate the existing truncation/resampling behavior. |
| D10 | Store the temporal improvement index and vector in one movable queue entry. | High-acceptance runs. Count copies and confirm FIFO target/row identity on repeated target indices. |
| D11 | Specialize the serial trial loop once per run for uniform versus normal bound repair, instead of selecting the sampling path at each failed coordinate. | High-bound-hit serial run. Preserve the actual sampling formula, draw order and unconstrained behavior. |
| D12 | Reuse rolling min/max work buffers for the ten-column normal-sampling history. | Use x0 and anisotropic sigma. Compare sigma, mean and proposals against the baseline on alternating generations. |
| D13 | Reuse storage for the serial trial and the optional second temporal candidate. | Allocation counts by accepted/rejected trial and observed best on both callback paths. |
| D14 | Return a capsule-owned column-major candidate buffer as a C-contiguous Python row batch. | Measure repeated public asks and `population()`; keep old arrays alive, and check shape and memory independence. |
| D15 | Validate native donor pool size at construction to fail promptly when exclusion leaves too few distinct indices. | Bound tests below/at minimum P, including p==bestI and p!=bestI. Time valid tiny populations. |
| D16 | Try sampling donors without rejection using exclusion-aware index mapping. | Donor frequencies, RNG-independent quality and nanoseconds per ask at P=4/8/32/128. |
| D17 | Evaluate temporal-locality trials only after a configurable acceptance threshold. | Count follow-up calls, absolute improvements and time to target on Sphere, Rastrigin and Cassini1. |
| D18 | Compare rank-biased p-best mutation anchors with the implementation’s current anchor. | Under the D01 incumbent fix, assess multimodal success, 96D loss and converged-population collapse. |
| D19 | Trial an age-dependent reinitialization schedule using measured diversity. | Compare fresh starts per 1,000 calls, plateau escape probability and noisy-objective behavior. |
| D20 | Trial success-weighted F/CR adaptation in place of even/odd generation oscillation. | Equal calls and wall time on correlated, mixed-integer and Cassini1 holdouts. Record parameter history and failed runs. |

## 4. MODE

At P=256, the native SBX run spent 4.23 µs per evaluation in tell and 0.023 in ask. Tell performs survivor selection and prepares the next SBX population. The DE variant moves proposal work into ask. Begin with rank computation and gathering in both variants. [MODE selection and variation](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/modeoptimizer.cpp) · [binding](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/_fcmaescpp/bindings/mode_bindings.cpp) · [public driver](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/fcmaes/modecpp.py) · [regression cases](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/tests/test_mode_regression.py)

| ID | Proposed change to try | Deciding measurement or invariant |
|---|---|---|
| M01 | Dispatch the unconstrained 2-objective ranking case to a sorted sweep with a prefix-maximum tree that computes every front. Group identical pairs before inserting them. | Compare integer levels against existing pairwise sorting for duplicates, equal x or y, signed zero, chains and antichains. Benchmark P=32–1024. |
| M02 | Specialize one-objective feasible ranks using sorted distinct value groups. | Compare every level and output index, including reversed initial ordering, ties and the first generation’s sentinels. |
| M03 | Stop a generic pairwise dominance comparison as soon as each point is better in at least one objective. | Comparisons per pair in 3/5/10 objectives. Check exact levels on staircase and random fronts. |
| M04 | Replace per-point dominance vectors with compact reusable adjacency storage in the generic sorter. | Measure peak bytes and tell time on dense domination and largely incomparable inputs. |
| M05 | Bucket survivor indices by integer front level in one stable pass. | Many thin fronts versus one full front. Preserve iteration order within complete levels. |
| M06 | Gather only the selected columns into reusable survivor matrices at the end of `pop_update`. | Compare copied bytes, allocations and survivor identities when the last front fits exactly or only partly. |
| M07 | Reuse per-objective order and distance buffers for split-front crowding. | Large nondominated front. Preserve boundary sentinels and the current treatment of constant objectives. |
| M08 | Select only the required prefix of crowding scores with an explicit tie rule compatible with baseline ordering. | Oversized split fronts, equal distances and boundary ties. Compare survivor identities and time. |
| M09 | Accumulate constraint ranks and per-point violation counts while scanning each sorted constraint. | 1/4/16 constraints, mixed feasibility. Verify feasibility-first ordering and tie weights. |
| M10 | Skip redundant ranking work when every point is feasible, preserving the existing tie order. | All feasible, one infeasible and all infeasible cases. Compare exact survivor order and tell time. |
| M11 | Bind SBX parent blocks as views and write paired offspring directly into their destination columns. | Allocation bytes at d=16/128 and P=64/256. Preserve beta signs, draw order and clipping. |
| M12 | Hoist generation-specific crossover and mutation exponents out of the coordinate loops. | Compare variation time and offspring at mutation probabilities 0, default and 1. |
| M13 | Fill DE offspring directly from donor views in the second half of popX. | DE ask throughput and coordinate traces. Verify the parent half stays untouched while proposing. |
| M14 | Cache active integer indices for the MODE DE modifier. | Sparse/dense masks. Validate the public wrapper’s existing handling of integer masks in SBX mode. |
| M15 | Transfer the whole precomputed SBX block on ask when its cursor is aligned. | Test odd requested population sizes, switching DE/SBX and user-supplied populations. |
| M16 | Map C-contiguous Python objective rows as Eigen columns for tell when lifetimes permit. | Compare public/native tell and copied bytes. Test noncontiguous wrapper inputs, shape errors and independent solver instances. |
| M17 | Offer a matrix objective in the public serial MODE loop. | NumPy batch versus expensive row callbacks. Require ordered objective/constraint values and actual evaluation counts. |
| M18 | Trial an adaptive mixture of DE and SBX via `tell_switch`. | Compare fixed-budget HV, IGD and feasible fraction on ZDT1, disconnected ZDT3 and constrained holdouts. |
| M19 | Trial elite-parent bias through nonzero `pareto_update` in DE generations. | Compare extremes, duplicates and IGD with unbiased selection at P=64/256. |
| M20 | Inject parents periodically from a bounded external archive. | Compare with equal-cost no-injection runs scored by the same measurement archive. Check feasible HV, diversity and the public population contract. |

## 5. Diversifier

The orchestration loop is Python. CMA_CPP and DE_CPP use native population ask/tell. BITE_CPP calls `update_archive` once per scalar callback. With 2,048 cheap evaluations, the seeded in-process loop took 47.72 ms with Bite and 16.53 ms with CMA. Compare each emitter with its own control. [Diversifier](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/fcmaes/diversifier.py) · [shared improvement calculation](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/fcmaes/mapelites.py) · [public Bite ask/tell](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/fcmaes/bitecpp.py)

| ID | Proposed change to try | Deciding measurement or invariant |
|---|---|---|
| V01 | Pass `use_stats` by keyword when Diversifier constructs its archive. | Fresh archives with stats on/off. Verify per-cell counts and moments before comparing throughput. |
| V02 | Accept a master seed and spawn independent per-worker and per-restart streams, including initialization. | Replay one worker exactly. Record seeds and verify stream independence at 2/8 workers. |
| V03 | Replace unprotected shared evaluation increments with chunk reservations and reconciliation of unused work. | Budgets around B and W×B, early stops and short batches. Compare against independent objective counters. |
| V04 | Surface child failures at join and stop speculative work after a fatal worker failure. | Inject an exception after a known number of calls. Require a parent-visible error and verify cleanup and normal join time. |
| V05 | Replace the scalar Bite minimization callback with public `Bite_C` at batch size one first. | Compare archive contents and transformed costs step by step. Count objective calls after the global budget runs out. |
| V06 | Batch Bite niche queries and improvement scores using the actual length of each ask. | After B01/V05, compare widths 4/16/32 at equal calls/time. Include within-batch collisions and short final batches. |
| V07 | Accept a batched QD callback returning separate objective and descriptor arrays. | Cheap vectorized callback and planar arm with scalar fallback. Preserve order and reject malformed batches before writes. |
| V08 | Gather old elite values by indexing a shared numeric buffer with the niche-ID vector. | Repeated IDs must see the same pre-write baseline as the single-worker reference. Time sparse/full archives. |
| V09 | Fuse empty-niche score normalization into a dense numeric kernel. | Check the returned improvement vector for all empty, all occupied, mixed, negative and tied scores, including the 1e−9 offset. |
| V10 | Sort the transformed scores once per generation and reuse the result for stall detection and history. | Verify restart and tell counts on early stalls. Compare P=16/128 timing. |
| V11 | Reset a compatible native solver between restarts to reuse allocated state. | Fresh-instance control with identical seeds. Check covariance, pending tickets, RNG cache and prior best are reset. |
| V12 | Reuse worker-local objective, descriptor and score buffers across same-size generations. | Peak allocated bytes and elapsed time for alternating full/partial batches. Check data retained by callers. |
| V13 | Offer inline execution when explicitly running one worker. | 256–16,384 evaluations, grid/CVT and costly objectives. Check exceptions, process isolation expectations and archive quality. |
| V14 | End a sequential solver chain before constructing another stage once its budget has been consumed. | Finish exactly on and inside a stage boundary. Record extra callbacks, constructor count and best vector. |
| V15 | Pass the previous stage’s selected x0 into a chained BITE_CPP run. The dispatcher currently supplies None. | Check a known warm start reaches Bite, then compare DE→Bite and independent restarts on arm/multimodal QD. |
| V16 | Avoid eager RNG draws for default mean and sigma when explicit values were supplied. | Count draws and initialization time. Test supplied/absent parameters and validate the revised seed stream as a search change. |
| V17 | Vary per-restart evaluation budgets with the rate of newly occupied niches. | Compare fixed 256/1,024-call restarts. Record coverage over time, QD score and restart overhead. |
| V18 | Allocate worker time to CMA_CPP or DE_CPP using recent archive improvement per callback. | Compare equal shares and single emitters at matched calls and worker-seconds. |
| V19 | Start a fraction of restarts near underexplored occupied niches. | Compare with random starts on sparse/near-full archives. Record early coverage, elite quality and repeat visits. |
| V20 | Buffer worker-local submissions and synchronize with the shared archive periodically. | Compare sync intervals 1/4/16 batches. Measure staleness, score, coverage and contention on colliding descriptors. |

## 6. MAP-Elites

Python/NumPy handle the archive, processes and variation. CVT uses scikit-learn KMeans and KDTree, grid lookup has an optional Numba kernel, and the optional CMA emitter uses C++. Public MAP timings use a supplied archive with CMA emission disabled. The fresh-archive and CMA failures have separate probes. [Archive and emitters](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/fcmaes/mapelites.py) · [shared arrays](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/fcmaes/retry.py) · [arm workload](https://github.com/dietmarwo/fast-cma-es/blob/361904d747bbaebfe0d065978fd4e285fdc0fbc0/examples/arm.py)

| ID | Proposed change to try | Deciding measurement or invariant |
|---|---|---|
| Q01 | Initialize new archives via their own decision-vector view and forward the stats flag by keyword in fresh archive factories. | Run the public entry from scratch on grid/CVT and `empty_archive` with stats on/off. Check decision bounds and finite cells. |
| Q02 | Repair `random_xs_one` to read an existing decision row and sample an eligible ranked elite. Define the sparse case when best_n exceeds occupancy. | Enable the CMA emitter after one generation and on a sparse resumed archive. Check parent ID, fitness and actual callbacks. |
| Q03 | Check MAP worker exit codes and propagate child error context. | Fail one worker midway through a generation. Check parent outcome, joined workers and normal completion time. |
| Q04 | Pass a root seed through initial population, worker streams and CVT initialization. | Repeat fresh and cached single-worker runs. Record scheduling effects when multiple processes write. |
| Q05 | Keep a persistent worker pool across outer MAP iterations. | Compare 1/10/100 outer iterations at the same total inner generations. Preserve sorting barriers, child-local views and cleanup. |
| Q06 | Avoid the Python list conversion in `get_ys` on the internal `argsort` path by using a shared-buffer view. | Sort 256/4,096/65,536 niches. Preserve public snapshot ownership and exact index ranking. |
| Q07 | Index the sorted niche array directly with the random parent-index vector. | Compare seeded parent IDs and gather time on sparse and near-full archives, including replacement sampling. |
| Q08 | Incrementally merge changed cells into the occupied ranking when few elites changed. | Compare with full sort at 1/50/99% occupancy, including ties and load/resume. Measure crossover. |
| Q09 | Batch archive submissions by niche, applying equal-value tie handling and statistical updates in submission order. | Replay all-to-one and all-distinct batches. Match serial x/d/y, counts and moments. |
| Q10 | Use striped process-shared locks for complete winner, statistics and first-occupancy transactions. | Same-cell and disjoint concurrent inserts. Recompute elite x/d/y, occupied count and worker scaling. |
| Q11 | Update per-cell statistical vectors using shared numeric views or a small native kernel with the existing online formulas. | Stats-on time, counts, moments and extrema, including after resume. |
| Q12 | Add a C++ fallback for grid lookup with the current flattening and clamping convention. | Compare JIT-free, warm Numba and native lookups at exact boundaries, outside bounds and non-perfect-power capacity. Include startup. |
| Q13 | Ask the existing KDTree for indices only, retaining the default leaf size initially. | Warm batch lookups at sizes 1/32/256 and descriptor dimensions 2/8/32. Verify nearest IDs on midpoint ties. |
| Q14 | Cache CVT centers with an atomic versioned write keyed by geometry and initialization settings. | Concurrent cold start, warm read and interrupted write. Verify center arrays and existing archive reload. |
| Q15 | Use broadcast bounds in SBX mutation in place of two fully tiled matrices. | Peak memory at P=32/512 and d=12/256. Compare offspring and RNG draws, including bound repair. |
| Q16 | Apply SBX powers only for entries surviving the later mask, preserving random draws and array order. | Operator and full-generation time. Check both tails of the current crossover formula and masked coefficients. |
| Q17 | Fuse Iso+Line arithmetic into reusable offspring storage after drawing the two existing random arrays. | Offspring equality, bounds, alias safety and allocation count at small and large batch sizes. |
| Q18 | Bulk initialize/reset shared archive arrays. | Cold time at 256/4,096/65,536 cells, stats on/off. Keep niche centers and reset all empty sentinels. |
| Q19 | Vary the mix of SBX and Iso+Line emitters with coverage. | Equal actual calls and worker time on projected Sphere and arm. Track coverage, score and elite quantiles under fixed geometry. |
| Q20 | Trial a configurable share of native CMA drill-down once Q02 restores the emitter. | Compare 0/10/30% of calls at sparse/full occupancy. Record new niches, score and total time at equal actual calls. |

## Replay appendix

The source listings and CSV observations are embedded below so this Markdown file can be used independently. Export the fenced contents of each **Replay file** subsection under the indicated filename. The following commands use a Linux checkout, a Python environment with NumPy/SciPy/scikit-learn/loguru installed, nanobind on the CMake/Python path and Eigen available to CMake. Pin the versions listed above for close numerical comparisons. Use the common Python environment without Numba for the base runs, then add the indicated Numba/llvmlite versions for the optional JIT rerun.

```bash
git clone https://github.com/dietmarwo/fast-cma-es.git fast-cma-es
git -C fast-cma-es checkout --detach 361904d747bbaebfe0d065978fd4e285fdc0fbc0
cmake -S fast-cma-es -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python)"
cmake --build build --parallel 3
# Set EIGEN_INCLUDE to the directory containing Eigen/Core.
c++ -O3 -DNDEBUG -std=c++17 -Ifast-cma-es/_fcmaescpp/include -I"$EIGEN_INCLUDE" native.cpp build/libfcmaes_backend.a -pthread -o native
c++ -O3 -DNDEBUG -std=c++17 -Ifast-cma-es/_fcmaescpp/include -I"$EIGEN_INCLUDE" direct.cpp build/libfcmaes_backend.a -pthread -o direct
export PYTHONPATH="$PWD/fast-cma-es${PYTHONPATH:+:$PYTHONPATH}"
export FCMAES_EXT_PATH="$(ls "$PWD"/build/_fcmaes_ext*.so)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
taskset -c 3 ./native > native.csv
taskset -c 3 ./direct > direct.csv
taskset -c 3 python state.py state.csv
taskset -c 4,5 python python_bench.py base public.csv
taskset -c 3 python qd_components.py components.csv
taskset -c 3 python arm_bench.py fast-cma-es/examples/arm.py arm.csv
taskset -c 2,7 python audit.py
# Optional JIT configuration, after installing Numba and llvmlite
taskset -c 6 python python_bench.py jit jit.csv
```

Choose available local CPU IDs if the displayed CPUs are unavailable. Raw run identifiers for public QD label repetitions and the initialization seed. The workers use independent entropy. CSV conventions are defined by the generating source. For public QD, q1/q2/q3 are coverage, QD score and best cost, and diag is reported occupancy minus the finite-cell count. For public scalar minimize, q1/q2/q3 are observed best, returned best and reevaluated result, and diag is the stop code. Warm-up rows are included in raw observations.

<details>
<summary>Replay file · native.cpp</summary>

```cpp
#include "acmaesoptimizer.hpp"
#include "biteoptimizer.hpp"
#include "deoptimizer.hpp"
#include "modeoptimizer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
using Eigen::MatrixXd;
using Eigen::VectorXd;
double us(Clock::time_point a, Clock::time_point b) {return std::chrono::duration<double,std::micro>(b-a).count();}
double sphere(const double* x, int d) {
    double v=0; for(int j=0;j<d;j++){double t=x[j]-.23; v+=t*t;} return v;
}
struct Metrics {int n=0; double ask=0,tell=0,quality=INFINITY;};
template<class State> void scalar(State& s,int budget,Metrics& m){
    while(m.n<budget){
        auto a=Clock::now(); MatrixXd x=s.ask(); auto b=Clock::now();
        if(!x.cols()) break;
        VectorXd y(x.cols()); for(int i=0;i<x.cols();i++) y[i]=sphere(x.col(i).data(),x.rows());
        m.quality=std::min(m.quality,y.minCoeff());
        auto c=Clock::now(); int stop=s.tell(y); auto e=Clock::now();
        m.ask+=us(a,b); m.tell+=us(c,e); m.n+=x.cols(); if(stop)break;
    }
}
void problem(const double* x,int d,double* y,bool constrained){
    double sum=0; for(int j=1;j<d;j++)sum+=x[j]; double g=1+9*sum/(d-1);
    y[0]=x[0]; y[1]=g*(1-std::sqrt(x[0]/g)); if(constrained)y[2]=.25-x[0];
}
double hv(const MatrixXd& rows,bool constrained){
    std::vector<std::pair<double,double>> f;
    for(int i=0;i<rows.cols();i++){double y[3];problem(rows.col(i).data(),rows.rows(),y,constrained);
        if((!constrained||y[2]<=0)&&y[0]<1.1&&y[1]<6) f.emplace_back(y[0],y[1]);}
    std::sort(f.begin(),f.end()); double last=6,h=0;
    for(auto [x,y]:f) if(y<last){h+=(1.1-x)*(last-y);last=y;}return h;
}
void run(std::string kind,int d,int p,int budget,int seed){
    VectorXd lo=VectorXd::Constant(d,-5), hi=-lo, x0=VectorXd::Zero(d), sigma=VectorXd::Constant(d,.3);
    Metrics m; auto t=Clock::now(); double total=0;
    if(kind=="CMA"){
        acmaes::AcmaState s(0,d,x0.data(),lo.data(),hi.data(),sigma.data(),budget,-INFINITY,-1,p/2,p,1,seed,true,true,-1);
        scalar(s,budget,m); total=us(t,Clock::now());
    }else if(kind=="DE"){
        differential_evolution::DeState s(0,d,seed,lo.data(),hi.data(),nullptr,nullptr,0,nullptr,200,p,.5,.9,.1,.5);
        scalar(s,budget,m); total=us(t,Clock::now());
    }else if(kind=="Bite"){
        biteopt::BiteState s(0,d,x0.data(),lo.data(),hi.data(),seed,1,0,p,budget,-INFINITY,0);
        scalar(s,budget,m); total=us(t,Clock::now());
    }else{
        bool con=kind=="MODE-C";lo.setZero();hi.setOnes();
        mode_optimizer::ModeState s(0,d,2,con,seed,lo.data(),hi.data(),nullptr,p,.5,.9,.5,15,.9,20,kind!="MODE-DE",0,.1,.5);
        while(m.n<budget){
            auto a=Clock::now(); MatrixXd x=s.ask();auto b=Clock::now();
            MatrixXd y(2+con,p); for(int i=0;i<p;i++)problem(x.col(i).data(),d,y.col(i).data(),con);
            auto c=Clock::now();s.tell(y);auto e=Clock::now();
            m.n+=p; m.ask+=us(a,b);m.tell+=us(c,e);
        }
        total=us(t,Clock::now());m.quality=hv(s.population(),con);
    }
    std::cout<<kind<<","<<d<<","<<p<<","<<budget<<","<<seed<<","<<m.n<<","<<total/1000<<","<<m.ask/m.n<<","<<m.tell/m.n<<","<<m.quality<<"\n";
}
int main(){
    std::cout<<std::setprecision(12)<<"kind,d,p,budget,seed,calls,ms,ask_us_per_eval,tell_us_per_eval,quality\n";
    using Case=std::tuple<std::string,int,int,int>;
    std::vector<Case> cases={{"CMA",16,32,4096},{"CMA",64,32,4096},{"CMA",128,64,4096},
        {"Bite",16,1,8192},{"Bite",16,16,8192},{"Bite",96,1,8192},{"Bite",96,32,8192},
        {"DE",16,32,8192},{"DE",96,32,8192},{"DE",16,128,8192},{"MODE-SBX",16,32,4096},
        {"MODE-SBX",16,128,4096},{"MODE-SBX",16,256,4096},{"MODE-DE",16,256,4096},{"MODE-C",16,128,4096}};
    for(int seed : {99,101,102,103,104,105,106,107}){
        std::rotate(cases.begin(),cases.begin()+4,cases.end());for(auto& [k,d,p,b]:cases)run(k,d,p,b,seed);
    }
}
```

</details>

<details>
<summary>Replay file · direct.cpp</summary>

```cpp
#include "acmaesoptimizer.hpp"
#include "biteoptimizer.hpp"
#include "deoptimizer.hpp"
#include "gtop.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using Eigen::VectorXd;
static int calls;static bool astro;static double observed;
bool objective(int d,const double*x,double*out){
    double v=0;
    if(astro){
        std::vector<double> rp;v=cassini1(std::vector<double>(x,x+d),rp);
        if(!std::isfinite(v))v=1e10;
    }else for(int i=0;i<d;i++){double a=x[i]-.23;v+=a*a;}
    observed=std::min(observed,v);++calls;*out=v;return false;
}
void batch(int p,int d,double*x,double*y){for(int i=0;i<p;i++)objective(d,x+i*d,y+i);}
int main(){
    std::cout<<std::setprecision(12)<<"kind,problem,seed,calls,ms,observed,reported\n";
    for(int seed: {99,101,102,103,104,105,106,107}) for(int prob=0;prob<2;prob++){
        astro=prob;int d=astro?6:16,b=astro?512:4096;
        VectorXd lo=VectorXd::Constant(d,-5),hi=-lo;
        if(astro){lo<<-1000,30,100,30,400,1000;hi<<0,400,470,400,2000,6000;}
        VectorXd x0=(lo+hi)/2,sigma=VectorXd::Constant(d,.3);
        for(int j=0;j<3;j++){
            int k=(j+seed)%3;calls=0;observed=INFINITY;
            auto t=std::chrono::steady_clock::now();double y;
            if(k==0)y=acmaes::optimize_acma(0,objective,batch,d,x0.data(),lo.data(),hi.data(),sigma.data(),b,-INFINITY,-1,16,32,1,seed,true,true,-1,1).y;
            else if(k==1)y=biteopt::optimize_bite(0,objective,d,seed,x0.data(),lo.data(),hi.data(),b,-INFINITY,1,0,0).y;
            else y=differential_evolution::optimize_de(0,objective,d,seed,lo.data(),hi.data(),nullptr,nullptr,0,nullptr,b,200,-INFINITY,32,.5,.9,.1,.5,1).y;
            auto t1=std::chrono::steady_clock::now();
            std::cout<<(k==0?"CMA":k==1?"Bite":"DE")<<","<<(astro?"Cassini1":"Sphere")<<","<<seed<<","<<calls<<","<<std::chrono::duration<double,std::milli>(t1-t).count()<<","<<observed<<","<<y<<"\n";
        }
    }
}
```

</details>

<details>
<summary>Replay file · python_bench.py</summary>

```python
import csv, os, sys, tempfile, time, multiprocessing as mp, random, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds
from fcmaes import astro, bitecpp, cmaescpp, decpp, modecpp, diversifier as dv, mapelites as me
from loguru import logger
logger.remove()
warnings.filterwarnings("ignore", category=DeprecationWarning)
tick=time.perf_counter
def box(d,l,h): return Bounds(np.full(d,float(l)),np.full(d,float(h)))
def sphere(x):
    q=x-.23
    return float(q@q)
class QD:
    def __init__(self):
        self.alloc=mp.Value('i',0); self.counts=mp.RawArray('l',10); self.pid=None
    def __call__(self,x):
        if self.pid!=os.getpid():
            with self.alloc.get_lock():
                self.slot=self.alloc.value; self.alloc.value+=1
            self.pid=os.getpid()
        self.counts[self.slot]+=1
        return 1+sphere(x),x[:2]
def archive(seed,k=256,geo='grid',stats=False):
    a=me.Archive(12,box(2,-1,1),k,use_stats=stats)
    r=np.random.default_rng(seed)
    if geo=='grid': a.init_niches(0)
    else:
        centers=np.random.default_rng(42).random((k,2))
        me.set_KDTree(a,centers);a.cvt_clustering=True
    a.xs_view[:]=r.uniform(-1,1,a.xs_view.shape);a.argsort()
    return a
def qd(case,seed):
    a=archive(seed,4096 if '4096' in case else 256,'cvt' if 'CVT' in case else 'grid','stats' in case)
    f=QD(); bounds=box(12,-1,1); budget=2048
    w=2 if 'w2' in case else 1;t=tick()
    if case.startswith('MAP'):
        me.optimize_map_elites(f,bounds,box(2,-1,1),workers=w,archive=a,iterations=1,
            me_params={'chunk_size':32,'generations':budget//(32*w),'use_sbx':'Iso' not in case},
            cma_params={'cma_generations':0})
    else:
        sol=case.split('-')[1]; p={'solver':sol,'popsize':32,'max_evals':256,'stall_criterion':20}
        dv.minimize(f,bounds,box(2,-1,1),workers=w,archive=a,max_evals=budget,opt_params=p)
    sec=tick()-t;calls=sum(f.counts);ys,ds,xs=a.get_occupied_data()
    assert calls>=budget and np.allclose([1+sphere(x) for x in xs],ys) and len(ys)>0
    return (calls,sec*1000,len(ys)/a.capacity,a.get_qd_score(),float(ys.min()),a.get_occupied()-len(ys))
def scalar(case,seed):
    kind,obj=case.split('-'); s=np.random.default_rng(seed)
    if obj=='Cassini1':
        problem=astro.Cassini1();bounds=problem.bounds;f=problem.fun;budget=512
    else: bounds=box(16,-5,5); f=sphere;budget=4096
    seen=[];p=32
    def counted(x):
        y=f(x);seen.append(y);return y
    mod={'CMA':cmaescpp,'Bite':bitecpp,'DE':decpp}[kind]
    options={'bounds':bounds,'max_evaluations':budget,'rg':s}
    if kind!='Bite': options['popsize']=p
    if kind!='DE':options['x0']=(bounds.lb+bounds.ub)/2
    t=tick();res=mod.minimize(counted,**options);sec=tick()-t
    assert res.success and res.nfev==len(seen) and np.isfinite(res.fun)
    return (len(seen),1000*sec,min(seen),res.fun,f(res.x),res.status)
def kernel(case,seed):
    k=4096;cap=64;a=archive(seed,k,'cvt' if 'CVT' in case else 'grid','stats' in case)
    r=np.random.default_rng(seed);xs=r.uniform(-1,1,(80,cap,12));ys=1+np.sum((xs-.23)**2,axis=2)
    batches=[list(zip(yy,xx[:,:2])) for yy,xx in zip(ys,xs)]
    t=tick()
    if 'lookup' in case:
        for xx in xs: a.index_of_niches(xx[:,:2])
    elif 'sort' in case:
        a.set_ys(r.random(k));t=tick()
        for _ in xs:a.argsort()
    else:
        for xx,yd in zip(xs,batches): me.update_archive(a,xx,yds=yd)
    return (len(xs)*cap,1000*(tick()-t),a.get_occupied()/k,a.get_qd_score(),0,0)
if __name__=='__main__':
    tag=sys.argv[1]; out=sys.argv[2]
    # Warm grid before forking, including the non-contiguous slice signature.
    dummy=archive(9); dummy.index_of_niches(np.zeros((32,2)));dummy.index_of_niches(np.zeros((32,12))[:,:2])
    cases=['CMA-Sphere','Bite-Sphere','DE-Sphere','CMA-Cassini1','Bite-Cassini1','DE-Cassini1',
       'MAP-grid','MAP-grid4096','MAP-CVT','MAP-Iso-grid','MAP-grid-w2',
       'DIV-CMA_CPP-grid','DIV-DE_CPP-grid','DIV-BITE_CPP-grid','DIV-CMA_CPP-CVT','DIV-CMA_CPP-grid-w2',
       'kernel-grid-lookup','kernel-CVT-lookup','kernel-grid-update','kernel-grid-update-stats','kernel-sort']
    if tag=='jit': cases=['MAP-grid','DIV-CMA_CPP-grid','DIV-BITE_CPP-grid','kernel-grid-lookup','kernel-grid-update','kernel-grid-update-stats']
    with open(out,'w',newline='') as file:
        wr=csv.writer(file);wr.writerow(['case','rep','calls','ms','q1','q2','q3','diag'])
        for s in [99,101,102,103,104,105,106,107]:
            order=cases.copy();random.Random(s).shuffle(order)
            for case in order:
                row=(kernel if case.startswith('kernel') else qd if case.startswith(('MAP','DIV')) else scalar)(case,s)
                wr.writerow((case,s,*row));file.flush()
```

</details>

<details>
<summary>Replay file · state.py</summary>

```python
from python_bench import *
def zdt(x):
    g=1+9*np.mean(x[...,1:],axis=-1)
    return np.stack([x[...,0],g*(1-np.sqrt(x[...,0]/g))],axis=-1)
def hv(y):
    points=y[np.argsort(y[:,0])];last=6.;v=0.
    for a,b in points:
        if a<1.1 and b<last:v+=(1.1-a)*(last-b);last=b
    return v
with open(sys.argv[1],'w',newline='') as output:
    wr=csv.writer(output);wr.writerow(['case','rep','calls','ms','quality'])
    for seed in [99,101,102,103,104,105,106,107]:
        cases=['CMA-state','Bite-state','DE-state','MODE-state','MODE-minimize']
        random.Random(seed).shuffle(cases)
        for case in cases:
            rg=np.random.default_rng(seed);b=box(16,-5,5) if 'MODE' not in case else box(16,0,1)
            n=0; qual=np.inf;t=tick()
            if case=='MODE-minimize':
                counter=[0]
                def fun(x):counter[0]+=1;return zdt(x)
                x,y=modecpp.minimize(fun,2,0,b,popsize=256,max_evaluations=4096,rg=rg)
                ms=1000*(tick()-t);n=counter[0];qual=hv(y);assert np.allclose(zdt(x),y)
            else:
                if case=='MODE-state': es=modecpp.MODE_C(2,0,b,popsize=256,rg=rg)
                elif case=='CMA-state': es=cmaescpp.ACMA_C(16,b,x0=np.zeros(16),popsize=32,rg=rg)
                elif case=='DE-state': es=decpp.DE_C(16,b,popsize=32,rg=rg)
                else: es=bitecpp.Bite_C(16,b,x0=np.zeros(16),batch_size=16,max_evaluations=4096,rg=rg)
                while n<4096:
                    x=es.ask()
                    if case=='MODE-state':y=zdt(x)
                    else:
                        y=np.sum((x-.23)**2,axis=1);qual=min(qual,np.min(y))
                    n+=len(x);stop=es.tell(y);assert stop>=0
                    if stop:break
                ms=1000*(tick()-t)
                if case=='MODE-state':qual=hv(zdt(es.population()))
            wr.writerow([case,seed,n,ms,qual]);output.flush()
```

</details>

<details>
<summary>Replay file · qd_components.py</summary>

```python
from python_bench import *
import contextlib,io
def record(name,s,n,t,a=None):
    cov=score=np.nan
    if a:
        cov=len(a.get_occupied_data()[0])/a.capacity;score=a.get_qd_score()
    wr.writerow([name,s,n,1000*(tick()-t) if a is None else elapsed,cov,score]);f.flush()
with open(sys.argv[1],'w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['case','rep','calls','ms','coverage','score'])
    for s in [99,101,102,103,104,105,106,107]:
        for solver in ['CMA_CPP','DE_CPP','BITE_CPP']:
            a=archive(s); n=mp.RawValue('l',0); obj=QD();rg=np.random.default_rng(s);t=tick()
            params={'solver':solver,'popsize':32,'max_evals':256,'stall_criterion':20}
            while n.value<2048: dv.minimize_(a,obj,box(12,-1,1),rg,n,2048,params)
            elapsed=1000*(tick()-t);record('InProcess-'+solver,s,sum(obj.counts),t,a)
        a=archive(s);obj=QD();t=tick()
        me.run_map_elites_(a,obj,box(12,-1,1),np.random.default_rng(s),{'chunk_size':32,'generations':64},{'cma_generations':0})
        elapsed=1000*(tick()-t);record('InProcess-MAP',s,sum(obj.counts),t,a)
        with tempfile.TemporaryDirectory() as tmp:
            previous=os.getcwd(); os.chdir(tmp);me.rng=np.random.default_rng(s);np.random.seed(s)
            try:
                for label in ['cold','cached']:
                    a=me.Archive(12,box(2,-1,1),128)
                    t=tick()
                    with contextlib.redirect_stdout(io.StringIO()): a.init_niches(8)
                    record('CVT-'+label,s,128,t)
            finally: os.chdir(previous)
```

</details>

<details>
<summary>Replay file · arm_bench.py</summary>

```python
from python_bench import *
import contextlib,io
from importlib.util import spec_from_file_location,module_from_spec
source=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp);spec=spec_from_file_location('arm_example',source);arm=module_from_spec(spec);spec.loader.exec_module(arm)
    with out.open('w',newline='') as file:
        wr=csv.writer(file);wr.writerow(['case','rep','calls','ms','coverage','score'])
        for s in [99,101,102,103,104,105,106,107]:
            for case in ['MAP-SBX','DIV-CMA_CPP']:
                q=arm.fitness(20);a=me.Archive(20,q.qd_bounds,256);a.init_niches(0)
                r=np.random.default_rng(s);a.xs_view[:]=r.uniform(q.bounds.lb,q.bounds.ub,a.xs_view.shape);a.argsort()
                calls=[0];n=mp.RawValue('l',0)
                def fn(x):
                    calls[0]+=1
                    return q(x)
                with contextlib.redirect_stdout(io.StringIO()):
                    t=tick()
                    if case=='MAP-SBX':me.run_map_elites_(a,fn,q.bounds,r,{'generations':32,'chunk_size':32},{'cma_generations':0})
                    else:
                        while n.value<1024:dv.minimize_(a,fn,q.bounds,r,n,1024,{'solver':'CMA_CPP','popsize':32,'max_evals':256})
                    ms=1000*(tick()-t)
                    ys,ds,xs=a.get_occupied_data()
                    assert calls[0]==1024 and all(np.isclose(q(x)[0],y) and np.allclose(q(x)[1],d) for x,y,d in zip(xs,ys,ds))
                wr.writerow([case,s,calls[0],ms,len(ys)/256,a.get_qd_score()]);file.flush()
```

</details>

<details>
<summary>Replay file · audit.py</summary>

```python
from python_bench import *
import traceback
def attempt(label,fn):
    try:print(label,fn())
    except Exception as e:print(label,type(e).__name__,str(e))
b=box(12,-1,1); db=box(2,-1,1)
params={'workers':1,'niche_num':64,'samples_per_niche':0,'iterations':1,'me_params':{'generations':1,'chunk_size':8},'cma_params':{'cma_generations':0}}
attempt('MAP fresh',lambda:me.optimize_map_elites(QD(),b,db,**params))
a=archive(100,k=64);f=QD()
attempt('MAP CMA child',lambda:me.optimize_map_elites(f,b,db,archive=a,**dict(params,cma_params={'cma_generations':1,'maxiters':2,'best_n':4,'popsize':8})).get_occupied())
print('MAP child actual calls',sum(f.counts))
attempt('MAP CMA worker direct',lambda:me.optimize_cma_(a,QD(),b,np.random.default_rng(5),{'best_n':4,'maxiters':1,'popsize':8}))
for tag,x in [('MAP.empty',me.empty_archive(12,db,64,0,True)),('DIV fresh',dv.minimize(QD(),b,db,64,0,32,1,opt_params={'solver':'DE_CPP','popsize':8,'max_evals':32},use_stats=True))]:
    print('stats request',tag,'name',x.name,'stats',x.use_stats)
for seed in [101,102,103,104,105,106,107]:
    es=bitecpp.Bite_C(12,b,batch_size=16,rg=np.random.default_rng(seed))
    x=es.ask(); print('Bite first batch',seed,len(x),len(np.unique(x,axis=0)))
es=decpp.DE_C(12,b,popsize=8,rg=np.random.default_rng(101));x=es.ask()
es.tell(np.arange(8,dtype=float))
print('DE first result',es.result().fun,'expected 0 at row 0','returned nfev',es.result().nfev)
for seed in [101,102,103]:
    bounds=box(4,10,20); observed=[]
    def f(v):
        observed.append(v.copy());return sphere(v)
    res=cmaescpp.minimize(f,bounds,x0=np.full(4,15.),popsize=12,workers=2,max_evaluations=60,rg=np.random.default_rng(seed))
    xx=np.array(observed);print('CMA delayed',seed,'calls',len(xx),'reported',res.nfev,'outside',np.any((xx<10)|(xx>20),axis=1).sum(), 'y',res.fun,'reeval',sphere(res.x))
for seed in [101,102,103]:
    seen=[]
    def positive_sphere(x):
        seen.append(x.copy());return float(np.sum(x*x))
    cmaescpp.minimize(positive_sphere,Bounds(np.full(4,10.),np.full(4,20.)),x0=np.full(4,15.),workers=2,popsize=12,max_evaluations=12,rg=np.random.default_rng(seed))
    print('CMA initial callbacks',seed,seen[:2])
```

</details>

<details>
<summary>Raw observations · native.csv</summary>

```csv
kind,d,p,budget,seed,calls,ms,ask_us_per_eval,tell_us_per_eval,quality
Bite,16,16,8192,99,8192,3.501209,0.1966796875,0.201194335938,1.07207262739e-19
Bite,96,1,8192,99,8192,12.725382,0.590979492187,0.652686645508,4.12452300465e-25
Bite,96,32,8192,99,8192,14.674516,1.06076416016,0.501978149414,0.000409552832215
DE,16,32,8192,99,8192,1.728631,0.140413696289,0.0619918212891,6.12434601667e-13
DE,96,32,8192,99,8192,5.713229,0.501931030273,0.140110839844,117.279068529
DE,16,128,8192,99,8192,1.869543,0.16593359375,0.0537294921875,3.70956301248e-05
MODE-SBX,16,32,4096,99,4096,5.082229,0.0301567382812,1.19908959961,6.23564457937
MODE-SBX,16,128,4096,99,4096,9.92642,0.0275168457031,2.38072827148,5.93329604728
MODE-SBX,16,256,4096,99,4096,15.701772,0.0222478027344,3.79053076172,5.42427889798
MODE-DE,16,256,4096,99,4096,15.671436,0.332238769531,3.47370410156,5.66917781766
MODE-C,16,128,4096,99,4096,11.0492,0.0245124511719,2.65652001953,4.68507826513
CMA,16,32,4096,99,4096,4.878312,0.439691162109,0.728759033203,1.73804149629e-09
CMA,64,32,4096,99,4096,52.169606,2.04102758789,10.6348146973,0.000258751636671
CMA,128,64,4096,99,4096,149.145257,5.01701782227,31.109262207,0.247433662008
Bite,16,1,8192,99,8192,4.620734,0.264287597656,0.208193115234,2.77333911992e-30
DE,96,32,8192,101,8192,5.562603,0.483493041992,0.141044189453,65.3441294037
DE,16,128,8192,101,8192,1.869994,0.164958740234,0.0546121826172,2.96494557662e-05
MODE-SBX,16,32,4096,101,4096,5.133165,0.0320717773437,1.20943969727,6.23288942602
MODE-SBX,16,128,4096,101,4096,12.147553,0.0321552734375,2.91827954102,5.99421912603
MODE-SBX,16,256,4096,101,4096,15.56728,0.0225607910156,3.75780395508,5.4766043753
MODE-DE,16,256,4096,101,4096,16.205209,0.336761230469,3.59960449219,5.80350327244
MODE-C,16,128,4096,101,4096,10.759052,0.0336157226563,2.57756103516,4.71892325765
CMA,16,32,4096,101,4096,4.585732,0.439068115234,0.667800048828,5.72491851561e-10
CMA,64,32,4096,101,4096,49.749835,1.91626708984,10.1739445801,0.000213023466715
CMA,128,64,4096,101,4096,156.078001,5.14335742188,32.7340683594,0.2340019126
Bite,16,1,8192,101,8192,4.607144,0.244991088867,0.226547973633,2.77333911992e-30
Bite,16,16,8192,101,8192,12.39864,1.20045373535,0.288393920898,2.49858133182e-27
Bite,96,1,8192,101,8192,19.48784,0.832418457031,1.23801806641,1.16105570165e-21
Bite,96,32,8192,101,8192,19.032344,0.740514892578,1.16398681641,1.81481561951e-23
DE,16,32,8192,101,8192,1.742963,0.141336914062,0.0623337402344,3.30212689879e-12
MODE-SBX,16,256,4096,102,4096,17.678978,0.0275122070312,4.26829101563,5.43361712839
MODE-DE,16,256,4096,102,4096,17.165574,0.372800048828,3.79825732422,5.71742782054
MODE-C,16,128,4096,102,4096,11.817135,0.0291157226562,2.84093334961,4.66155701895
CMA,16,32,4096,102,4096,4.924842,0.455204101563,0.722835205078,3.04028308592e-10
CMA,64,32,4096,102,4096,51.306838,1.96190869141,10.5080895996,0.000197642333439
CMA,128,64,4096,102,4096,153.418509,5.04030029297,32.1831691895,0.251599760823
Bite,16,1,8192,102,8192,4.690599,0.257574829102,0.229335205078,2.77333911992e-30
Bite,16,16,8192,102,8192,3.168338,0.152859741211,0.210227661133,1.65909515397e-20
Bite,96,1,8192,102,8192,13.599478,0.664126831055,0.685976928711,4.36560528899e-19
Bite,96,32,8192,102,8192,10.852202,0.479577636719,0.62655847168,7.4282881581e-13
DE,16,32,8192,102,8192,1.825938,0.15391796875,0.0604871826172,3.93843782687e-13
DE,96,32,8192,102,8192,5.994242,0.498825927734,0.178497070312,112.95684366
DE,16,128,8192,102,8192,2.146099,0.195738891602,0.0574353027344,4.72516149746e-05
MODE-SBX,16,32,4096,102,4096,5.330233,0.0319377441406,1.2571887207,6.24241186867
MODE-SBX,16,128,4096,102,4096,10.234754,0.023673828125,2.45641381836,5.94472222915
CMA,64,32,4096,103,4096,52.866334,2.06704345703,10.781861084,0.000162274726197
CMA,128,64,4096,103,4096,164.529781,5.12715576172,34.782685791,0.222544246963
Bite,16,1,8192,103,8192,4.604119,0.264124023438,0.212596069336,2.77333911992e-30
Bite,16,16,8192,103,8192,3.050261,0.196145263672,0.152191162109,3.07453480746e-18
Bite,96,1,8192,103,8192,13.618487,0.645808959961,0.713546020508,2.90489199517e-19
Bite,96,32,8192,103,8192,11.347559,0.572564331055,0.590849975586,1.53770427889e-20
DE,16,32,8192,103,8192,1.791506,0.141329956055,0.0688472900391,1.19812235687e-12
DE,96,32,8192,103,8192,5.655232,0.480934204102,0.153588378906,59.2532597891
DE,16,128,8192,103,8192,1.878588,0.162942871094,0.0577919921875,1.9886144238e-05
MODE-SBX,16,32,4096,103,4096,5.538275,0.0437795410156,1.27739013672,6.24103542238
MODE-SBX,16,128,4096,103,4096,10.779163,0.0296342773438,2.58707446289,5.97005800892
MODE-SBX,16,256,4096,103,4096,17.337804,0.0220324707031,4.19157324219,5.3938824781
MODE-DE,16,256,4096,103,4096,16.642978,0.332700927734,3.70408813477,5.7621533752
MODE-C,16,128,4096,103,4096,11.841782,0.0269704589844,2.84441308594,4.75670206981
CMA,16,32,4096,103,4096,5.041056,0.463478271484,0.754822021484,9.50978419118e-10
Bite,96,1,8192,104,8192,13.699198,0.649109619141,0.702677734375,4.79201217455e-23
Bite,96,32,8192,104,8192,14.060791,0.762631225586,0.667073120117,4.04992412737e-22
DE,16,32,8192,104,8192,1.951938,0.163737670898,0.0658118896484,4.93113205094e-13
DE,96,32,8192,104,8192,6.126772,0.510058227539,0.17895703125,115.248812545
DE,16,128,8192,104,8192,2.009454,0.175308105469,0.0614584960937,3.62821975466e-05
MODE-SBX,16,32,4096,104,4096,5.23606,0.0289836425781,1.237359375,6.23628574698
MODE-SBX,16,128,4096,104,4096,11.330393,0.030169921875,2.72140332031,5.96508077419
MODE-SBX,16,256,4096,104,4096,17.503262,0.0231918945312,4.22914746094,5.53159010131
MODE-DE,16,256,4096,104,4096,16.648536,0.352478759766,3.69231152344,5.83531033692
MODE-C,16,128,4096,104,4096,11.798898,0.0254509277344,2.83883349609,4.71570195129
CMA,16,32,4096,104,4096,5.10912,0.477726318359,0.757412597656,4.53470379451e-10
CMA,64,32,4096,104,4096,53.563804,1.99072509766,11.0303300781,0.000265005247042
CMA,128,64,4096,104,4096,169.264392,5.11418139648,35.9310654297,0.207761593243
Bite,16,1,8192,104,8192,6.254083,0.448196777344,0.225026489258,2.77333911992e-30
Bite,16,16,8192,104,8192,7.260768,0.618877929688,0.24323828125,1.30225092638e-16
DE,16,128,8192,105,8192,1.995873,0.171565063477,0.0632000732422,6.14237950654e-05
MODE-SBX,16,32,4096,105,4096,6.68427,0.030017578125,1.589640625,6.22285985211
MODE-SBX,16,128,4096,105,4096,10.891932,0.0330795898437,2.61083129883,6.01920377374
MODE-SBX,16,256,4096,105,4096,17.898828,0.0227004394531,4.32413964844,5.49166829436
MODE-DE,16,256,4096,105,4096,17.199585,0.353190185547,3.82626855469,5.70533843014
MODE-C,16,128,4096,105,4096,12.190416,0.0431586914063,2.91782592773,4.60526840498
CMA,16,32,4096,105,4096,4.622788,0.446930908203,0.669339111328,1.85299900125e-10
CMA,64,32,4096,105,4096,52.079271,1.97181323242,10.6864978027,0.000337915520129
CMA,128,64,4096,105,4096,161.427673,5.20193139648,33.9517275391,0.335646178337
Bite,16,1,8192,105,8192,4.561386,0.245552246094,0.22404675293,2.77333911992e-30
Bite,16,16,8192,105,8192,3.170602,0.167090209961,0.195231079102,2.03580191195e-24
Bite,96,1,8192,105,8192,12.69797,0.600827270508,0.646581787109,3.71577318587e-23
Bite,96,32,8192,105,8192,10.355345,0.489218261719,0.560604248047,4.07840135615e-21
DE,16,32,8192,105,8192,1.799518,0.149115356445,0.0619619140625,9.19181234815e-15
DE,96,32,8192,105,8192,5.974642,0.53351550293,0.139005004883,67.3931857857
MODE-DE,16,256,4096,106,4096,29.688672,0.431858642578,6.79544750977,5.80130978433
MODE-C,16,128,4096,106,4096,16.523378,0.0487421875,3.94661181641,4.80348364939
CMA,16,32,4096,106,4096,6.543107,0.622453857422,0.915872802734,4.68710446739e-10
CMA,64,32,4096,106,4096,141.87774,6.40353588867,27.8835109863,0.000185204363864
CMA,128,64,4096,106,4096,168.204162,5.48439746094,34.0879443359,0.293868008574
Bite,16,1,8192,106,8192,6.408515,0.290073974609,0.275170532227,3.42784715222e-29
Bite,16,16,8192,106,8192,4.273422,0.202070678711,0.281666137695,3.92154523499e-21
Bite,96,1,8192,106,8192,17.766018,0.618666625977,1.2386282959,1.53881732915e-22
Bite,96,32,8192,106,8192,12.464459,0.602944946289,0.69234753418,1.98818365639e-17
DE,16,32,8192,106,8192,2.214202,0.180225463867,0.0652969970703,1.84951989242e-11
DE,96,32,8192,106,8192,6.43125,0.563698730469,0.155303466797,77.3388505057
DE,16,128,8192,106,8192,2.238189,0.207069213867,0.0569769287109,6.0931756099e-05
MODE-SBX,16,32,4096,106,4096,6.922519,0.0676162109375,1.59393310547,6.24427080582
MODE-SBX,16,128,4096,106,4096,11.18299,0.0350671386719,2.6794765625,6.04917997211
MODE-SBX,16,256,4096,106,4096,17.793585,0.0242431640625,4.29884887695,5.4692921415
CMA,128,64,4096,107,4096,177.171537,6.76496972656,36.1661921387,0.215430849497
Bite,16,1,8192,107,8192,5.383783,0.278582641602,0.273254394531,5.83017512765e-30
Bite,16,16,8192,107,8192,4.123727,0.248825073242,0.230053466797,1.72696946617e-20
Bite,96,1,8192,107,8192,14.439292,0.658394042969,0.775638183594,7.14219826719e-20
Bite,96,32,8192,107,8192,11.505086,0.514164428711,0.648074951172,1.31839186904e-17
DE,16,32,8192,107,8192,1.91353,0.150306884766,0.0739884033203,1.50993179094e-13
DE,96,32,8192,107,8192,7.170713,0.655389038086,0.157773071289,150.374161386
DE,16,128,8192,107,8192,2.343557,0.18715637207,0.0885738525391,5.86311730904e-05
MODE-SBX,16,32,4096,107,4096,5.28236,0.0317475585938,1.24527270508,6.24007593982
MODE-SBX,16,128,4096,107,4096,11.465757,0.0827709960938,2.68953076172,5.95188866482
MODE-SBX,16,256,4096,107,4096,17.056791,0.0224143066406,4.12007739258,5.37861757126
MODE-DE,16,256,4096,107,4096,16.623018,0.376679931641,3.66198339844,5.74590051036
MODE-C,16,128,4096,107,4096,11.488891,0.0270522460937,2.76264379883,4.76658014896
CMA,16,32,4096,107,4096,4.992884,0.468288574219,0.738770751953,1.29010680074e-09
CMA,64,32,4096,107,4096,53.352567,2.073453125,10.8958083496,0.000136305912969
```

</details>

<details>
<summary>Raw observations · direct.csv</summary>

```csv
kind,problem,seed,calls,ms,observed,reported
CMA,Sphere,99,4096,4.759417,9.23218087161e-10,9.23218087161e-10
Bite,Sphere,99,4096,3.532773,1.00379716453e-24,1.00379716453e-24
DE,Sphere,99,4106,0.972853,1.99074734738e-11,1.99074734738e-11
CMA,Cassini1,99,512,4.624895,17.3664252673,17.3664252673
Bite,Cassini1,99,512,4.051633,18.8862771706,18.8862771706
DE,Cassini1,99,513,3.872083,36.3924962363,36.3924962363
DE,Sphere,101,4140,0.756177,9.31886351737e-12,9.31886351737e-12
CMA,Sphere,101,4096,4.589922,6.25948204263e-10,6.25948204263e-10
Bite,Sphere,101,4096,2.055001,4.7232054538e-21,4.7232054538e-21
DE,Cassini1,101,537,4.187818,29.8647632637,29.8647632637
CMA,Cassini1,101,512,4.036,29.7889534783,29.7889534783
Bite,Cassini1,101,512,3.829749,26.5558631418,26.5558631418
CMA,Sphere,102,4096,4.749893,6.78500057559e-10,6.78500057559e-10
Bite,Sphere,102,4096,1.981269,3.43212900293e-24,3.43212900293e-24
DE,Sphere,102,4104,0.819703,1.90144869905e-09,1.90144869905e-09
CMA,Cassini1,102,512,4.190612,29.9916408674,29.9916408674
Bite,Cassini1,102,512,4.244323,18.4605826419,18.4605826419
DE,Cassini1,102,513,3.769519,16.1939449072,16.1939449072
Bite,Sphere,103,4096,1.923353,6.19707347185e-25,6.19707347185e-25
DE,Sphere,103,4100,0.901656,1.06509612458e-11,1.06509612458e-11
CMA,Sphere,103,4096,5.003274,5.93870451757e-10,5.93870451757e-10
Bite,Cassini1,103,512,4.144552,27.2207734733,27.2207734733
DE,Cassini1,103,538,3.909108,18.075698151,18.075698151
CMA,Cassini1,103,512,3.894277,27.0893600177,27.0893600177
DE,Sphere,104,4122,0.754204,2.96164528766e-10,2.96164528766e-10
CMA,Sphere,104,4096,4.505205,5.28552978358e-10,5.28552978358e-10
Bite,Sphere,104,4096,1.976242,9.55758049897e-25,9.55758049897e-25
DE,Cassini1,104,532,3.977091,18.855952582,18.855952582
CMA,Cassini1,104,512,3.884622,28.6157100331,28.6157100331
Bite,Cassini1,104,512,3.826054,26.3102115209,26.3102115209
CMA,Sphere,105,4096,4.512416,2.04913772019e-10,2.04913772019e-10
Bite,Sphere,105,4096,1.890543,1.7707622177e-20,1.7707622177e-20
DE,Sphere,105,4144,0.797439,1.87117654666e-11,1.87117654666e-11
CMA,Cassini1,105,512,3.949901,18.1741154109,18.1741154109
Bite,Cassini1,105,512,4.751175,19.9530270147,19.9530270147
DE,Cassini1,105,548,3.828087,11.6344717231,11.6344717231
Bite,Sphere,106,4096,1.943623,7.13677415017e-22,7.13677415017e-22
DE,Sphere,106,4147,0.77785,5.05680607898e-09,5.05680607898e-09
CMA,Sphere,106,4096,4.655902,3.45116376003e-10,3.45116376003e-10
Bite,Cassini1,106,512,4.007908,18.5009974953,18.5009974953
DE,Cassini1,106,531,3.833555,13.7529495508,13.7529495508
CMA,Cassini1,106,512,4.060948,45.2044018439,45.2044018439
DE,Sphere,107,4130,0.749176,9.42783613481e-11,9.42783613481e-11
CMA,Sphere,107,4096,4.84077,3.70609065138e-09,3.70609065138e-09
Bite,Sphere,107,4096,1.932346,1.36584111128e-22,1.36584111128e-22
DE,Cassini1,107,538,3.840035,17.1337268796,17.1337268796
CMA,Cassini1,107,512,3.949569,43.3941024538,43.3941024538
Bite,Cassini1,107,512,4.372876,12.967888216,12.967888216
```

</details>

<details>
<summary>Raw observations · state.csv</summary>

```csv
case,rep,calls,ms,quality
Bite-state,99,4096,8.403436997468816,5.590001124895296e-25
DE-state,99,4096,2.028211998549523,0.00011244252313771426
CMA-state,99,4096,14.909727000485873,8.111766432070993e-10
MODE-minimize,99,4096,102.3181470009149,5.443709925734993
MODE-state,99,4096,32.37267700023949,5.4878242496991385
CMA-state,101,4096,13.963402998342644,3.6908723638599e-09
MODE-state,101,4096,32.637995998811675,5.344841494761307
DE-state,101,4096,1.985838000109652,6.5317151556336184e-06
Bite-state,101,4096,7.9672499996377155,1.9421508998237027e-20
MODE-minimize,101,4096,97.28456200173241,5.338031403317166
MODE-minimize,102,4096,89.791406000586,5.3670326610001124
MODE-state,102,4096,37.490107999474276,5.389403339682816
CMA-state,102,4096,14.327231001516338,6.1517307279400555e-09
DE-state,102,4096,1.9086219981545582,2.93931532894685e-05
Bite-state,102,4096,7.792296997649828,1.7271487905588038e-18
MODE-minimize,103,4096,95.90734900120879,5.393680608349523
CMA-state,103,4096,14.076954001211561,1.797143690365657e-08
DE-state,103,4096,2.312680000613909,9.749398345019417e-05
Bite-state,103,4096,8.01272800163133,2.812487947794414e-22
MODE-state,103,4096,37.18605100220884,5.407528405777281
MODE-state,104,4096,49.54145399824483,5.431677658211827
MODE-minimize,104,4096,132.9404309981328,5.421875815668172
DE-state,104,4096,1.958386998012429,4.425110980425543e-05
Bite-state,104,4096,7.974099997227313,6.008823026612233e-18
CMA-state,104,4096,13.997395002661506,5.698951666551741e-09
Bite-state,105,4096,7.992967999598477,1.66225964520329e-14
CMA-state,105,4096,9.88361399868154,1.0439314763538225e-08
MODE-state,105,4096,37.552091998804826,5.430413354282999
DE-state,105,4096,2.0786280001630075,4.693244837353145e-06
MODE-minimize,105,4096,93.88218200183474,5.4072039574462
MODE-minimize,106,4096,94.80201599944849,5.352079304232613
Bite-state,106,4096,7.875500999944052,9.204398686885486e-20
DE-state,106,4096,2.2175370031618513,3.6613936493211512e-06
CMA-state,106,4096,14.027750003151596,2.0986688672747177e-09
MODE-state,106,4096,33.63238600286422,5.380324855466782
CMA-state,107,4096,13.97257700227783,2.8027757962969217e-09
DE-state,107,4096,1.9375960000616033,2.2516506275712476e-06
MODE-minimize,107,4096,93.19940100249369,5.366976167143202
MODE-state,107,4096,33.45206400263123,5.386134463006196
Bite-state,107,4096,8.468715997878462,4.941961555851536e-15
```

</details>

<details>
<summary>Raw observations · public.csv</summary>

```csv
case,rep,calls,ms,q1,q2,q3,diag
MAP-grid-w2,99,2048,84.97207300024456,0.94921875,101.2816731627695,1.3124891271389179,0
CMA-Sphere,99,4096,14.399471001524944,8.368466732930547e-10,8.368466732930547e-10,8.368466732930547e-10,0
DIV-CMA_CPP-CVT,99,2048,48.1161149982654,0.7890625,52.66098132439403,1.9288496235279151,0
kernel-CVT-lookup,99,5120,6.045720001566224,0.0,0,0,0
DIV-CMA_CPP-grid-w2,99,2080,35.522009999112925,0.87109375,67.81287796547615,1.7427804102469264,0
CMA-Cassini1,99,512,5.149479999090545,39.69539499039098,39.69539499039098,39.69539499039098,0
DIV-CMA_CPP-grid,99,2048,33.54285099703702,0.90234375,69.1477718199794,1.8491213747134385,0
DIV-BITE_CPP-grid,99,2048,64.47774900152581,0.98828125,89.37502603920225,1.0017384308272095,0
MAP-Iso-grid,99,2048,35.61236499808729,0.98828125,91.32426435107897,1.303281490663109,0
MAP-CVT,99,2048,47.5038910008152,0.875,103.88402942008425,1.1508242679494025,0
kernel-grid-update,99,5120,19.604596996941837,0.714111328125,602.1742506339372,0,0
Bite-Cassini1,99,512,5.399695000960492,17.948488281315225,17.948488281315225,17.948488281315225,0
Bite-Sphere,99,4096,18.45500800118316,4.5003574875327524e-23,4.5003574875327524e-23,4.5003574875327524e-23,0
kernel-grid-update-stats,99,5120,69.52690999969491,0.714111328125,602.1742506339372,0,0
DE-Sphere,99,4120,11.032959002477583,1.7162733310392e-11,1.7162733310392e-11,1.7162733310392e-11,0
kernel-grid-lookup,99,5120,5.898826002521673,0.0,0,0,0
MAP-grid4096,99,2048,73.31676200192305,0.223876953125,263.95606235874476,1.3704566733054904,0
DE-Cassini1,99,534,5.190408999624196,17.083723798409622,17.083723798409622,17.083723798409622,0
MAP-grid,99,2048,43.07059799975832,0.94921875,110.1779977695264,1.2149815415034901,0
kernel-sort,99,5120,25.017621999722905,0.0,38086.66484495521,0,0
DIV-DE_CPP-grid,99,2048,32.18150099928607,0.97265625,66.8315668868382,1.9214984050244182,0
DIV-CMA_CPP-grid-w2,101,2080,36.60969199700048,0.83203125,67.16762357479124,1.3329444043320449,0
DE-Cassini1,101,547,7.6253439983702265,18.45511401588966,18.45511401588966,18.45511401588966,0
kernel-sort,101,5120,25.37316700181691,0.0,33557.29230732408,0,0
kernel-grid-update-stats,101,5120,75.31630000084988,0.71484375,602.4097443249815,0,0
CMA-Sphere,101,4096,14.683813002193347,1.8288513995917718e-10,1.8288513995917718e-10,1.8288513995917718e-10,0
DIV-BITE_CPP-grid,101,2048,66.45616500100004,0.99609375,86.2834782753931,1.0010833918612179,0
DE-Sphere,101,4113,50.206961001094896,6.659268756964721e-11,6.659268756964721e-11,6.659268756964721e-11,0
kernel-grid-lookup,101,5120,6.252497001696611,0.0,0,0,0
MAP-grid4096,101,2048,111.29444800099009,0.235595703125,235.5979791455787,1.7329947126538088,0
Bite-Cassini1,101,512,5.368158002966084,24.935161480491708,24.935161480491708,24.935161480491708,0
DIV-DE_CPP-grid,101,2048,31.298441001126776,0.9609375,70.37408432469701,1.8568070453726322,0
MAP-Iso-grid,101,2048,34.95814899724792,0.99609375,86.03389427207631,1.7509751031817244,0
CMA-Cassini1,101,512,5.202506999921752,23.326986172502362,23.326986172502362,23.326986172502362,0
MAP-CVT,101,2048,45.79671500323457,0.875,98.31135879337556,1.281661768369291,0
MAP-grid-w2,101,2048,56.33593299717177,0.95703125,110.11237623413818,1.3119592790300076,0
Bite-Sphere,101,4096,12.89673499923083,3.5461509988586674e-21,3.5461509988586674e-21,3.5461509988586674e-21,0
DIV-CMA_CPP-CVT,101,2048,42.75875500024995,0.921875,76.20306485646611,1.633204925746535,0
DIV-CMA_CPP-grid,101,2048,36.446697002247674,0.8671875,69.30229465632857,1.5310506133428743,0
kernel-CVT-lookup,101,5120,5.686376000085147,0.0,0,0,0
MAP-grid,101,2048,42.838524997932836,0.97265625,102.2705267628336,1.3640287634233996,0
kernel-grid-update,101,5120,16.631995000352617,0.71484375,602.4097443249815,0,0
DIV-CMA_CPP-CVT,102,2048,44.025565999618266,0.6328125,54.71442018223989,1.541666921547582,0
MAP-grid-w2,102,2048,53.974097998434445,0.9375,101.18584780246476,1.3778340428456908,0
MAP-grid4096,102,2048,57.86597999758669,0.218505859375,245.55278868758884,1.6057473089261631,0
Bite-Sphere,102,4096,13.22736200017971,2.5915313331674646e-28,2.5915313331674646e-28,2.5915313331674646e-28,0
DIV-CMA_CPP-grid-w2,102,2080,31.753856997966068,0.828125,53.98414661725452,1.9143313347667263,0
CMA-Cassini1,102,512,5.321556996932486,47.11906018941279,47.11906018941279,47.11906018941279,0
kernel-grid-lookup,102,5120,6.0012789981556125,0.0,0,0,0
DE-Sphere,102,4152,11.158589000842767,2.645579016024798e-11,2.645579016024798e-11,2.645579016024798e-11,0
MAP-CVT,102,2048,45.27868600052898,0.8828125,104.58101814927815,1.1784981626689859,0
CMA-Sphere,102,4096,15.556305999780307,1.0172207720683995e-09,1.0172207720683995e-09,1.0172207720683995e-09,0
kernel-grid-update,102,5120,18.21714000107022,0.71875,605.8467826243896,0,0
DIV-BITE_CPP-grid,102,2048,98.20287600086886,0.98828125,83.80814442630944,1.0003879526203927,0
MAP-grid,102,2048,46.74080599943409,0.94921875,110.2114857723671,1.3185350612564832,0
MAP-Iso-grid,102,2048,37.8332340005727,0.99609375,95.7850823596012,1.344101501962632,0
kernel-sort,102,5120,63.01309899936314,0.0,32738.51730325132,0,0
DIV-DE_CPP-grid,102,2048,35.133921999658924,0.953125,64.76488298734301,1.7795483495790174,0
kernel-CVT-lookup,102,5120,5.927467998844804,0.0,0,0,0
DE-Cassini1,102,538,5.451952998555498,18.345320186890472,18.345320186890472,18.345320186890472,0
DIV-CMA_CPP-grid,102,2048,39.154821999545675,0.92578125,67.45356707569329,1.7027400634431853,0
kernel-grid-update-stats,102,5120,73.63217699821689,0.71875,605.8467826243896,0,0
Bite-Cassini1,102,512,5.188977000216255,17.500540034948607,17.500540034948607,17.500540034948607,0
kernel-grid-lookup,103,5120,7.421406000503339,0.0,0,0,0
MAP-CVT,103,2048,76.87059900126769,0.9140625,99.9005079683476,1.284223335195898,0
kernel-grid-update-stats,103,5120,74.3772869973327,0.715087890625,599.5243779967994,0,0
MAP-Iso-grid,103,2048,44.97112399985781,0.9765625,95.63779589674579,1.243642713269814,0
DIV-BITE_CPP-grid,103,2048,82.84803299829946,0.98828125,87.55226727707597,1.0006636372650553,0
CMA-Sphere,103,4096,15.679330997954821,3.666634742546504e-09,3.666634742546504e-09,3.666634742546504e-09,0
Bite-Cassini1,103,512,5.4047320008976385,6.934450357746738,6.934450357746738,6.934450357746738,0
kernel-CVT-lookup,103,5120,7.769127998471959,0.0,0,0,0
kernel-grid-update,103,5120,21.825429001182783,0.715087890625,599.5243779967994,0,0
DIV-CMA_CPP-grid,103,2048,57.5172900025791,0.578125,48.126264447417,1.7925661477347168,0
MAP-grid4096,103,2048,81.62208700014162,0.227783203125,244.4647229784034,1.7632543953184174,0
Bite-Sphere,103,4096,13.148823996743886,5.015821430022983e-22,5.015821430022983e-22,5.015821430022983e-22,0
MAP-grid-w2,103,2048,37.13864800010924,0.94921875,101.25977088930448,1.4032375293052888,0
DE-Cassini1,103,547,5.094345000543399,11.883582210763574,11.883582210763574,11.883582210763574,0
DIV-CMA_CPP-grid-w2,103,2080,31.971563999832142,0.95703125,68.93029959177207,1.9306003698555947,0
kernel-sort,103,5120,24.245540000265464,0.0,55010.977187636585,0,0
DIV-DE_CPP-grid,103,2048,45.79588500200771,0.95703125,62.972155404872915,2.3652894327245626,0
CMA-Cassini1,103,512,5.682589999196352,17.07806417287042,17.07806417287042,17.07806417287042,0
DE-Sphere,103,4118,11.210115997528192,1.0240685955244797e-09,1.0240685955244797e-09,1.0240685955244797e-09,0
MAP-grid,103,2048,44.06659199958085,0.96484375,111.83159180063333,1.2323805935856926,0
DIV-CMA_CPP-CVT,103,2048,43.120389000250725,0.8125,63.34987833672411,1.7899946186370204,0
kernel-grid-update,104,5120,20.86600600159727,0.712158203125,596.9795831627409,0,0
kernel-grid-update-stats,104,5120,76.10837299944251,0.712158203125,596.9795831627409,0,0
CMA-Cassini1,104,512,6.284675997449085,25.7499227160259,25.7499227160259,25.7499227160259,0
Bite-Cassini1,104,512,6.1402589999488555,16.720895691565904,16.720895691565904,16.720895691565904,0
MAP-CVT,104,2048,61.82547199932742,0.859375,100.01237173751359,1.2930167916473714,0
DE-Cassini1,104,519,5.971984999632696,25.33712938744882,25.33712938744882,25.33712938744882,0
DIV-CMA_CPP-CVT,104,2048,46.122224997816375,0.875,63.08563135687783,2.0618664023338837,0
MAP-Iso-grid,104,2048,37.905371998931514,0.96484375,94.03751871236466,1.5495415813272109,0
kernel-grid-lookup,104,5120,5.955570999503834,0.0,0,0,0
Bite-Sphere,104,4096,14.708390001032967,1.0066845142230497e-20,1.0066845142230497e-20,1.0066845142230497e-20,0
DIV-BITE_CPP-grid,104,2048,77.79080399996019,1.0,81.74769576227484,1.0009317699139453,0
DIV-CMA_CPP-grid-w2,104,2080,44.79548900053487,0.65625,44.65253466665057,2.1019126017345817,0
kernel-CVT-lookup,104,5120,6.691448001220124,0.0,0,0,0
DE-Sphere,104,4101,12.89087599798222,1.1142182896296938e-10,1.1142182896296938e-10,1.1142182896296938e-10,0
DIV-CMA_CPP-grid,104,2048,41.1513510007353,0.89453125,59.8788611524672,1.4432952944295443,0
kernel-sort,104,5120,26.456309999048244,0.0,41452.54958412384,0,0
MAP-grid-w2,104,2048,53.661969999666326,0.9375,101.1142202146124,1.2648250124465634,0
DIV-DE_CPP-grid,104,2048,32.40591699795914,0.97265625,64.48519597904183,2.1978926961011203,0
MAP-grid4096,104,2048,53.4211679987493,0.22900390625,244.37158807243813,1.5530227620447756,0
MAP-grid,104,2048,39.79728799822624,0.96875,109.17314079334089,1.3056682707722551,0
CMA-Sphere,104,4096,14.417532001971267,1.7809138898901926e-09,1.7809138898901926e-09,1.7809138898901926e-09,0
MAP-grid4096,105,2048,54.1993380029453,0.212646484375,216.11660324263585,1.6400272488614918,0
MAP-grid-w2,105,2048,51.22254900197731,0.9609375,95.43259730100849,1.3625694946785136,0
DIV-CMA_CPP-CVT,105,2048,37.865756999963196,0.92578125,67.91514695180176,1.642922416590665,0
kernel-CVT-lookup,105,5120,5.54228900000453,0.0,0,0,0
kernel-grid-lookup,105,5120,6.286538999120239,0.0,0,0,0
Bite-Sphere,105,4096,13.246600999991642,1.452442306322231e-22,1.452442306322231e-22,1.452442306322231e-22,0
MAP-grid,105,2048,38.41363599713077,0.91796875,116.74634376841163,1.2264523580087339,0
DIV-BITE_CPP-grid,105,2048,69.19208099861862,0.9921875,86.5353380863967,1.0000547901989627,0
kernel-grid-update-stats,105,5120,67.27783600217663,0.71337890625,602.2344521519472,0,0
DE-Sphere,105,4135,10.523885001020972,1.1260365984770523e-09,1.1260365984770523e-09,1.1260365984770523e-09,0
Bite-Cassini1,105,512,4.932349998853169,13.04447994102778,13.04447994102778,13.04447994102778,0
DE-Cassini1,105,522,4.748764000396477,28.62155891532782,28.62155891532782,28.62155891532782,0
DIV-CMA_CPP-grid-w2,105,2144,47.86978500123951,0.9765625,81.10121350955193,1.6819908179398921,0
DIV-DE_CPP-grid,105,2048,33.15321300033247,0.96875,66.02120754339664,1.8285629620640482,0
MAP-CVT,105,2048,39.865282000391744,0.91015625,105.0342586086337,1.3206171550082832,0
MAP-Iso-grid,105,2048,34.05496899722493,0.96484375,94.81978775301916,1.4482914363835535,0
CMA-Sphere,105,4096,14.150710998364957,8.926228837794556e-10,8.926228837794556e-10,8.926228837794556e-10,0
CMA-Cassini1,105,512,4.997559000912588,25.7750686624034,25.7750686624034,25.7750686624034,0
kernel-sort,105,5120,24.145700001099613,0.0,65873.93190772687,0,0
DIV-CMA_CPP-grid,105,2048,35.8128330008185,0.87890625,52.265464828625824,2.395688636547398,0
kernel-grid-update,105,5120,16.14274800158455,0.71337890625,602.2344521519472,0,0
kernel-grid-update-stats,106,5120,66.83244499799912,0.704345703125,591.0484113888691,0,0
MAP-Iso-grid,106,2048,34.82924299896695,0.98828125,100.35930670339563,1.4189666408099595,0
DIV-BITE_CPP-grid,106,2048,63.065563001146074,1.0,81.77030925990108,1.0000644014106104,0
DIV-CMA_CPP-grid,106,2048,35.09122199830017,0.79296875,51.664103758107174,2.0054309207157672,0
DE-Sphere,106,4110,10.614410999551183,2.860155779109262e-11,2.860155779109262e-11,2.860155779109262e-11,0
MAP-CVT,106,2048,46.209395997721,0.88671875,101.5559181206051,1.423070760705495,0
MAP-grid-w2,106,2048,46.312590999150416,0.90625,117.42175062413551,1.1732797628585965,0
MAP-grid,106,2048,37.71035899990238,0.93359375,105.01109297932564,1.2816868376131723,0
DE-Cassini1,106,547,5.132501999469241,14.27492603019407,14.27492603019407,14.27492603019407,0
MAP-grid4096,106,2048,52.78448899844079,0.22119140625,228.44772700344532,1.8168088588312985,0
CMA-Cassini1,106,512,5.2743260021088645,31.964666139733623,31.964666139733623,31.964666139733623,0
kernel-grid-update,106,5120,16.699725998478243,0.704345703125,591.0484113888691,0,0
DIV-CMA_CPP-CVT,106,2048,39.10745200118981,0.859375,58.244600016972655,1.7460424626158035,0
Bite-Cassini1,106,512,5.3354169976955745,22.704799697209356,22.704799697209356,22.704799697209356,0
DIV-DE_CPP-grid,106,2048,32.193469000048935,0.91015625,65.18834619469416,1.9653602826574406,0
Bite-Sphere,106,4096,12.526366001111455,5.8517860951235895e-21,5.8517860951235895e-21,5.8517860951235895e-21,0
kernel-sort,106,5120,24.438610998913646,0.0,48357.27755572533,0,0
kernel-grid-lookup,106,5120,6.012846999510657,0.0,0,0,0
kernel-CVT-lookup,106,5120,5.580155000643572,0.0,0,0,0
CMA-Sphere,106,3872,13.145507997251116,1.1249148823211791e-08,1.1249148823211791e-08,1.1249148823211791e-08,5
DIV-CMA_CPP-grid-w2,106,2048,47.83010499886586,0.80859375,63.9309244824569,1.6113782247953647,0
CMA-Cassini1,107,512,5.04873700265307,35.698620410267964,35.698620410267964,35.698620410267964,0
DIV-DE_CPP-grid,107,2048,32.77985000022454,0.96875,65.85296605676247,1.9894623443090471,0
DE-Cassini1,107,535,4.854263999732211,34.08237294842272,34.08237294842272,34.08237294842272,0
kernel-sort,107,5120,23.88464599789586,0.0,38112.358997211355,0,0
DIV-CMA_CPP-grid,107,2048,36.19520899883355,0.71484375,65.32268325865402,1.3151501598206403,0
DIV-CMA_CPP-CVT,107,2048,34.353648999967845,0.9296875,67.92893279059008,1.7595193626684185,0
kernel-CVT-lookup,107,5120,5.889681000553537,0.0,0,0,0
kernel-grid-lookup,107,5120,6.186417998833349,0.0,0,0,0
CMA-Sphere,107,4096,14.079715001571458,4.1223885206552584e-10,4.1223885206552584e-10,4.1223885206552584e-10,0
MAP-grid-w2,107,2048,33.17234099813504,0.96484375,100.89367354966504,1.3345901745860442,0
Bite-Cassini1,107,512,5.174406000151066,31.050439680454076,31.050439680454076,31.050439680454076,0
Bite-Sphere,107,4096,12.480166999011999,4.863284192894462e-21,4.863284192894462e-21,4.863284192894462e-21,0
MAP-CVT,107,2048,42.8686099985498,0.87890625,104.53320413660776,1.3634717940536674,0
MAP-grid,107,2048,39.14393599916366,0.9140625,99.60897224097833,1.3944881420477997,0
DE-Sphere,107,4147,10.64069000130985,8.845323209140448e-12,8.845323209140448e-12,8.845323209140448e-12,0
MAP-Iso-grid,107,2048,38.46059300121851,0.98828125,96.78300790446096,1.5132936635367775,0
DIV-CMA_CPP-grid-w2,107,2048,30.23926800233312,0.890625,72.47321821647567,1.69176829524401,-1
DIV-BITE_CPP-grid,107,2048,71.5616029992816,0.984375,92.25304253908271,1.0025008555731871,0
kernel-grid-update,107,5120,22.960514997976134,0.708740234375,591.6079226545213,0,0
kernel-grid-update-stats,107,5120,77.75045900052646,0.708740234375,591.6079226545213,0,0
MAP-grid4096,107,2048,58.70075399798225,0.2353515625,242.05577809202737,1.626036650634677,0
```

</details>

<details>
<summary>Raw observations · jit.csv</summary>

```csv
case,rep,calls,ms,q1,q2,q3,diag
kernel-grid-update,99,5120,10.26321400058805,0.714111328125,602.1742506339372,0,0
MAP-grid,99,2048,59.30645799890044,0.94140625,108.29493970471674,1.2990873998050865,0
DIV-BITE_CPP-grid,99,2048,60.13486600204487,0.98046875,91.36743906649306,1.00012831325348,0
DIV-CMA_CPP-grid,99,2048,34.76715500073624,0.8125,65.28350608109793,1.4681584243886998,0
kernel-grid-update-stats,99,5120,63.28500599920517,0.714111328125,602.1742506339372,0,0
kernel-grid-lookup,99,5120,0.1594499990460463,0.0,0,0,0
kernel-grid-lookup,101,5120,0.13090699940221384,0.0,0,0,0
MAP-grid,101,2048,43.27883199948701,0.95703125,105.18436892032413,1.3617367836066188,0
kernel-grid-update-stats,101,5120,63.86747199940146,0.71484375,602.4097443249815,0,0
DIV-BITE_CPP-grid,101,2048,58.471782998822164,0.99609375,92.5270787281035,1.000005018527346,0
DIV-CMA_CPP-grid,101,2048,32.173662999412045,0.84765625,69.78208380218132,1.5354231281282016,0
kernel-grid-update,101,5120,10.313138998753857,0.71484375,602.4097443249815,0,0
kernel-grid-update-stats,102,5120,61.532398001872934,0.71875,605.8467826243896,0,0
kernel-grid-lookup,102,5120,0.19041699852095917,0.0,0,0,0
MAP-grid,102,2048,38.19690499949502,0.98046875,111.28826126349986,1.1851982827497063,0
DIV-BITE_CPP-grid,102,2048,55.15795600149431,0.98046875,87.71516827405077,1.0002976174560265,0
kernel-grid-update,102,5120,10.307961001672084,0.71875,605.8467826243896,0,0
DIV-CMA_CPP-grid,102,2048,33.801202000177,0.85546875,68.85310150042278,1.5577991489582494,0
kernel-grid-update,103,5120,10.222551998595009,0.715087890625,599.5243779967994,0,0
MAP-grid,103,2048,38.46626100130379,0.92578125,102.4313300707341,1.3950941251443008,0
DIV-BITE_CPP-grid,103,2048,56.60426299800747,1.0,84.58970906655023,1.020132948869307,0
DIV-CMA_CPP-grid,103,2048,32.248315001197625,0.85546875,64.27797627932998,1.5667785182926037,0
kernel-grid-lookup,103,5120,0.1758740036166273,0.0,0,0,0
kernel-grid-update-stats,103,5120,72.06970900006127,0.715087890625,599.5243779967994,0,0
kernel-grid-update-stats,104,5120,65.19707199913682,0.712158203125,596.9795831627409,0,0
DIV-BITE_CPP-grid,104,2048,57.72557000091183,0.9921875,83.19018972625186,1.0011060265429184,0
kernel-grid-lookup,104,5120,0.1754140030243434,0.0,0,0,0
kernel-grid-update,104,5120,10.804166999150766,0.712158203125,596.9795831627409,0,0
DIV-CMA_CPP-grid,104,2048,33.34182999969926,0.95703125,64.15097149043682,1.8780978418102852,0
MAP-grid,104,2048,35.92356499939342,0.92578125,97.14416702442925,1.383826585646591,0
kernel-grid-update-stats,105,5120,63.38125000183936,0.71337890625,602.2344521519472,0,0
DIV-CMA_CPP-grid,105,2048,33.911217000422766,0.88671875,65.94226383993323,1.4647563117116182,0
kernel-grid-lookup,105,5120,0.13919999764766544,0.0,0,0,0
MAP-grid,105,2048,37.634849999449216,0.94921875,100.11501597364608,1.351312752753407,0
DIV-BITE_CPP-grid,105,2048,60.59099200138007,0.99609375,90.61377191102588,1.000571785119904,0
kernel-grid-update,105,5120,10.93478300026618,0.71337890625,602.2344521519472,0,0
kernel-grid-update,106,5120,10.4226129988092,0.704345703125,591.0484113888691,0,0
DIV-CMA_CPP-grid,106,2048,82.49273700130288,0.8046875,59.71479887862791,1.4033490261982948,0
DIV-BITE_CPP-grid,106,2048,63.96215499989921,0.99609375,87.22494851532971,1.0001451470529663,0
MAP-grid,106,2048,42.662855001253774,0.9140625,108.98855127889593,1.3325601853192957,0
kernel-grid-lookup,106,5120,0.34135299938498065,0.0,0,0,0
kernel-grid-update-stats,106,5120,68.82838700039429,0.704345703125,591.0484113888691,0,0
MAP-grid,107,2048,38.97823999795946,0.90234375,108.86625797949344,1.3168749288951505,0
DIV-BITE_CPP-grid,107,2048,59.755849000794115,0.9921875,89.13837158254144,1.000090186389006,0
kernel-grid-update-stats,107,5120,65.66642799953115,0.708740234375,591.6079226545213,0,0
kernel-grid-lookup,107,5120,0.12388600225676782,0.0,0,0,0
kernel-grid-update,107,5120,9.60554300036165,0.708740234375,591.6079226545213,0,0
DIV-CMA_CPP-grid,107,2048,34.738000998913776,0.703125,44.808573036333,2.224402364262855,0
```

</details>

<details>
<summary>Raw observations · components.csv</summary>

```csv
case,rep,calls,ms,coverage,score
InProcess-CMA_CPP,99,2048,14.976768998167245,0.828125,62.994963726726766
InProcess-DE_CPP,99,2048,14.104768997640349,0.9609375,62.60903712345702
InProcess-BITE_CPP,99,2048,45.81577900171396,0.99609375,86.35131114510723
InProcess-MAP,99,2048,30.080409000220243,0.859375,102.11323209140913
CVT-cold,99,128,14.413212000363274,nan,nan
CVT-cached,99,128,1.3880759979656432,nan,nan
InProcess-CMA_CPP,101,2048,15.731053001218243,0.9765625,82.50469612690156
InProcess-DE_CPP,101,2048,12.618314998690039,0.984375,61.45050743312302
InProcess-BITE_CPP,101,2048,47.50681099903886,0.9921875,84.37737259532364
InProcess-MAP,101,2048,21.26600199699169,0.9453125,112.70677986733713
CVT-cold,101,128,10.647181999956956,nan,nan
CVT-cached,101,128,0.5592799971054774,nan,nan
InProcess-CMA_CPP,102,2048,15.695879999839235,0.8671875,57.13986162362908
InProcess-DE_CPP,102,2048,14.706883001053939,0.92578125,59.73818800103163
InProcess-BITE_CPP,102,2048,46.59831599929021,0.9921875,83.07429866171667
InProcess-MAP,102,2048,20.63711699884152,0.890625,107.73739522751208
CVT-cold,102,128,11.00596200194559,nan,nan
CVT-cached,102,128,0.533702001121128,nan,nan
InProcess-CMA_CPP,103,2048,14.519521999318385,0.80859375,50.38527458431011
InProcess-DE_CPP,103,2048,12.91183599823853,0.94140625,67.96626222702604
InProcess-BITE_CPP,103,2048,205.05068700003903,0.98046875,81.90609203578836
InProcess-MAP,103,2048,37.90699299861444,0.8359375,104.57458517464244
CVT-cold,103,128,21.992232999764383,nan,nan
CVT-cached,103,128,1.3659529977303464,nan,nan
InProcess-CMA_CPP,104,2048,17.365708998113405,0.8984375,72.70147948615897
InProcess-DE_CPP,104,2048,13.940272001491394,0.98046875,75.42628604400417
InProcess-BITE_CPP,104,2048,47.7158710018557,0.99609375,89.50434178480533
InProcess-MAP,104,2048,21.168667000893038,0.890625,98.53340708696004
CVT-cold,104,128,12.966243000846589,nan,nan
CVT-cached,104,128,0.5782119987998158,nan,nan
InProcess-CMA_CPP,105,2048,18.76821899713832,0.90234375,74.25184226708154
InProcess-DE_CPP,105,2048,15.074099999765167,0.921875,72.60379649212571
InProcess-BITE_CPP,105,2048,47.92621500018868,0.99609375,91.97404728419875
InProcess-MAP,105,2048,23.36396499958937,0.9609375,105.75160242234473
CVT-cold,105,128,10.67098299972713,nan,nan
CVT-cached,105,128,0.5709800025215372,nan,nan
InProcess-CMA_CPP,106,2048,16.52952400036156,0.953125,78.5219530919349
InProcess-DE_CPP,106,2048,13.596133001556154,0.9921875,72.37790176178922
InProcess-BITE_CPP,106,2048,48.31403900243458,0.98828125,91.68090006574027
InProcess-MAP,106,2048,25.76020000196877,0.9140625,103.66856577827319
CVT-cold,106,128,10.20464099929086,nan,nan
CVT-cached,106,128,0.48635300117894076,nan,nan
InProcess-CMA_CPP,107,2048,18.653716000699205,0.75390625,52.87955942892006
InProcess-DE_CPP,107,2048,12.389564002660336,0.94921875,65.81623931407174
InProcess-BITE_CPP,107,2048,44.2467899993062,1.0,90.42755353729657
InProcess-MAP,107,2048,22.44417800102383,0.9375,100.41939111282146
CVT-cold,107,128,11.38526000067941,nan,nan
CVT-cached,107,128,0.4935529977956321,nan,nan
```

</details>

<details>
<summary>Raw observations · arm.csv</summary>

```csv
case,rep,calls,ms,coverage,score
MAP-SBX,99,1024,171.61341899918625,0.50390625,160.12390047063826
DIV-CMA_CPP,99,1024,165.02932799994596,0.2734375,70.39968938461098
MAP-SBX,101,1024,163.3115730001009,0.453125,139.6968122200053
DIV-CMA_CPP,101,1024,205.6590049978695,0.2109375,39.08618991581032
MAP-SBX,102,1024,164.3706260001636,0.46484375,146.6334948426954
DIV-CMA_CPP,102,1024,158.0344169997261,0.32421875,86.32918965073326
MAP-SBX,103,1024,158.62777000074857,0.47265625,147.54683489269803
DIV-CMA_CPP,103,1024,156.2724059986067,0.25390625,47.27510841312611
MAP-SBX,104,1024,165.8206579995749,0.4609375,134.40845880865015
DIV-CMA_CPP,104,1024,158.10289900036878,0.12890625,20.878784531865165
MAP-SBX,105,1024,162.9020370019134,0.41796875,132.9931445888508
DIV-CMA_CPP,105,1024,159.33403100279975,0.19921875,37.12740993745798
MAP-SBX,106,1024,179.68944099993678,0.46875,150.83859188726265
DIV-CMA_CPP,106,1024,266.396689999965,0.3203125,71.94880431607505
MAP-SBX,107,1024,168.68033799983095,0.45703125,148.56714898139984
DIV-CMA_CPP,107,1024,161.53232800206752,0.2578125,51.63539731717456
```

</details>
<!-- {% endraw %} -->
