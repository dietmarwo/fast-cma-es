# fast-cma-es remaining C++ performance campaign

27 November 2026 · Base commit `2cea79f9223f2153655bb8a56934cb9a5452c5d2` · fcmaes 2.2.0

The campaign ran **200 iterations** across the five native optimizers and five front/rank algorithms in the plan. It kept 4 correctness fixes, 34 implementation changes in the integrated build and 3 matrix objective entry points. 158 iterations stayed outside the patch and 1 measured an existing fast path. The ZIP includes the prototype diffs, timing pairs and correctness reproductions.

**Patch SHA-256** `679bcd9c071094049aaedfc15cd37130ffffe3ef765086a78eaa1f52f32e68ef`. Apply `changes.patch` at the pinned base commit. `reference-corrections.patch` reproduces the reference used to measure speed changes after correcting evaluation accounting and incumbents.

## Correctness fixes

| Case | Pinned base | Patch |
| --- | ---: | ---: |
| Dual Annealing local, 16D, budget 40 | 165 calls | 40 calls |
| Dual Annealing local, 128D, budget 260 | 1,285 calls | 260 calls |
| CR-FM-NES, 8D [2,3] boundary, 64 calls | Reported −21.876736 at a point with coordinate 3.245442 | Reported and replayed −23.419984 inside the box |
| CR-FM-NES and PGPE, population 32, budget 33 | 64 calls each | 33 calls each |

CR-FM-NES uses the raw minimum objective for the returned physical point. The penalty remains part of the distribution update. The two evolution strategies charge the final partial batch to the budget and consider each evaluated row for the incumbent. Dual Annealing now counts local finite differences and initial retries, reports nonzero iteration counts where appropriate and retains the best completed probe if the budget cuts a stencil short.

## Integrated performance

These are **30 paired observations** on AMD EPYC 9V74 Linux (GCC 14.2 release, Eigen 3.4.0, Python 3.12.14, nanobind 3.1.0). One logical CPU was used for the timed native and public paths, with one objective worker and BLAS/OpenMP threads set to one for Python. Seeds 1001–1030 were held out from the seven-pair candidate screens. The speedup is the median of per-pair reference/patch times. Brackets show a paired bootstrap 95% interval. The two elapsed columns are independent medians. Clock noise and short input sizes limit interpretations near 1×.

The baseline in this table includes the four correctness fixes. The six algorithms from the earlier campaign were excluded. N-D archive timing covers insertion, rank timing covers queries on an existing index, and IGD includes construction of any index. Python entry points are used for the 2D archive timings and for the matrix callback cases.

| Algorithm | Workload | Reference ms | Patch ms | Paired speedup |
| --- | --- | --- | --- | --- |
| CR-FM-NES | 128D state | 16.990 | 17.470 | 1.01× [0.96, 1.04] |
| CR-FM-NES | 512D state, P128 | 75.103 | 74.878 | 1.07× [0.98, 1.23] |
| CR-FM-NES | boundary/invalid inputs | 3.217 | 2.976 | 1.08× [1.08, 1.09] |
| CR-FM-NES | 128D direct | 16.538 | 16.074 | 1.03× [1.02, 1.05] |
| PGPE | 128D state | 8.095 | 6.176 | 1.28× [1.26, 1.40] |
| PGPE | 512D state P128 | 43.070 | 28.840 | 1.49× [1.46, 1.52] |
| PGPE | 16D direct | 1.266 | 0.999 | 1.26× [1.22, 1.28] |
| PGPE | 128D direct | 7.953 | 6.560 | 1.21× [1.10, 1.25] |
| Dual Annealing | 128D global | 21.673 | 18.605 | 1.19× [1.13, 1.51] |
| Dual Annealing | 128D local | 17.013 | 15.064 | 1.13× [1.12, 1.14] |
| Dual Annealing | 16D local Rastrigin | 2.215 | 2.063 | 1.06× [1.04, 1.07] |
| Dual Annealing | 16D short local | 0.036 | 0.030 | 1.10× [1.08, 1.24] |
| jSO | 128D, budget tail | 4.544 | 4.501 | 1.03× [1.00, 1.05] |
| jSO | 32D correlated P128 | 1.991 | 1.998 | 1.01× [0.98, 1.02] |
| jSO | 8D P128 | 2.120 | 2.063 | 1.03× [0.99, 1.05] |
| jSO | 128D Rastrigin, no archive | 14.209 | 14.520 | 0.98× [0.95, 1.00] |
| ArchiveDE | ZDT1 16D, tiered | 2.806 | 2.796 | 1.01× [1.00, 1.02] |
| ArchiveDE | 128D, Borda | 9.571 | 9.332 | 1.01× [0.96, 1.07] |
| ArchiveDE | 5 objectives, P256 Borda | 6.772 | 6.034 | 1.10× [1.03, 1.14] |
| ArchiveDE | 5 objectives, gate off | 7.057 | 6.910 | 1.00× [0.98, 1.02] |
| ParetoArchive2D | 2D trade-off, payload 32 | 1.601 | 1.565 | 1.01× [0.99, 1.09] |
| ParetoArchive2D | 2D uniform, no payload | 0.123 | 0.120 | 1.03× [1.02, 1.03] |
| ParetoArchiveND | 5D narrow sums, 512 | 1.294 | 0.196 | 6.48× [6.29, 6.67] |
| ParetoArchiveND | 5D narrow sums, 2048 | 21.484 | 0.586 | 36.63× [35.97, 37.13] |
| ParetoArchiveND | 5D narrow sums, 4096 | 87.890 | 1.187 | 73.67× [71.98, 75.99] |
| ParetoArchiveND | 5D uniform | 1.265 | 1.275 | 1.00× [0.99, 1.00] |
| ParetoArchiveND | 3D mixed | 68.096 | 69.221 | 0.98× [0.90, 1.04] |
| ParetoArchiveND | 8D uniform | 16.388 | 9.646 | 1.70× [1.69, 1.73] |
| ParetoArchiveND | 2D line | 0.482 | 0.489 | 0.98× [0.95, 1.03] |
| IGD | 1D large | 4.810 | 0.284 | 16.83× [16.54, 17.26] |
| IGD | 2D sorted | 0.245 | 0.236 | 1.00× [0.97, 1.10] |
| IGD | 3D random | 0.615 | 0.540 | 1.15× [1.13, 1.16] |
| IGD | 4D random | 9.401 | 0.913 | 10.04× [7.51, 11.83] |
| IGD | 8D random | 16.693 | 4.461 | 3.66× [2.95, 4.13] |
| IGD | 8D extreme | 90.678 | 1.637 | 56.29× [46.42, 59.37] |
| BordaBoxIndex | 4D narrow | 1.702 | 1.244 | 1.14× [0.99, 2.96] |
| BordaBoxIndex | 4D wide | 15.071 | 14.660 | 1.05× [1.02, 1.22] |
| BordaBoxIndex | 8D sparse, ties | 99.505 | 87.152 | 1.22× [0.99, 1.50] |
| GridBoxIndex | 4D wide | 44.412 | 32.981 | 1.33× [1.26, 1.51] |
| GridBoxIndex | 4D narrow | 0.427 | 0.397 | 1.10× [1.00, 1.19] |
| GridBoxIndex | 8D sparse, ties | 31.359 | 9.310 | 3.68× [2.58, 4.51] |
| GridBoxIndex | 4D sparse directory | 4.050 | 1.688 | 2.60× [2.27, 2.84] |
| GridBoxIndex | 8D wide sparse scan | 74.139 | 72.364 | 1.02× [0.94, 1.16] |
| CR-FM-NES | matrix objective 16D | 14.018 | 3.420 | 5.31× [3.01, 5.71] |
| CR-FM-NES | matrix objective 128D | 45.415 | 13.617 | 2.04× [1.94, 2.52] |
| CR-FM-NES | matrix objective 6D Cassini1 | 10.984 | 16.784 | 0.96× [0.60, 1.20] |
| PGPE | matrix objective 16D | 37.164 | 1.801 | 15.55× [7.77, 18.03] |
| PGPE | matrix objective 128D | 49.376 | 22.536 | 2.01× [1.84, 2.45] |
| PGPE | matrix objective 6D Cassini1 | 21.863 | 20.632 | 0.95× [0.90, 1.05] |
| jSO | matrix objective 16D | 14.161 | 2.546 | 5.66× [5.59, 5.94] |
| jSO | matrix objective 128D | 21.714 | 6.160 | 3.37× [3.09, 3.76] |
| jSO | matrix objective 6D Cassini1 | 13.910 | 9.288 | 1.30× [1.14, 1.94] |

For the matrix rows, both callbacks use the integrated build. The scalar and matrix functions evaluate the same objective. Cassini1 still evaluates each trajectory inside the callback.

### Cassini1 budget control

On the seven Cassini1 seeds from the plan, Dual Annealing with local search had a requested budget of 1,024. The pristine build used 1,183–5,096 evaluations and produced a median best value of 34.95. The patch used 1,024 evaluations per run and produced a median best value of 41.86. This changes the search effort, so the time comparison for Dual Annealing above uses the budget-corrected reference.

### ParetoArchiveND scaling

For points a and b with a dominating b, every coordinate difference bⱼ − aⱼ lies between zero and the difference of their mathematical objective sums. The implementation encloses each sum with outward-rounded bounds, indexes a coordinate when the enclosing band is sufficiently narrow, and tests strict dominance in both directions inside the candidate interval. It preserves accepted duplicates and the rounded objective-sum and insertion ordering of snapshots. A wide update migrates back to the general archive. A replay suite checks admission masks, duplicates and migration against exhaustive dominance.

### Native allocation samples

The allocation counter wraps allocation entry points referenced by the statically linked GCC driver and backend. The totals include fixture construction and warm-up. Requested bytes are cumulative traffic. Peak RSS is a separate process measurement and includes the runtime.

| Native path | Driver case | Allocation calls | Requested MB | Peak MiB |
| --- | --- | --- | --- | --- |
| CR-FM-NES | 128D, 4,096, flags 0 | 65,119 → 39,851 | 239.53 → 206.06 | 9.1 → 9.2 |
| PGPE | 128D, 4,096, flags 32 | 32,318 → 30,268 | 70.76 → 48.74 | 9.1 → 9.2 |
| Dual Annealing | 128D, 4,096, flags 64 | 88,851 → 80,659 | 61.78 → 53.39 | 9.1 → 9.2 |
| jSO | 128D, 4,093, flags 0 | 11,836 → 11,836 | 8.89 → 8.89 | 9.1 → 9.2 |
| ArchiveDE | 16D, 4,096, flags 176 | 53,196 → 43,192 | 8.43 → 7.87 | 9.2 → 9.2 |
| ParetoArchiveND | 5D, 4,096, flags 1 | 16,422 → 25,118 | 2.23 → 2.14 | 9.2 → 9.2 |
| IGD | 8D, 512, flags 2 | 6 → 10 | 0.20 → 0.39 | 9.2 → 9.2 |
| BordaBoxIndex | 8D, 65,537, flags 18 | 10,801 → 10,801 | 68.80 → 68.80 | 32.4 → 32.4 |
| GridBoxIndex | 4D, 16,384, flags 32 | 1,442 → 1,446 | 6.37 → 6.53 | 9.2 → 9.2 |

## Experiment ledger

Each performance iteration was compiled and measured against the same corrected reference. The seven-pair screen includes timings, work counts and deterministic traces for a seed outside the timed pairs. Timings shown here use the primary case, with details and companion results in the audit files. Where a 30-pair screening follow-up exists, its timing is used. Production changes that compose several experiments use the integrated evidence above. Scope substitutions and gates for individual prototypes are recorded in the last column.

C, P, D and J use dimension/evaluation budget/fixture flags. A uses decision dimension/evaluation budget/fixture flags. Archives H and N use objectives/input rows/fixture flags. IGD uses objectives/front rows/fixture flags. Borda and Grid use dimension/indexed rows/fixture flags. The driver comments and case maps document the flag combinations and output columns. Mask and front exports are timed through their public APIs when the changed code lives in the binding.

### CR-FM-NES

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| C01 | Correctness | Track the raw best evaluated row separately from the penalty-ranked adaptation winner. Store its clipped, decoded coordinates. | Callback count and replay | On the 64-call boundary fixture the old point exceeded 3.24. The final result replays to the best observed value, −23.419984 with normalization off and −23.704992 with normalization on. |
| C02 | Correctness | Give minimize a strict budget tail that evaluates the remaining candidates and updates the incumbent without applying a full-population distribution update. | Callback count and replay | At population 32, a requested budget of 33 now performs 33 callbacks. Verified budgets 1, 31, 32 and 33 for both normalization modes. |
| C03 | Kept in combination | Pass the noise matrix into sort_indices_by by const reference. | 0.96× [0.91, 1.03], 128/4096/0 | Removed one population copy inside the consolidated distribution update. |
| C04 | Kept in combination | Make w_dist_hat accept a column expression without constructing a dynamic matrix for every column. | 0.98×, 128/4096/0 | Compute norms directly on noise columns. The standalone reference-view change had no clear gain. |
| C05 | Kept in combination | Compute distance weights only when the path norm selects movement weights. | 1.04× [1.02, 1.06], 128/4096/0 | Compute distance weights only in the movement regime. |
| C06 | Kept in combination | Hoist alpha_dist(lambF) and eta_B(lambF) out of their population loops. | 0.99×, 128/4096/0 | Hoist the two generation-invariant coefficients in the consolidated path. |
| C07 | Reverted | Skip distance-weight exponentials for zero rank weights on the active branch. | 0.76× [0.62, 0.98], 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| C08 | Reverted | Generate Gaussian draws directly into the positive half of z, then mirror the columns. | 1.05×, 128/4096/0 | At least one companion case regressed in the screen. |
| C09 | Reverted | Store the projected row vbarᵀz once, apply the rank-one transform into a preallocated y buffer. | 0.92×, 128/4096/0 | At least one companion case regressed in the screen. |
| C10 | Reverted | Compute the transform only on the sampled half of the population and obtain the other half by sign symmetry. | 1.08×, 128/4096/0 | At least one companion case regressed in the screen. |
| C11 | Kept in combination | Fuse penalty accumulation with per-column physical decoding into scratch storage. | 1.97× [1.28, 2.01], 128/4096/0 | Reuse one penalty decoding vector. The isolated gain varied by fixture. |
| C12 | Reverted | Keep an index permutation for sorted x and perform the weighted center sum indirectly. | 0.97×, 128/4096/0 | At least one companion case regressed in the screen. |
| C13 | Reverted | Reuse a single column-gather buffer for y and z, eliminating duplicate full-matrix copies during permutation. | 1.04×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| C14 | Reverted | Size exY once per state and fill the final evolution-path column in place. | 1.04×, 128/4096/0 | At least one companion case regressed in the screen. |
| C15 | Reverted | Replace the allocated vbar broadcast matrix with coefficient-wise broadcast expressions. | 1.06×, 128/4096/0 | At least one companion case regressed in the screen. |
| C16 | Reverted | Build s_step1 in its destination by fusing the square and component product. | 1.05×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| C17 | Reverted | Tile the s/t updates over columns and immediately accumulate their contributions to v and D. | 1.48×, 128/4096/0 | Changed the update trajectory and regressed on a companion workload. |
| C18 | Reverted | Compute G_s from cached squared column norms instead of another dense squared-noise expression. | 0.99×, 128/4096/0 | Changed the update trajectory with negligible throughput gain. |
| C19 | Opt-in | Expose an opt-in matrix objective through the existing batch evaluator, carrying one physical row per candidate. | 5.20× [5.14, 5.29], 16/4096/0 | Public `vectorized=True` with one worker and owned physical inputs. |
| C20 | Reverted | Transfer ownership of a contiguous decoded batch to NumPy in the state binding to remove a layout copy. | 1.01×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |

### PGPE

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| P01 | Kept in combination | Generate normal values directly into scaled_noises, scaling each column in place. | 1.03× [0.93, 1.05], 128/4096/32 | Fill and scale reusable storage in random draw order within the shared sampler. |
| P02 | Kept in combination | Write center ± noise straight to adjacent output columns. Remove the intermediate x1/x2 populations. | 1.19× [1.12, 1.24], 128/4096/32 | Write mirrored proposals directly to adjacent columns. |
| P03 | Kept in combination | Decode and clip each antithetic pair into popX during generation on the state path. | 1.17× [1.16, 1.18], 128/4096/32 | Generate physical coordinates directly on the state entry point. |
| P04 | Reverted | Reuse one population workspace across successive direct-minimize generations. | 0.99×, 128/4096/96 | At least one companion case regressed in the screen. |
| P05 | Kept in combination | Give the direct batch evaluator a contiguous output buffer for evaluated coordinates that can become popX. | 1.13×, 128/4096/96 | Kept the ownership swap and reuse of the evaluated matrix. The callback still gets its own workspace. |
| P06 | Reverted | Represent the REINFORCE score baseline by a scalar. Broadcast only at use sites. | 1.04×, 128/4096/32 | Measured gain was limited or inconsistent across screening cases. |
| P07 | Reverted | Compute grad_center as a matrix–vector product on the pairwise score differences. | 1.00×, 128/4096/32 | The candidate changed a recorded result or trajectory. Its gain did not justify replacing the reference. |
| P08 | Reverted | Accumulate grad_stdev one column pair at a time without building total_sigma. | 1.05×, 128/4096/32 | The candidate changed a recorded result or trajectory. Its gain did not justify replacing the reference. |
| P09 | Reverted | Hoist reciprocal sigma and sigma squared out of the sample loop. | 0.96×, 128/4096/32 | Measured gain was limited or inconsistent across screening cases. |
| P10 | Reverted | Read paired fitness values through strided views and fill the differences in a reusable vector. | 1.05×, 128/4096/32 | Measured gain was limited or inconsistent across screening cases. |
| P11 | Reverted | Fuse the proposed sigma step and both allowed-change clamps into one coefficient loop. | 0.85×, 128/4096/32 | At least one companion case regressed in the screen. |
| P12 | Kept in combination | Compute ADAM bias correction and epsilon as scalars instead of allocating constant vectors. | 1.02× [0.98, 1.05], 128/4096/32 | Use scalar bias and epsilon terms. The combined build provides the timing evidence. |
| P13 | Reverted | Cache the step-size and bias scalars while the integer ADAM time index stays the same. | 1.00×, 128/4096/32 | Measured gain was limited or inconsistent across screening cases. |
| P14 | Reverted | Fuse the two ADAM moments and center update in one coordinate pass. | 0.98×, 128/4096/32 | The candidate changed a recorded result or trajectory. Its gain did not justify replacing the reference. Single coordinate ADAM pass with scalar bias terms. |
| P15 | Reverted | Reuse rank sort indices and precompute the fixed-population rank values. | 1.04×, 128/4096/32 | Measured gain was limited or inconsistent across screening cases. Compare the existing sort tie rule before integration. |
| P16 | Reverted | Use a read-only score view and omit the extra state-to-core copy on tell. | 0.98×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| P17 | Reverted | Scan once for the lowest raw score and copy only the winning incumbent row. | 1.02×, 128/4096/32 | Measured gain was limited or inconsistent across screening cases. |
| P18 | Opt-in | Add an opt-in vectorized objective to minimize using the existing batch trampoline. | 7.63× [7.58, 7.84], 16/4096/0 | Public `vectorized=True` with one worker and owned physical inputs. |
| P19 | Correctness | Implement an explicit final-prefix evaluation in minimize. For an incomplete pair/generation, update only the reported incumbent. | Callback count and replay | At population 32, a requested budget of 33 now performs 33 callbacks. Verified odd final counts and physical incumbent replay. |
| P20 | Reverted | As an opt-in search trial, decay center learning rate by a measured generation schedule on high-D problems. | 0.99× [0.96, 1.00], 128/8192/32 | Generation-index decay changed the trajectory. It gave no consistent holdout throughput gain. Use the direct generation-based time index and test decay 0.999. |

### Dual Annealing

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| D01 | Correctness | Enforce the remaining callback budget inside the local evaluator, with controlled unwinding and propagation of the best completed probe. | Callback count and replay | Local-search probes and initial retries are counted. The 40- and 260-call probes now use exactly 40 and 260, including when the budget interrupts a derivative. |
| D02 | Reverted | Offer a remaining-budget trigger that skips a local solve when even one complete gradient cannot fit. | 0.97× [0.96, 0.98], 128/260/65 | Changed search at tight budgets. The held-out nonseparable case slowed down. |
| D03 | Reverted | Cache the temperature-dependent factor in visit_fn once per strategy chain. | 1.02× [1.00, 1.03], 128/4096/0 | The isolated holdout regressed in the 128D local workload. |
| D04 | Reverted | Specialize the scalar visit path to generate the two normal draws and heavy-tailed variate without dynamic vectors. | 0.99×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| D05 | Reverted | Reuse x/y Gaussian work vectors for the all-coordinate move. | 1.01×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| D06 | Reverted | Fuse absolute value, log, exponent and denominator construction for each visiting coordinate. | 1.00×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| D07 | Kept in combination | Replace the extra a/b vectors in periodic wrapping with an in-place component loop. | 1.13× [1.12, 1.13], 128/4096/0 | Kept the existing two fmod reductions and removed the intermediate wrapping matrices. |
| D08 | Reverted | Try a bounded fast path for already interior visit coordinates, preserving the near-zero rule. | 1.00×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| D09 | Reverted | Modify one coordinate in a reusable candidate scratch vector for single-coordinate proposals. | 0.96×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| D10 | Reverted | Fuse normalized clipping and physical decoding into the callback argument workspace. | 0.95× [0.92, 0.97], 128/4096/0 | Global-only holdout regressed. |
| D11 | Kept in combination | Replace two full copies per finite-difference coordinate with one reusable stencil buffer, restoring each coordinate after its probes. | 0.99× [0.96, 1.01], 128/4096/64 | Use one temporary vector per gradient. Tests replay the order of the original finite-difference probes. |
| D12 | Reverted | Reuse unchanged center values in clipped one-sided stencils only when the caller opts into deterministic-objective semantics. | 1.03×, 128/4096/96 | The prototype changes evaluation order and had a small throughput effect on these inputs. |
| D13 | Reverted | Batch independent finite-difference probes behind an explicit matrix callback and a hard remaining-budget check. | 0.93×, 16/4096/0 | The batch stencil slowed the 16D cheap case and adds an API for which the application results were mixed. |
| D14 | Reverted | Persist normalized lower/upper arrays and initial-vector workspace across local solves. | 1.03×, 16/4096/66 | Measured gain was limited or inconsistent across screening cases. |
| D15 | Reverted | Reuse L-BFGS-B solver storage between polish calls, resetting all quasi-Newton history on every new solve. | 1.05×, 16/4096/66 | The small kernel gain does not justify a reentrant thread-local solver. Prototype reuses a thread-local solver with a nesting guard; integrated version would own scratch per run. |
| D16 | Reverted | Evaluate a cheaper one-sided gradient as an opt-in local scheme. | 0.87× [0.86, 0.88], 128/4096/64 | One-sided stencils changed the search and regressed on inexpensive objectives. |
| D17 | Reverted | Expose L-BFGS-B history size as a polish option and try m=3/6/12. | 1.02× [0.99, 1.04], 16/4096/67 | The history size changed the search. The holdout benefit varied by problem. Compare history size 3 with the stock size 6. |
| D18 | Reverted | Make the maximum local iterations depend on remaining calls and dimension. | 1.06×, 16/4096/67 | Measured gain was limited or inconsistent across screening cases. |
| D19 | Reverted | Trigger a polish only after a minimum relative global improvement or a measured stagnation interval, as an opt-in policy. | 1.05× [1.00, 1.12], 16/4096/66 | The trigger changed some evaluation sequences. The holdout improvement was workload-dependent. |
| D20 | Reverted | Cache the finite sequence of default chain temperatures for repeated short restarts. | 1.02×, 8/128/0 | Measured gain was limited or inconsistent across screening cases. |

### jSO

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| J01 | Reverted | Fill pending trial rows directly instead of copying the whole parent before crossing over. | 1.05×, 128/4093/0 | Measured gain was limited or inconsistent across screening cases. |
| J02 | Reverted | Reserve and reuse trial-row storage for failed offspring instead of freeing it every generation. | 1.02×, 128/4093/0 | Measured gain was limited or inconsistent across screening cases. |
| J03 | Reverted | Test a flat population slab with row indices for parents. | 1.02×, 128/4093/0 | Measured gain was limited or inconsistent across screening cases. Pack a parent slab per generation while retaining the owning rows as a compatibility view. |
| J04 | Reverted | Store displaced parents in contiguous archive slots with constant-time replacement. | 1.02×, 128/4093/0 | At least one companion case regressed in the screen. |
| J05 | Reverted | Cache dimensionwise bound widths for initialization. | 1.01×, 128/256/16 | Measured gain was limited or inconsistent across screening cases. |
| J06 | Kept in combination | Hoist pbest attraction coefficient and references to parent/pbest/donor rows outside the coordinate loop. | 1.04×, 128/4093/0 | Hoist mutation coefficients and row addresses before crossing over. |
| J07 | Reverted | Fast-path finite, in-bound mutant coordinates before midpoint repair. | 1.04×, 128/4093/0 | Measured gain was limited or inconsistent across screening cases. |
| J08 | Reverted | Separate the forced coordinate into prefix/forced/suffix loops while preserving uniform draws. | 1.04×, 128/4093/0 | Measured gain was limited or inconsistent across screening cases. |
| J09 | Kept in combination | Replace full stable ranking for pbest with top-k selection and deterministic ordering of the selected prefix. | 0.99× [0.98, 1.01], 32/8192/17 | Partially sort the p-best prefix with explicit index ties. The large-population screen was stronger than the holdout. |
| J10 | Reverted | Reuse the order scratch and exploit the already-ranked survivors immediately after population reduction. | 0.58×, 128/4093/0 | At least one companion case regressed in the screen. |
| J11 | Reverted | Reuse survivor scratch across shrink events, moving row ownership in sorted survivor order. | 0.99×, 128/4093/0 | Measured gain was limited or inconsistent across screening cases. |
| J12 | Reverted | Preallocate the successes buffer and reset its length each generation. | 0.96×, 128/4093/0 | Measured gain was limited or inconsistent across screening cases. |
| J13 | Reverted | Fold the maximum improvement scale into the success-collection pass. | 0.98×, 128/4093/0 | Measured gain was limited or inconsistent across screening cases. |
| J14 | Reverted | Fuse invalid-score sanitization and first-minimum discovery in accept. | 1.36×, 128/4093/0 | At least one companion case regressed in the screen. |
| J15 | Reverted | Retain current population and archive capacities when compacting after shrink. | 0.93×, 128/4093/0 | At least one companion case regressed in the screen. |
| J16 | Reverted | Add a native contiguous ask-output path for the binding, avoiding row-by-row intermediate object traversal. | 0.99×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| J17 | Reverted | Pass read-only contiguous score spans through the binding and allocate owned values only on initialization when required. | 0.99×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| J18 | Opt-in | Add opt-in batch objective evaluation to minimize. Define full-batch target accounting separately from serial prefix stopping. | 5.50× [5.31, 5.75], 16/4096/0 | Public `vectorized=True` with exact row accounting and full-batch target stops. |
| J19 | Reverted | Specialize tiny active populations near the end of the run with a small fixed sorting network and original-index tie keys. | 1.03×, 16/8192/0 | Measured gain was limited or inconsistent across screening cases. |
| J20 | Reverted | Try a slower linear population-reduction schedule as a named search option. | 1.00× [0.98, 1.01], 128/8192/0 | Slower population shrink changed the trace and gave no consistent benefit. |

### ArchiveDE

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| A01 | Reverted | Decode each unit-space trial directly into the final physical buffer. | 1.05×, 16/4096/0 | At least one companion case regressed in the screen. |
| A02 | Kept in combination | Precompute dimension widths and flags selecting the normal or extreme-width decode path. | 0.99× [0.95, 1.01], 16/4096/0 | Cache decision widths and preserve the overflow-safe interpolation path. |
| A03 | Reverted | Reuse capacity for pending unit rows across asks. | 1.06×, 16/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| A04 | Reverted | Use one row-major candidate buffer instead of a vector allocation per pending row. | 1.04×, 16/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| A05 | Reverted | Delay DE difference arithmetic until crossover selects that coordinate. | 0.81×, 16/4096/0 | At least one companion case regressed in the screen. |
| A06 | Reverted | Keep a per-batch candidate-index pool referencing parents and offspring. | 1.00×, 16/4096/0 | At least one companion case regressed in the screen. |
| A07 | Reverted | Reserve next-parent slots and copy only selected rows after selection commits. | 1.01×, 16/4096/0 | At least one companion case regressed in the screen. Reserve all next vectors before moving any parent or pending row; moves of vector<double> are noexcept. |
| A08 | Kept in combination | Add a Borda-only selection path that never materializes the unused coordinate-rank matrix. | 1.03× [1.00, 1.07], 128/4093/128 | Stable Borda selection skips coordinate-rank materialization. |
| A09 | Reverted | Replace tiered floating rank rows by contiguous 32-bit ranks with checked wide arithmetic for distances. | 1.05×, 16/4096/0 | The proposed 32-bit layout slowed the larger five-objective workload. 32-bit layout prototype. Candidate pool sizes above UINT32_MAX would need a wider route. |
| A10 | Kept in combination | Record objective champions during per-objective ranking, respecting the “unused champion” rule. | 0.94× [0.89, 0.99], 16/4096/0 | Reuse the stable objective orders to choose unused champions. |
| A11 | Reverted | Reuse stable-sort index and score buffers for every generation. | 1.02×, 16/4096/0 | Measured gain was limited or inconsistent across screening cases. The prototype uses thread-local work buffers to avoid modifying public state layout. |
| A12 | Kept in combination | Try stable integer-key sorting of Borda sums above a measured population cutoff. | 1.03× [1.01, 1.07], 16/8192/32 | Use stable radix sorting on integer Borda sums above 127 pool rows. |
| A13 | Reverted | Compact spread candidates with a live mask instead of erasing the chosen element and shifting the tail. | 0.96×, 16/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| A14 | Reverted | Specialize two-objective rank distance with exact checked arithmetic. | 1.00×, 16/4096/0 | At least one companion case regressed in the screen. |
| A15 | Reverted | Block the farthest-first nearest-distance update across candidate rows. | 1.02×, 16/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| A16 | Reverted | Cache the deterministic parent-only selection result keyed by exact ordered parents and policy. | 0.99×, 16/4096/0 | At least one companion case regressed in the screen. |
| A17 | Reverted | Add an internal validated-batch archive insertion path after the whole tell matrix has passed finiteness checks. | 1.00×, 16/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| A18 | Reverted | Remove one copy in the owned ask export while retaining the canonical pending data for repeated ask. | 1.01×, 128/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| A19 | Reverted | Reserve final population snapshot storage from rows, dimensions and objectives. | 1.01×, 16/4096/0 | At least one companion case regressed in the screen. |
| A20 | Reverted | Try a configurable spread-candidate fraction in the tiered selector. | 1.05× [0.92, 1.14], 16/8192/0 | Wider spread selection changed the front and slowed companion cases. Screen 75% instead of 60% of the Borda-sorted pool. |

### ParetoArchive2D

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| H01 | Existing control | Add an explicit rightmost append fast path for ascending-x, decreasing-y streams. | 1.02×, 2/16384/3 | Existing rightmost append code. Timed its removal as an ablation. Stock already has the rightmost append hint. This ablation measures its contribution. |
| H02 | Reverted | Add the analogous leftmost insertion hint for decreasing-x arrival. | 0.89×, 2/16384/4 | At least one companion case regressed in the screen. |
| H03 | Reverted | Broaden the existing gap hint to a bounded walk over adjacent successors. | 0.92×, 2/8192/33 | At least one companion case regressed in the screen. |
| H04 | Reverted | Keep a second dominance witness selected from a different region of the front. | 1.00×, 2/8192/33 | Measured gain was limited or inconsistent across screening cases. |
| H05 | Reverted | Use the last rejecting predecessor as a witness for subsequent nearby insertions. | 1.32× [1.23, 1.37], 2/8192/33 | Additional witness did not win consistently across the held-out streams. |
| H06 | Reverted | Precompute finite/reference/witness rejection masks for small blocks under a frozen archive prefix. Commit in original order. | 1.09×, 2/8192/33 | Measured gain was limited or inconsistent across screening cases. |
| H07 | Kept in combination | Hoist payload-shape validation out of the per-row internal batch insertion. | 1.01×, 2/32768/16 | Validate the batch once, and share the insert loop between mask consumers. |
| H08 | Reverted | Specialize the common zero-payload insertion path in the adapter. | 0.99×, 2/32768/16 | Measured gain was limited or inconsistent across screening cases. |
| H09 | Reverted | Pre-size the admission-mask buffer and write by row index. | 0.91×, 2/32768/16 | At least one companion case regressed in the screen. |
| H10 | Kept in combination | Fill the Python output mask directly to remove its intermediate native representation. | 1.01× [0.98, 1.13], 2/32768/16 | Write the Python bool admission mask directly into owned storage. |
| H11 | Reverted | Give the no-dominated-successor case a checked rectangle-only area-update path after normal map lookup. | 1.04×, 2/8192/33 | Measured gain was limited or inconsistent across screening cases. |
| H12 | Reverted | Specialize same-x replacement when it removes no successor. | 0.96×, 2/65536/37 | At least one companion case regressed in the screen. |
| H13 | Reverted | Try erasing successors individually using the returned iterator after the area delta has been validated. | 1.01×, 2/65536/2 | At least one companion case regressed in the screen. |
| H14 | Reverted | Stage payload materialization only once for same-key replacement, with a small fixed-width copy specialization. | 1.00×, 2/8192/37 | Measured gain was limited or inconsistent across screening cases. The primary fixture exercises repeated same-key replacements. Width four specialization is tested in the production-path gate if retained. |
| H15 | Reverted | Reuse an already allocated payload buffer on eligible same-key updates after all throwing validations. | 0.97×, 2/65536/37 | Measured gain was limited or inconsistent across screening cases. |
| H16 | Reverted | Use a cached witness block to reject repeated identical rows in add_batch. | 0.76×, 2/65536/6 | At least one companion case regressed in the screen. |
| H17 | Reverted | Add ordered-input detection that feeds successive map insertion hints for an ascending subrun. | 1.02×, 2/8192/35 | Measured gain was limited or inconsistent across screening cases. Substituted endpoint lookup hints because the existing call path already recognizes monotone right append; tests same-key and reverse streams. |
| H18 | Reverted | Size snapshot output once and write coordinates and payloads by offset. | 1.04×, 2/8192/33 | At least one companion case regressed in the screen. |
| H19 | Kept in combination | Fill owned NumPy snapshot arrays directly in a single map traversal. | 1.04× [1.01, 1.07], 2/8192/33 | Allocate owned arrays and write the snapshot in one map traversal. |
| H20 | Reverted | For the public wrapper only, prototype a small sorted-vector front with promotion to the map above a measured size. | 0.23×, 2/32768/16 | Small-vector prototype lost throughput. Incremental area and overflow semantics would also need more work. Cold small front prototype recomputes the final area from its current sorted vector; it is ineligible for release without online overflow checks and compensated incremental area. |

### ParetoArchiveND

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| N01 | Kept in combination | Specialize strict dominance for three objectives. | 1.17× [1.15, 1.18], 3/4096/1 | Specialize strict comparisons for three objectives. |
| N02 | Reverted | Try an unrolled five-objective dominance test with early rejection. | 0.91×, 5/4096/1 | At least one companion case regressed in the screen. |
| N03 | Kept in combination | Compare vectorized all-less-or-equal/any-strict tests to scalar early exit for 8–32 objectives. | 1.81× [1.79, 1.83], 8/4096/1 | Use a branch-free comparison loop for eight or more objectives. |
| N04 | Reverted | Probe likely discriminating objective axes first, keeping stored objective order intact. | 1.04×, 5/4096/1 | Measured gain was limited or inconsistent across screening cases. Try the smallest comparator coordinate first. Recomputing its index is charged to the trial. |
| N05 | Reverted | Introduce a small cache of recent valid dominance witnesses. | 1.05×, 5/4096/1 | At least one companion case regressed in the screen. |
| N06 | Reverted | Add a strict min/max bounding box per block of entries to skip impossible dominators. | 0.97×, 5/4096/1 | Computing temporary lower bounds cost more than the exact comparisons saved. Compute a one-axis lower bound on each block on demand to test pruning before adding cached-index maintenance. |
| N07 | Reverted | Use a separate bounding-box rejection for the suffix scan that removes entries dominated by a newcomer. | 0.93×, 5/4096/1 | On-demand upper bounds plus compaction lost time. On-demand upper bound over 32 rows, followed by a single compaction. |
| N08 | Reverted | Store objectives in padded fixed-size inline arrays for common widths 2–5. | 1.02×, 5/4096/1 | At least one companion case regressed in the screen. |
| N09 | Reverted | Put decision payloads in reusable slots referenced by entries. | 1.03×, 5/4096/33 | Measured gain was limited or inconsistent across screening cases. Run-local benchmark of reusable allocator slots. The prototype uses a thread-local resource and does not establish a cross-thread lifetime contract. |
| N10 | Reverted | Keep a compact vector of sorted entry handles, moving handles instead of full entries on insertion/erase. | 0.96×, 5/4096/1 | At least one companion case regressed in the screen. |
| N11 | Reverted | Fuse survivor marking with compaction so insert-before-remove does not shift a long suffix twice. | 1.01×, 5/4096/1 | Measured gain was limited or inconsistent across screening cases. |
| N12 | Reverted | Test blocked sorted storage with local splits under many nondominated insertions. | 0.93×, 5/4096/1 | At least one companion case regressed in the screen. Substituted the standard library's segmented deque for a custom splitting container so the storage trade-off can be measured first. |
| N13 | Reverted | Add a 3D skyline dominance index above a measured threshold while retaining duplicate multiplicities and arrival ordering. | 0.90×, 3/4096/1 | The first-coordinate 3D index still visited large ranges. 3D index on one objective; scan opposite sides for dominance and removal. |
| N14 | Reverted | Prototype an ND-tree with per-node bounds as a side index for large fronts. | 0.76×, 5/4096/1 | The bounded bucket prototype slowed general/front-heavy cases. Substituted a bounded tree directory with fixed initial buckets. Historical minima stay conservative after deletion. |
| N15 | Reverted | For nobj=2, accelerate exact-line migration using a sorted temporary set of references and hinted map insertion. | 0.99×, 2/8192/2 | Measured gain was limited or inconsistent across screening cases. The 2D mixed-front companion includes migration from its initial exact-line prefix. |
| N16 | Reverted | Coalesce identical 2D skyline keys into groups with per-arrival payloads. | 1.01×, 2/8192/6 | At least one companion case regressed in the screen. |
| N17 | Reverted | Cache sorted snapshot handles with invalidation on any accepted or erasing insertion. | 1.03×, 2/8192/2 | The first snapshot had little gain and the cached handles need lifetime restrictions. Only the first snapshot is timed by this screen; caching is useful only if reads repeat. |
| N18 | Kept in combination | Add exact, overflow-safe certificates for an equal-weight hyperplane in 3–5D before admitting a row without a scan. | 74.29× [65.51, 75.76], 5/4096/1 | Index the widest coordinate while the outward-rounded objective sums stay close. Revert to the general representation on wide updates. An outward-rounded sum interval supplies an exact first-coordinate range for possible comparisons in both dominance directions. |
| N19 | Reverted | Vectorize input validation across blocks while committing the valid prefix in order. | 1.03×, 5/4096/16 | At least one companion case regressed in the screen. |
| N20 | Reverted | Try size-based dispatch between a compact scan and the best validated index, with hysteresis around the crossover. | 38.00×, 5/4096/1 | A later index threshold gave less improvement at 4,096 points. Higher dispatch threshold with periodic reevaluation of the coordinate-range path. |

### IGD

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| I01 | Kept in combination | Add a sorted scalar search for d=1, seeded by lower_bound. | 25.90× [17.21, 32.91], 1/4096/0 | Use exact sorted search in one dimension for fronts with at least 32 rows. |
| I02 | Reverted | Choose the existing 2D scan/sorted crossover using both front and reference counts. | 0.98×, 2/64/17 | At least one companion case regressed in the screen. |
| I03 | Reverted | Detect already x-sorted input before copying/sorting the 2D front. | 0.94×, 2/8192/1 | At least one companion case regressed in the screen. |
| I04 | Reverted | Initialize the 2D best distance from the pivot neighbor before the leftward scan. | 0.98×, 2/8192/1 | Measured gain was limited or inconsistent across screening cases. |
| I05 | Reverted | Process references in x order using a moving pivot, restoring per-row results to original reduction order. | 1.04×, 2/8192/1 | At least one companion case regressed in the screen. |
| I06 | Reverted | Seed each coherent 2D query with the previous query's nearest front point as a valid upper bound. | 0.98×, 2/8192/1 | At least one companion case regressed in the screen. |
| I07 | Kept in combination | Sweep 3D kd-tree leaf capacities 8/16/32/64. | 1.00×, 3/512/0 | Use a leaf capacity of 64 for 3D. Tested 8, 16, 32 and 64 on the same paired fixtures. Use leaf capacity 16. |
| I08 | Reverted | Split 3D nodes on greatest coordinate spread in place of a rotating axis. | 0.67×, 3/4096/0 | At least one companion case regressed in the screen. |
| I09 | Reverted | Store subtree boxes for tighter pruning in the native 3D tree. | 0.81×, 3/4096/0 | At least one companion case regressed in the screen. |
| I10 | Reverted | Try an iterative 3D nearest traversal with a bounded local stack and safe spill. | 0.88×, 3/512/0 | At least one companion case regressed in the screen. |
| I11 | Reverted | Group 3D leaf coordinates by axis for vectorized squared-distance evaluation. | 0.96×, 3/4096/0 | Measured gain was limited or inconsistent across screening cases. |
| I12 | Reverted | Use dimension-specialized brute-force kernels at 4 and 8 objectives. | 0.81×, 8/512/0 | At least one companion case regressed in the screen. |
| I13 | Reverted | Tile reference/front matrices to keep front coordinate blocks hot in cache. | 0.55×, 8/4096/0 | At least one companion case regressed in the screen. |
| I14 | Reverted | Cut off a scalar high-dimensional distance once the partial squared sum reaches the current best. | 0.27×, 8/512/0 | Branching for early exit generally lost to uninterrupted arithmetic in the tested scan. |
| I15 | Kept in combination | Try a 4–8D kd-tree with data-driven fallback when query pruning is ineffective. | 8.32× [7.39, 15.59], 4/4096/0 | Use indexed search in 4D and 8D at measured sizes. Use a sampled visit count to fall back to a fixed-dimensional scan. |
| I16 | Reverted | Add blocked SIMD finiteness/magnitude checks before dispatch. | 0.92×, 8/32/16 | At least one companion case regressed in the screen. |
| I17 | Kept in combination | Prototype a power-of-two rescaling path for uniform extreme magnitudes. Retain robust fallback for mixed scales. | 63.45× [61.21, 64.09], 8/512/2 | Use power-of-two scaling for a narrow exponent range and keep the robust path for mixed magnitudes and the overflow edge. |
| I18 | Reverted | Expose an immutable prepared-front option for repeated IGD calls against different references. | 16.50×, 3/4096/80 | The local prepared-front prototype sped repeated queries. It needs a public ownership and validation design before release. Prototype compares one build and ten query sets against rebuilding ten times. Extreme-value validation stays outside this experiment. |
| I19 | Reverted | Parallelize independent reference queries above a size threshold, then reduce in original row order. | 1.33×, 8/4096/0 | Parallel trial helped its large workload. Its scheduling cost and interaction with parallel retries were outside the throughput target for this patch. |
| I20 | Reverted | Seed search using the nearest coarse cell representative on spatially clustered high-D fronts. Finish with exact pruning/scan. | 0.21×, 8/4096/0 | At least one companion case regressed in the screen. |

### BordaBoxIndex

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| B01 | Reverted | Split the IDs-only single-query kernel from the stats-producing batch kernel to omit the unused score-count pass. | 0.87×, 4/16384/1 | At least one companion case regressed in the screen. |
| B02 | Reverted | Fuse score-candidate counting into the same block loop as cross-axis filtering. | 1.02×, 4/16384/0 | Measured gain was limited or inconsistent across screening cases. |
| B03 | Reverted | Use the rank-sum filter to seed the hit mask, allowing exact rejection before reading cross-ranks. | 0.96×, 4/16384/0 | Measured gain was limited or inconsistent across screening cases. |
| B04 | Kept in combination | Skip remaining rank columns for blocks whose hit mask is already empty. | 1.51× [1.46, 1.53], 8/65537/18 | Skip later axes when a rank block has no matches. |
| B05 | Reverted | Check active axes in order of increasing rank-span width. | 0.98×, 4/16384/0 | Measured gain was limited or inconsistent across screening cases. |
| B06 | Reverted | Replace byte hit flags by a 64-bit bitset and enumerate set bits in ascending position. | 0.88×, 4/16384/0 | At least one companion case regressed in the screen. |
| B07 | Reverted | Tune scan block widths 16/32/128 according to rank width and dimension. | 2.27×, 4/16384/0 | At least one companion case regressed in the screen. Scan blocks of 32 rows. |
| B08 | Reverted | Specialize one- and two-active-axis kernels. | 0.53×, 3/16384/1 | At least one companion case regressed in the screen. |
| B09 | Reverted | Reserve the output vector adaptively from observed intersection widths. | 0.99×, 4/16384/0 | At least one companion case regressed in the screen. |
| B10 | Kept in combination | Append IDs to the batch destination through a sink instead of building one vector per query. | 1.00× [1.00, 1.02], 4/16384/0 | Append matching IDs directly into the batch result. |
| B11 | Reverted | Reuse spans and active-axis scratch per batch caller. | 0.98×, 4/16384/0 | Measured gain was limited or inconsistent across screening cases. |
| B12 | Reverted | Try branchless lower/upper bound searching over large sorted coordinate arrays. | 0.97×, 4/16384/0 | Measured gain was limited or inconsistent across screening cases. |
| B13 | Reverted | Store cross-ranks in contiguous per-axis slabs to remove per-column allocation and pointer chasing. | 0.71×, 8/65537/18 | At least one companion case regressed in the screen. |
| B14 | Reverted | Switch to on-demand cross-rank lookup at high dimensions to reduce quadratic storage. | 0.74×, 8/65537/18 | At least one companion case regressed in the screen. |
| B15 | Reverted | Derive 32-bit score storage when the maximum rank sum provably fits. Keep wide query arithmetic. | 0.97×, 4/16384/0 | At least one companion case regressed in the screen. |
| B16 | Reverted | Build axis rank orders independently in parallel with per-worker radix scratch. | 1.04×, 8/65537/18 | At least one companion case regressed in the screen. |
| B17 | Reverted | Change radix digit width for the shared stable rank constructor. | 1.22×, 8/65537/18 | At least one companion case regressed in the screen. |
| B18 | Reverted | Build cross-rank slabs in cache-sized row tiles. | 1.03× [0.99, 1.07], 8/65537/18 | Measured gain was limited or inconsistent across screening cases. |
| B19 | Reverted | Reuse previously found rank spans for repeated coordinate endpoints within a batch. | 0.77×, 4/16384/3 | At least one companion case regressed in the screen. |
| B20 | Reverted | Process independent queries in parallel with private buffers and ordered concatenation. | 1.05×, 4/16384/0 | Parallel throughput was sensitive to selectivity and startup. |

### GridBoxIndex

| ID | Decision | Experiment | Speedup and case | Outcome |
| --- | --- | --- | --- | --- |
| G01 | Reverted | Reuse per-axis query scratch across a batch. | 0.98×, 4/16384/1 | Measured gain was limited or inconsistent across screening cases. |
| G02 | Kept in combination | Emit IDs directly into the binding's batch destination. | 0.98×, 4/16384/1 | Use the same direct result buffer as the Borda binding. |
| G03 | Reverted | Reserve output capacity using a bounded estimate from rank spans. | 1.03× [1.03, 1.04], 4/16384/1 | Measured gain was limited or inconsistent across screening cases. |
| G04 | Kept in combination | Specialize the partial-cell inner test for d=2 and d=4. | 2.03× [2.02, 2.06], 4/16384/1 | Specialize the four-dimensional boundary test. The two-dimensional specialization stayed outside the patch. |
| G05 | Reverted | Test the narrowest constrained axis first in partial cells. | 0.86×, 4/16384/3 | At least one companion case regressed in the screen. |
| G06 | Reverted | Store cell ranks in 16 bits when N ≤ 65,536. | 0.52×, 4/16384/1 | Compact ranks helped some large inputs but slowed other cases. |
| G07 | Reverted | Lay out ranks by axis within cell blocks for vectorized boundary tests. | 2.77× [2.68, 2.80], 4/16384/1 | The column layout won on wide queries and regressed on the 8D companion. |
| G08 | Reverted | Coalesce consecutive fully covered cells into one contiguous range copy. | 1.02× [1.02, 1.03], 4/16384/1 | Measured gain was limited or inconsistent across screening cases. |
| G09 | Reverted | Replace recursive prefix enumeration with an iterative mixed-radix counter. | 1.00×, 4/16384/1 | At least one companion case regressed in the screen. |
| G10 | Reverted | Skip empty prefix ranges using binary search on occupied keys. | 0.51×, 8/8192/34 | At least one companion case regressed in the screen. |
| G11 | Reverted | Select occupied-key scan versus prefix enumeration from calibrated cost estimates. | 1.07×, 8/8192/34 | At least one companion case regressed in the screen. |
| G12 | Reverted | Cache per-cell decoded coordinates for the occupied-key scan. | 1.27× [1.25, 1.31], 8/8192/33 | Cached cell coordinates helped the occupied-key scan at the cost of extra build memory. |
| G13 | Reverted | Decode key digits using shifts and masks for power-of-two resolutions. | 1.34× [1.31, 1.36], 8/8192/49 | Measured gain was limited or inconsistent across screening cases. |
| G14 | Kept in combination | Replace sparse comparison sort of point IDs with stable radix sorting by cell key. | 1.37× [1.32, 1.40], 8/65537/34 | Stable key radix sorting for sparse grid builds with at least 256 input rows. |
| G15 | Kept in combination | Try a two-level directory for moderately sparse key spaces. | 1.97× [1.89, 2.04], 8/65537/18 | Limit sparse searches by a bounded directory on high key bits. |
| G16 | Reverted | Calibrate the dense first-occupied table cutoff by measured occupancy and memory budget. | 1.06×, 4/4096/32 | Measured gain was limited or inconsistent across screening cases. Half the dense-directory limits. Build and query times are both recorded. |
| G17 | Reverted | Reuse first/last rank-cell boundaries in a per-resolution table for repeated queries. | 1.01×, 4/16384/0 | At least one companion case regressed in the screen. |
| G18 | Reverted | Populate cellRanks in cache-sized output blocks during construction. | 1.06×, 8/65537/18 | At least one companion case regressed in the screen. |
| G19 | Reverted | Parallelize independent cell-rank gathering during construction after stable cell order is established. | 1.14×, 8/65537/18 | At least one companion case regressed in the screen. |
| G20 | Reverted | Parallelize batch queries with per-worker ID buffers and stable prefix-sum assembly. | 1.63×, 4/16384/1 | Four-thread batching sped the wide case and was much slower for narrow boxes. |

## Validation and limits

- Final GCC release package, full Python suite. 343 passed, 1 warning in 193.65s (0:03:13).
- An earlier combined run hit a BiteOpt worker crash in the simulator case. The test also passed in isolation on both the pristine and candidate builds.
- The rebuilt cp312 wheel passed 36 smoke and targeted regression tests after installation into an isolated target.
- Clang 20.1 release build, Python front, jSO and new regression suites. 110 passed, 1 warning in 6.73s.
- All six native CTest targets passed under AddressSanitizer and UndefinedBehaviorSanitizer. Leak detection was disabled because LeakSanitizer failed under the runtime's process restrictions.
- CR-FM-NES, PGPE, jSO and ArchiveDE each used exactly 257 callbacks with one and eight evaluation workers. The isolated checks also replayed the reported results.
- The kernel trace checks compare candidate matrices or archive membership where relevant. The uniformly scaled IGD path changes rounding, and the delivery uses exhaustive-distance tests with a numerical tolerance for that path.
- Alternative search policies were screened at fixed evaluation budgets and left out. The default arithmetic and search choices for the measured scalar workloads were checked by replay against the corrected reference. Timing evidence covers the fixtures in this log and the raw rows.
