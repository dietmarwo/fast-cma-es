# fast-cma-es performance campaign implementation log

The campaign executed **120 proposals** across Active CMA-ES, BiteOpt, DE, MODE, Diversifier and MAP-Elites. The patch retains **60 changes** and **7 opt-in changes**. Four other experiments used existing settings and 49 candidates stayed outside the patch. The ZIP contains the diff, benchmark source, raw pairs, the input plan and validation logs.

**Base commit** `361904d747bbaebfe0d065978fd4e285fdc0fbc0` on `master`  
**Package** `fcmaes` 2.2.0 source checkout  
**Patch SHA-256** `5f8ec683582ff9835717195708133c886539fdb299616aa46df8931e7fbc8a1e`  
**Execution date** 27 November 2026

## Results from the integrated patch

Each row uses seven matched seed or repetition pairs after warm-up. Speedup is the median of the individual reference/patch ratios. The brackets give a paired bootstrap 95% interval. The two time columns independently summarize elapsed milliseconds. Kernel rows report work units in the calls column. Both QD and C++ timings came from one shared AMD EPYC 9V74 Linux host.

| Workload | Calls | Reference ms | Patch ms | Paired speedup |
| --- | --- | --- | --- | --- |
| CMA-ES, supplied points, 128D/P64 | 4,096 | 344.28 | 218.30 | 1.21× [1.18–1.33] |
| CMA-ES, standard ask/tell, 128D/P64 | 4,096 | 366.73 | 368.47 | 1.03× [0.98–1.25] |
| BiteOpt, native scalar, 16D | 8,192 | 4.18 | 3.80 | 1.11× [1.04–1.18] |
| BiteOpt, 96D, batch width 32, corrected reference | 8,192 | 13.88 | 13.07 | 1.06× [1.00–1.07] |
| BiteOpt, same batch against pinned checkout | 8,192 | 11.64 | 13.13 | 0.89× [0.75–1.01] |
| DE, native ask/tell, 96D/P32 | 8,192 | 6.85 | 6.07 | 1.12× [1.00–1.20] |
| DE, native Cassini1 | 1,038 | 7.92 | 7.66 | 1.01× [0.53–1.04] |
| MODE, 2 objectives, SBX, P256 | 8,192 | 32.46 | 8.12 | 3.96× [3.67–4.22] |
| MODE, 2 objectives, DE, P256 | 8,192 | 33.07 | 7.13 | 5.27× [4.56–6.44] |
| MODE, 2 objectives with constraint, P128 | 8,192 | 22.12 | 9.63 | 2.35× [2.20–2.50] |
| MODE, 5 objectives, P256 | 8,192 | 67.00 | 42.30 | 1.58× [1.53–1.60] |
| Diversifier, CMA emitter, grid, corrected reference | 2,048 | 53.49 | 45.41 | 1.16× [1.03–1.60] |
| MAP-Elites, 4,096-cell grid, JIT-free | 4,096 | 86.67 | 51.71 | 1.37× [1.37–1.45] |
| MAP-Elites, 4,096 grid lookups, JIT-free | 4,096 | 5.39 | 0.39 | 16.98× [13.68–20.48] |
| MAP-Elites, stats-on archive updates | 4,096 | 69.20 | 32.43 | 1.99× [1.89–2.04] |

The BiteOpt batch fix replaces duplicate initialization with distinct candidates. It costs work against the original batch path. The corrected baseline measures that work on both sides. Scalar BiteOpt is also compared directly with the pinned checkout. The CMA history correction can let a run continue past the former premature plateau stop. C09 altered floating-point trajectories when recovering coordinates from supplied points. In the 30-seed native supplied-point holdout, the paired patch/reference loss ratio was 0.995 with bootstrap interval [0.886, 1.199]. The time ratio in the same experiment is shown below.

| Supplied-point control | Calls | Reference ms | Patch ms | Paired speedup |
| --- | --- | --- | --- | --- |
| 128D native, 30 held-out seeds | 4,096 | 218.64 | 178.18 | 1.18× [1.14–1.23] |

## Opt-in entry points

Both sides use the integrated build. Vectorized measurements compare the specified NumPy row objective with its matrix equivalent. They include objective batching and binding overhead. Cassini1 still evaluates each trajectory inside the matrix callback.

| Feature and workload | Calls | Default ms | Opt-in ms | Paired speedup |
| --- | --- | --- | --- | --- |
| CMA, matrix objective on shifted Sphere | 4,096 | 60.18 | 9.03 | 6.67× [6.54–6.84] |
| CMA, matrix objective on Cassini1 | 2,048 | 48.07 | 25.32 | 1.82× [1.81–1.97] |
| MODE, matrix objective on ZDT1 | 4,096 | 148.90 | 15.34 | 10.54× [7.84–11.77] |
| Diversifier, Bite batch width 16, cheap grid | 2,048 | 121.33 | 60.43 | 2.11× [1.78–2.80] |
| Diversifier, CMA with matrix QD callback, arm | 2,048 | 102.66 | 49.28 | 2.21× [2.02–2.74] |
| Diversifier, inline single worker, cheap grid | 2,048 | 40.63 | 21.05 | 1.96× [1.71–2.05] |
| MAP, persistent worker across ten iterations | 1,920 | 370.45 | 98.72 | 3.77× [3.28–4.05] |

Use `vectorized=True` with one worker for `cmaescpp.minimize` and `modecpp.minimize`. MODE expects objectives first, followed by constraints. Diversifier accepts `batch_fitness` and BiteOpt `batch_size` in its emitter options. `inline=True` runs Diversifier's single worker in the caller. `persistent_workers=True` reuses MAP-Elites processes across iterations and preserves local objective state between them.

## Search and correctness decisions

- The `C17` covariance refresh setting of 8 gave more progress in the fixed-time rotated and Rosenbrock fixtures. Fixed-call accuracy varied with the refresh interval, so the campaign recorded it through the existing `update_gap` option.
- `B15` at population cap 96 brought median 128D Rastrigin loss from approximately 697 to 555 at 4,096 calls. It also improved the fixed-wall rotated fixture. `Bite_C(..., popsize=96)` already exposes that configuration.
- `M19` used `pareto_update=1.5` with the DE update. Both ZDT1 and ZDT3 held-out medians improved. The bias remains a setting for a chosen workload. The ZIP includes fixed-reference hypervolume and inverted generational distance.
- `V06` at width 16 improved speed in the 30-seed cheap-grid and arm holdouts. The cheap-grid median QD score moved from 87.13 to 83.44 and the arm median moved from 358.9 to 370.4. The default feedback width is one.
- The correctness checkpoint fixes DE best-observed results and delayed candidate identity, CMA double decoding and history movement, and BiteOpt's duplicate initial proposals. QD fixes include fresh archives, CMA emitters, shared evaluation budgets, process failure propagation and complete elite/statistics updates.

## Experiment record

The timing column shows the primary screening fixture and median paired speedup. Whole correctness checkpoints pair the pinned build with the corrected reference. Their timing includes the combined fixes. Individual correctness claims come from the regression and stress probes. Exact trials compare full batch traces where available. A compiled native case reports evaluation counts and observed objective values in the raw evidence.

### Active CMA-ES

| ID | Decision | Primary screen | Finding |
| --- | --- | --- | --- |
| C01 | Kept | `delayed/low/bounds` 1.18× | Delayed batches queue encoded coordinates. The eight-worker and noncentral-bound probes saw feasible points and matching incumbent costs. |
| C02 | Kept | `delayed/small/sphere` 0.94× | Reserved the delayed window, consumed completed results and recorded the best even when the budget ends before a full generation. |
| C03 | Kept | `state/small/plateau` 0.14× | Shifted the fitness history backward. The plateau case used more calls and reached a better loss after the premature stop was corrected. |
| C04 | Held | `state/high/sphere` 1.08× | One GEMM was exact on the small fixture. At 128D it changed floating-point trajectories and gave no repeatable end-to-end speedup. |
| C05 | Held | `state/high/sphere` 1.10× | Reusing the Eigen solver was trace-equivalent. The isolated native repeats did not establish a throughput benefit. |
| C06 | Kept | `state/high/sphere` 0.92× | Mirrored the upper covariance triangle in place. Native runs matched outcomes and removed two full temporary matrices. |
| C07 | Held | `state/high/rotated` 0.92× | Self-adjoint rank updates changed rounding and search trajectories. Both holdout workloads showed timing near parity. |
| C08 | Held | `state/high/rotated` 0.98× | Direct subtraction of the active rank update matched traces on both cases. Repeated gains were below the selection threshold. |
| C09 | Kept | `external/high/sphere` 1.46× | Used the orthogonal eigensystem and diagonal scales for supplied or delayed points. Thirty native and thirty public seed pairs showed faster updates. Paired loss uncertainty includes parity. |
| C10 | Held | `repair/high/bounds` 1.02× | Packing repaired columns gave different rounding and no clear improvement across clipping-heavy holdouts. |
| C11 | Held | `state/high/sphere` 1.06× | Parent gather buffers matched traces. The extra retained storage did not earn a repeatable speedup. |
| C12 | Kept | `direct/small/sphere` 1.15× | Decode writes directly to the batch output buffer. Removes two coordinate temporaries per callback row and matches scalar callback results. |
| C13 | Kept | `state/high/sphere` 1.11× | Shared the capsule-owned Eigen matrix conversion across CMA, Bite and DE. Held arrays retain their own read-only data across later asks. |
| C14 | Held | `state/high/sphere` 0.97× | Bulk population feedback shared the generation finishing routine. Traces matched and native timings were near parity. |
| C15 | Held | `state/low/sphere` 1.08× | Invariant path scalars produced the same traces. The small-D screen did not give a clear end-to-end gain. |
| C16 | Opt-in | `direct/small/sphere` 5.97× | Vectorized public CMA callback accepted after candidate counts, seeded result parity and malformed output tests passed. |
| C17 | Existing setting | `state/high/rotated` 3.36× | Update gap 8 gave more progress under a 90 ms allowance on both high-dimensional fixtures, with some loss of equal-call accuracy. The existing update_gap option covers this trade-off. |
| C18 | Held | `state/high/rotated` 1.24× | Initial diagonal-only covariance made some runs faster. Quality and speed shifted by workload and transition length. |
| C19 | Existing setting | `state/high/bounds` 0.78× | Repairing clipped z improved the bounded median with more evaluations on that fixture. The existing repair_z option remains the tested control. |
| C20 | Held | `direct/small/rastrigin` 1.07× | Growing the population across four restarts worsened the held-out medians on Rastrigin and Cassini1. |

### BiteOpt

| ID | Decision | Primary screen | Finding |
| --- | --- | --- | --- |
| B01 | Kept | `batch/small/sphere` 1.03× | Each pending initialization candidate comes from a different unevaluated population row. Compared sequential and batched proposals, deep mode and short last batches. |
| B02 | Kept | `single/small/sphere` 1.01× | Avoided sorting for singleton feedback. Included with the copy reductions and validated against the corrected reference. |
| B03 | Kept | `batch/large/sphere` 1.17× | Passed Eigen costs directly, removing the redundant per-tell allocation and copy. |
| B04 | Kept | `batch/large/sphere` 1.07× | One pending proposer identifies an entire batch. Deep-mode traces and outcomes matched. |
| B05 | Held | `batch/large/sphere` 0.78× | Lazy workspace allocation reduced the levels allocated on first ask. The deep timing fixture showed no repeatable speedup. |
| B06 | Kept | `batch/large/sphere` 1.15× | Wrote the real-valued proposals directly into saved batch rows. The restored feedback path reads those rows when needed. |
| B07 | Kept | `batch/large/sphere` 1.00× | Used saved real values on accepted proposals. Jointly with B06 this removed a round trip through the real-valued scratch buffer. |
| B08 | Kept | `direct/small/sphere` 0.96× | The scalar loop uses a stack result without the one-element Eigen result vector. Native Sphere improved and Cassini1 remained near parity. |
| B09 | Kept | `batch/wide/sphere` 1.00× | Filled the candidate matrix with the existing bulk copy and transferred its storage to a NumPy capsule. |
| B10 | Held | `batch/high/sphere` 1.19× | A contiguous parallel cost index added storage and writes. Native tests showed no consistent win over pointer insertion. |
| B11 | Held | `batch/large/sphere` 0.79× | Running-distance bounds preserved selected outcomes. Hot-loop short-circuiting did not beat the existing centroid loop. |
| B12 | Held | `batch/high/sphere` 0.90× | Separate wrap and decode loops were trace-exact. Both primary timings were near parity or slower. |
| B13 | Held | `batch/low/sphere` 1.01× | Small-batch insertion sort preserved order. Native runs showed no stable gain at either tested width. |
| B14 | Held | `batch/wide/sphere` 1.00× | Selector-ID lookup matched traces. Translating the pointers added work in the tested configurations. |
| B15 | Existing setting | `batch/high/rastrigin` 0.96× | A cap of 96 improved the high-D Rastrigin holdout and fixed-wall progress on the two fixtures. An explicit popsize already exposes that configuration. Caps 160 and 256 were also measured. |
| B16 | Held | `batch/large/rastrigin` 1.24× | Smaller initialization batches preserved best losses on the tested continuous fixtures, with no consistent fixed-wall gain. |
| B17 | Held | `direct/small/rastrigin` 1.05× | Compared depth 2 with independent depth-1 restarts over matched budgets. Neither configuration dominated on Rastrigin and Cassini1. |
| B18 | Held | `direct/small/plateau` 0.92× | Restarted or continued from the last best after measuring progress within a budget chunk. The noise and plateau results diverged by case. |
| B19 | Held | `batch/small/plateau` 0.78× | Rejecting identical costs reduced population variety on the quantized fixture and worsened the Rastrigin holdout. |
| B20 | Held | `batch/small/rastrigin` 1.26× | Ask-order feedback changed the search and worsened equal-call losses on the shifted multimodal and rotated cases. |

### Differential evolution

| ID | Decision | Primary screen | Finding |
| --- | --- | --- | --- |
| D01 | Kept | `state/low/sphere` 0.99× | Observed finite values independently of the active mutation anchor. Row-zero and exact-candidate incumbent checks passed. |
| D02 | Opt-in | `direct/low/sphere` 1.00× | strict_budget counts each primary and temporal trial in serial mode. The delayed worker path reserves submitted calls by design. |
| D03 | Kept | `delayed/low/sphere` 0.80× | Recycled only completed evaluation slots. A worker delayed beyond ten population windows retained its original candidate. |
| D04 | Kept | `delayed/large/sphere` 0.87× | Workers exchange bounded slot indices. Shared state owns candidate storage through completion and shutdown. |
| D05 | Kept | `direct/large/sphere` 1.09× | The stack scalar path distinguishes unevaluated work and invalid callbacks. Native speed varied so this was retained with the accounting fixes. |
| D06 | Kept | `direct/large/sphere` 1.00× | Read donor and target columns as Eigen views. Native serial 96D timing improved with matching final losses. |
| D07 | Kept | `state/large/sphere` 0.97× | Wrote ordinary DE asks into pending columns. The temporal path retained its FIFO target mapping. |
| D08 | Kept | `state/large/sphere` 1.32× | Fused batch crossover and coordinate clipping without changing its random draws on the measured traces. |
| D09 | Kept | `state/large/mixed` 1.16× | Cached integer coordinate indices with the same random-draw order. The native mixed-variable 96D fixture improved. |
| D10 | Kept | `state/large/sphere` 1.01× | Stored each temporal candidate and index together in a movable queue entry, removing a copy and synchronized queue bookkeeping. |
| D11 | Held | `direct/large/bounds` 0.95× | Split uniform and normal repair callbacks inside the serial loop. Both high-bound-hit and ordinary serial timings were near parity. |
| D12 | Held | `state/large/bounds` 1.17× | Reused a normal-sampling sigma buffer. The agreeing traces did not establish an end-to-end benefit. |
| D13 | Kept | `direct/large/sphere` 1.15× | Kept reusable trial vectors as part of the serial copy reductions. The standalone timing gain was small. |
| D14 | Kept | `state/wide/sphere` 1.04× | Transferred the owned ask/population buffer directly to NumPy. The public row layout and lifetime checks passed. |
| D15 | Kept | `state/low/sphere` 0.85× | Rejected populations too small to supply two donors after excluding the target and mutation anchor. |
| D16 | Held | `direct/low/rastrigin` 0.96× | Unbiased exclusion-mapped sampling improved Rastrigin under an equal-call cap. Fixed-wall results and rotated timing were mixed. |
| D17 | Held | `direct/small/rastrigin` 0.88× | Gating the follow-up trial produced no consistent gain on Rastrigin or Cassini1. |
| D18 | Held | `direct/small/rastrigin` 0.89× | Rank-biased mutation anchors improved Rastrigin and worsened the rotated holdout. The trade-off persisted across budgets. |
| D19 | Held | `direct/small/plateau` 1.25× | Contraction-triggered reinitialization rarely fired in the measured budgets and delivered no established gain. |
| D20 | Held | `direct/large/mixed` 0.99× | Success-weighted F/CR adaptation worsened the mixed-variable holdout. Cassini1 showed no clear offsetting benefit. |

### MODE

| ID | Decision | Primary screen | Finding |
| --- | --- | --- | --- |
| M01 | Kept | `state/wide/zdt1` 3.28× | Exact two-objective levels use a prefix-max sweep. Equal objectives are grouped before the sweep inserts them. The selected population traces matched. |
| M02 | Kept | `state/wide/scalar` 2.38× | Computed one-objective levels from sorted groups of equal scores. Single-objective traces matched. |
| M03 | Kept | `state/wide/dtlz5` 1.56× | Broke out of a dominance comparison once both candidates were known to be better on some axis. Native five-objective timing improved. |
| M04 | Held | `state/wide/dtlz5` 0.96× | Flat adjacency storage preserved rank outcomes. Repeated three- and five-objective runs showed parity or a slowdown. |
| M05 | Kept | `state/max/tied` 0.88× | Partitioned survivor indices into stable level buckets. Alone the gain was weak. Removing it from the combined fast sorter slowed the P512 tie workload. |
| M06 | Kept | `state/wide/zdt1` 1.00× | Gathered selected candidates into reusable matrices. The combined two-objective path benefited more than the isolated trial. |
| M07 | Held | `state/max/zdt1` 0.95× | Scratch order storage preserved tie behavior. Crowding remained a small part of the measured budget. |
| M08 | Held | `state/max/zdt1` 1.09× | Partial crowding selection fell back on ties. After the extra selection checks it did not establish an independent gain. |
| M09 | Held | `state/max/constrained` 1.03× | Accumulated ranks after sorting each constraint column. Dense feasible and constrained fixtures showed no clear timing gain. |
| M10 | Kept | `state/max/feasible` 0.97× | Skipped redundant constrained ranking when every row is finite and feasible. General constraint ordering keeps the existing path. |
| M11 | Held | `state/wide/zdt1` 0.80× | Wrote offspring directly from SBX parent views. Candidate traces matched, scoring was unchanged and no timing win repeated. |
| M12 | Held | `state/wide/zdt1` 0.99× | Hoisted crossover and mutation constants. Compilation and end-to-end timing showed no clear benefit. |
| M13 | Kept | `de/large/zdt1` 1.03× | Wrote DE offspring directly into the pending half of the population. Mixed and continuous MODE selection regressions passed. |
| M14 | Held | `de/large/zdt1` 1.07× | Cached integer coordinate indices in MODE. It produced no clear gain on the tested populations. |
| M15 | Held | `state/large/zdt1` 1.01× | Transferred aligned SBX blocks all at once. No repeatable improvement justified another branch for switching. |
| M16 | Kept | `state/wide/zdt1` 0.91× | Ordinary tell maps C-contiguous objective rows as native columns. Shape checks and the caller-owned array lifetime remain enforced. |
| M17 | Opt-in | `row/wide/zdt1` 2.75× | Batched public MODE objectives gave identical returned points and costs in the regression fixture and faster NumPy callbacks. |
| M18 | Held | `de/wide/zdt3` 0.99× | Alternating DE/SBX blocks changed fronts without a stable HV and IGD win across constrained and unconstrained cases. |
| M19 | Existing setting | `de/wide/zdt3` 0.96× | Elite bias at 1.5 improved median ZDT1 and ZDT3 HV in the holdout and under fixed wall time. pareto_update already exposes the experiment. |
| M20 | Held | `de/wide/zdt3` 0.97× | Periodic injection from a bounded 64-point archive gave mixed constrained and disconnected-front results. |

### Diversifier

| ID | Decision | Primary screen | Finding |
| --- | --- | --- | --- |
| V01 | Kept | `div-process/small/sphere` 0.89× | Passed the stats flag by keyword on fresh archives. Recorded moments and flags in tests. |
| V02 | Kept | `div-process/small/sphere` 0.92× | A root seed derives initialization, worker streams and CVT. Single-worker results replay with a deterministic objective. |
| V03 | Kept | `div-w2/small/collision` 0.93× | Reserved real callback counts under one lock and evaluated the last short batch. The eight-worker stress test met its 517-call cap. |
| V04 | Kept | `div-w2/small/collision` 0.91× | Surfaced failed workers and let sibling workers stop between batches. The archive remained reusable after an injected objective exception. |
| V05 | Kept | `bite-v/small/sphere` 0.83× | Switched scalar Bite feedback to public Bite_C with a width of one by default. Real incumbent points can pass between chained stages. |
| V06 | Opt-in | `bite-v/small/sphere` 1.83× | Bite batches share descriptor lookup. Widths 4, 16 and 32 were tried. Width 16 improved throughput, with a cheap-grid versus arm quality trade-off. |
| V07 | Opt-in | `div-process/small/sphere` 1.51× | Native emitters accept one matrix callback per population. The pipeline validates score and descriptor row counts before archive writes. |
| V08 | Kept | `update/large/sphere` 1.03× | Gathered old elite values from shared numeric storage. Repeated niche IDs use the same feedback snapshot. |
| V09 | Kept | `update/large/sphere` 1.12× | Reduced masks and list work in empty-niche normalization. Archive update traces matched on the tested mixed and collision cases. |
| V10 | Kept | `div-process/small/sphere` 0.66× | Sorted each transformed score vector once and reused it for the stall test and next generation. |
| V11 | Held | `de-local/small/sphere` 0.98× | A native DE reset prototype matched restart traces. Retaining the solver state did not improve the measured cheap loops. |
| V12 | Held | `update/large/sphere` 0.98× | Worker scratch for objective and descriptor conversion gave no clear gain and would restrict returned-array ownership. |
| V13 | Opt-in | `div-process/small/sphere` 1.87× | Inline execution avoids child startup for one worker. Seeded archive results were identical on both feature fixtures. |
| V14 | Kept | `chain/small/sphere` 2.80× | Stopped chained stages when the shared budget was exhausted. The partial budget produced 103 calls. |
| V15 | Kept | `chain/small/sphere` 0.90× | The Bite stage receives the caller's warm start and returns the best real objective point observed during that stage. |
| V16 | Held | `defaults/small/sphere` 1.80× | Skipping eager RNG draws made configured setup cheaper, but changes the seed stream. There was no matching end-to-end benefit on the emitter runs. |
| V17 | Held | `local/small/sphere` 1.04× | Shorter early restarts based on coverage improved one cheap score and degraded the planar arm. This prototype used coverage as a proxy for discovery rate. |
| V18 | Held | `choose/small/sphere` 1.02× | Allocating restarts by recent QD gain per call underperformed the fixed mixed sequence. |
| V19 | Held | `centers/small/sphere` 0.97× | Starting some runs near seldom-visited elites improved the arm fixture. Larger and independent cases are needed before making it an emitter policy. |
| V20 | Held | `buffer/small/sphere` 0.96× | Buffered submissions produced stale feedback scores. Tested synchronization periods changed coverage and did not accelerate the shared path. |

### MAP-Elites

| ID | Decision | Primary screen | Finding |
| --- | --- | --- | --- |
| Q01 | Kept | `init/large/sphere` 1.01× | Restored fresh MAP archives and forwarded the stats flag by keyword. Grid and CVT archive creation were verified. |
| Q02 | Kept | `local/small/arm` 1.00× | Selected a ranked occupied elite for CMA and used a checked fallback when no cell is occupied. Native CMA work was observed in the end-to-end regression. |
| Q03 | Kept | `map-pool/small/sphere` 0.91× | Propagated nonzero child exits to the caller and stopped siblings at a batch boundary when possible. |
| Q04 | Kept | `map-pool/small/sphere` 0.83× | A root seed controls fresh decisions, worker streams and CVT setup. Workers on identical single-process inputs replay. |
| Q05 | Opt-in | `map-pool/small/sphere` 3.02× | Persistent worker processes preserved the seeded archive result across ten rounds. The option also preserves per-process callback state between rounds. |
| Q06 | Kept | `sort/large/sphere` 2.10× | Sorted from a numeric view of shared fitness storage. Public get_ys still produces an independent result. |
| Q07 | Kept | `local/large/sphere` 0.96× | Gathered ranked parents with NumPy array indices. Exact selection traces matched and the Python generator loop was removed. |
| Q08 | Held | `sort/large/sphere` 1.44× | Incremental ranking helped a dense archive with few dirty cells. Sparse archives and ties hit a full-sort fallback after extra work. |
| Q09 | Held | `submit/large/collision` 1.03× | Niche-grouped submission preserved per-cell order. Both all-collision and mostly distinct streams showed weak timing. |
| Q10 | Kept | `div-w2/small/sphere` 0.93× | Locked complete elite updates, occupancy and stats per stripe. An eight-process collision test verified the winner and every count. |
| Q11 | Kept | `stats/large/sphere` 1.74× | Updated moments and extrema in shared array views. Stats-on archive replays kept the same elite and moment data. |
| Q12 | Kept | `lookup/large/sphere` 44.77× | Added native grid lookup for installations without Numba. Boundary and integer-bound tests reproduced the legacy cell mapping. |
| Q13 | Kept | `cvtlookup/large/sphere` 1.12× | Queried CVT nearest indices without returning distances. Identical niche results and a small lookup improvement. |
| Q14 | Kept | `cvtcold/short/sphere` 1.05× | Wrote complete compressed CVT caches atomically and keyed seeded geometry by seed. Concurrent cold load and replay passed. |
| Q15 | Kept | `sbx/wide/sphere` 1.01× | Broadcast upper and lower bounds for SBX mutation. Candidate traces matched and avoided two dense tiled bounds matrices. |
| Q16 | Kept | `sbx/wide/sphere` 1.34× | Applied SBX powers only to unmasked sites, retaining the existing random draws and coefficient formula. |
| Q17 | Kept | `iso/wide/sphere` 1.00× | Reused the Iso+Line random buffers for arithmetic and clipping. Fewer arrays with matched offspring. |
| Q18 | Kept | `init/large/sphere` 35.66× | Initialized fitness, descriptors and stats in bulk. Resetting a large archive was substantially faster. |
| Q19 | Held | `local/small/sphere` 1.07× | Coverage-dependent variation made the arm score worse on held-out seeds. The cheap projected sphere did not justify a default. |
| Q20 | Held | `local/small/sphere` 1.03× | Tried 10% and 30% CMA allocation under the same requested calls. Coverage and score trade-offs varied across the two objectives. |

## Benchmark design and verification

- **4,926 usable paired observations** after warm-up and retry deduplication. This includes 1,526 primary screening pairs, 1,950 search holdout pairs, 210 fixed-wall pairs and 259 one-change removals from the selected combination. The artifacts contain the other benchmark phases, raw rows from retries and the decision CSV.
- Held-out search seeds were 1001–1030. Fixed-wall checks used seven fresh seeds and a measured 30 or 90 ms allowance, depending on optimizer. Ask/tell checks finish a generation and QD checks finish the current group of proposals. Raw rows record the resulting counts and elapsed time.
- Instrumented variants were built separately from the patch. The accepted exact changes were replayed through leave-one-out checks. A/B comparisons for public options ran against the final combined build. Fast-path outputs, allocation reductions and correctness were also used to keep simple changes when their individual wall-time intervals overlapped parity.
- Release toolchain was GCC 14.2.0, C++17, Eigen 3.4.1, CMake 3.31.10 and nanobind 3.1.0. Python was 3.12.14 with NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. A separate lookup run used Numba 0.63.1 and llvmlite 0.46.0. BLAS and OpenMP were limited to one thread.
- The installed CPython wheel passed **308 Python tests**. The six native CTest targets passed with AddressSanitizer and UndefinedBehaviorSanitizer. Leak detection was disabled for CTest because LeakSanitizer could not attach in the execution environment. Hardware-counter and second-host comparisons were unavailable.
- Separate installed-package probes covered 2 and 8 delayed workers, strict budgets around the initial window, a DE straggler, 1,032 colliding archive updates, seeded CVT cache races, process spawning, saved-archive reload and cooperative failure handling. The two-worker and eight-worker Diversifier runs each used exactly 517 callbacks.
- Release comparisons also reevaluated saved decisions and behavior descriptors against the repository's planar-arm objective. The warm-up and seven per-seed application runs appear under `evidence/app/`. Parallel QD archive updates remain order-dependent. End-to-end timings for simulated dynamics, expensive objectives and large worker counts depend on the actual objective and host.

## Applying the patch

The ZIP contains `fast-cma-es-performance.patch`. Extract it next to a clean checkout at the base SHA, then run the following from the checkout. The C++ files keep their existing line endings.

```bash
git -c core.whitespace=cr-at-eol apply --check /path/to/fast-cma-es-performance.patch
git -c core.whitespace=cr-at-eol apply /path/to/fast-cma-es-performance.patch
```

[Base revision](https://github.com/dietmarwo/fast-cma-es/tree/361904d747bbaebfe0d065978fd4e285fdc0fbc0)
