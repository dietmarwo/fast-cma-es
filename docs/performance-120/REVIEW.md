# Review of the 120-trial optimizer campaign

27 November 2026. The delivery is
`fast-cma-es-120-trial-performance-patch.zip` (711,817 bytes, SHA-256
`4963a159b9a2a89ea09d63ef9e9d7e56d98754a03ce1ac8a720233bc62f98343`). Its
patch (SHA-256
`5f8ec683582ff9835717195708133c886539fdb299616aa46df8931e7fbc8a1e`) applied
unchanged to `361904d` (fcmaes 2.2.0 plus the native front tools); that is
the first commit of the pull request. [RESULTS.md](RESULTS.md) is the
delivered report, left as delivered; [PLAN.md](PLAN.md) is the plan it
worked from. [`archive.sha256`](archive.sha256) lists the archive's 76
files; its measurements and replay scripts are not imported.

**The correctness fixes are real, the speedups reproduce, and nothing
measured here got worse. Two of the fixes change what users will see far
more than the delivered changelog said:** CMA-ES no longer stops early, and
MAP-Elites runs a CMA-ES emitter that failed silently in 2.2.0. The review
added tests for both and wrote the consequences into the changelog.

## What was measured

Both trees were built and installed here from source (GCC 13.3, Eigen
3.4.1, Python 3.12, NumPy 2.5, two cores; no Numba). All comparisons use
the public wrappers.

| Check | Delivered | Here |
| --- | --- | --- |
| Python tests on the installed package | 308 | 308; 310 after this review |
| Native CTest targets | 6, with sanitizers | 6, release build without sanitizers |
| Paired timing observations | 4,926 | not repeated; wrapper-level timings below |

### Seeded runs: 2.2.0 against the patch

1,155 seeded runs of every scalar and multi-objective wrapper, each with a
digest of every point passed to the objective.

| Entry point | Runs | Same calls and result | Remark |
| --- | ---: | ---: | --- |
| `bitecpp.minimize` (two settings) | 180 | 180 | |
| `Bite_C`, `batch_size=1` | 45 | 45 | |
| `Bite_C`, `batch_size=8` | 45 | 0 | first batches no longer repeat members |
| `decpp.minimize` (plain, integer, with `x0`) | 270 | 259 | same calls in all 270; in 11 the reported point changed |
| `DE_C` ask/tell | 45 | 44 | same calls; one reported point among equal values |
| `modecpp.minimize`, `MODE_C` | 120 | 120 | one to three objectives, constraints, ties, integers, both updates |
| `cmaescpp.minimize`, `ACMA_C` | 450 | 128 | see below |

In 2.2.0, DE reported a value worse than one the objective had returned in
4 of the 180 plain and integer runs; the patch reports the best value seen
in all of them.

### CMA-ES: the history of generation bests

The loop that shifts the history copied its newest entry into every slot.
Two stop criteria and the flat-fitness step-size rule read that history. A
further 225 serial runs with every call recorded:

- In 154 runs the calls of 2.2.0 are an exact prefix of the patched run:
  the search is the same and stops later. In 26 the runs are identical. In
  45, all on an objective with many equal values, the trajectory differs,
  because the flat-fitness rule now fires differently.
- Evaluations rose from 238,044 to 329,772 (39%). The result improved in
  145 runs and worsened in none. On a sphere in two dimensions the final
  value moves from about 1e-15 to 1e-17.
- With generation bests supplied by hand, 2.2.0 stops after five equal
  bests; the patch needs a full window of 25. This is the new regression
  test.

### Worker paths

208 runs with two and four worker threads, odd budgets and delayed
callbacks, counting every real call.

| Path | 2.2.0 | Patch |
| --- | --- | --- |
| Delayed CMA-ES (96 runs) | 71 over budget, 92 with `nfev` different from the calls, 19 reporting worse than the best seen | none |
| Parallel DE (96 runs) | 92 over budget, 18 reporting worse than the best seen (once 9.0 against 5.0) | none |
| Parallel MODE (16 runs) | none | none |

### Quality diversity on the planar arm

Twelve joints, 1,024 niches, 40,000 calls; eight runs with one worker and
five with two, per configuration and build.

| Configuration, one worker | 2.2.0 QD score (median, range) | Patch |
| --- | --- | --- |
| elites | 2,429 (2,423 to 2,442) | 2,431 (2,417 to 2,442) |
| elites + CMA | 2,416 (2,379 to 2,449) | 2,430 (2,410 to 2,453) |
| elites + BiteOpt | 2,457 (2,414 to 2,478) | 2,449 (2,383 to 2,476) |
| BiteOpt | 2,421 (2,333 to 2,470) | 2,454 (2,400 to 2,459) |
| elites + CMA + BiteOpt | 2,455 (2,425 to 2,494) | 2,473 (2,444 to 2,496) |
| elites + CR-FM-NES | 2,446 (2,414 to 2,471) | 2,452 (2,415 to 2,488) |
| elites + DE | 2,279 (2,181 to 2,331) | 2,252 (2,082 to 2,357) |

The ranges overlap in every row, also with two workers: the rewritten
BiteOpt emitter (ask/tell, warm start from the preceding emitter) does not
cost quality here. The real call counts differ: 2.2.0 made 40,032 to
40,960 calls with two workers and 40,448 with the elites emitter alone;
the patch made exactly 40,000 in all 91 runs.

`mapelites.optimize_map_elites` with default parameters and 12 iterations:
2.2.0 made 2,400 calls and filled 504 niches (QD score 1,119); the patch
made about 745,000 calls and filled 832 (2,486). In 2.2.0 the CMA-ES
emitter raised `AttributeError` in each of the 96 iterations; the worker
process printed a traceback and the call returned normally. The call also
failed with `NameError` when asked to create the archive itself. Both are
fixed, and a failing worker now raises in the caller.

### Wrapper-level timings

Optimizer overhead with a vectorized NumPy objective, best of six to ten
repetitions on one pinned core.

| Loop | 2.2.0 | Patch | Ratio |
| --- | ---: | ---: | ---: |
| MODE ask/tell, 2 objectives, population 256, NSGA update | 132.6 ms | 31.4 ms | 4.22 |
| the same with the DE update | 123.0 ms | 25.9 ms | 4.76 |
| MODE, 2 objectives and 2 constraints, population 256 | 134.4 ms | 40.1 ms | 3.35 |
| MODE, 5 objectives, population 256 | 199.5 ms | 137.9 ms | 1.45 |
| MODE, 1 objective, population 256 | 132.5 ms | 22.0 ms | 6.02 |
| MODE, 2 objectives, population 64 | 15.5 ms | 8.4 ms | 1.84 |
| DE ask/tell, 30 variables, population 128 | 13.6 ms | 11.8 ms | 1.15 |
| CMA-ES with supplied points, 30 variables | 37.0 ms | 33.7 ms | 1.10 |
| CMA-ES ask/tell, 10 variables, population 32 | 2.9 ms | 3.1 ms | 0.92 |
| `bitecpp.minimize`, 30 variables, 30,000 calls | 46.7 ms | 43.4 ms | 1.08 |

The MODE figures match the report (3.96 and 5.27 for two objectives, 2.35
constrained, 1.58 for five objectives). The DE, BiteOpt and
supplied-point gains are smaller than reported but present. The small
CMA-ES loop was 8% slower here on a 3 ms measurement; the report has no
figure for that case, and the larger one is level.

## Findings and changes

1. **The changelog understated two changes of behavior.** "Fitness history
   now advances correctly" stands for 39% more evaluations and better
   results in most CMA-ES runs. "MAP-Elites can ... start its native CMA
   emitter" stands for a default cost per iteration that rises from 200 to
   up to 62,200 calls. The changelog now opens with a section on what to
   expect, including how to get the former MAP-Elites cost
   (`cma_params={'cma_generations': 0}`), and `ai-context.md` states the
   cost of an iteration.
2. **Neither change had a test of its own.** Added: the history stop needs
   a full window of equal bests (fails on 2.2.0, where five suffice), and
   MAP-Elites makes exactly the variation calls with the emitter off and
   more, within its bound, with it on (fails on 2.2.0).
3. **The Diversifier's BiteOpt emitter changed without a quality check in
   the report** (two of its trials were kept at measured slowdowns of 0.83
   and 0.90). The arm comparison above closes that gap: no loss.

## Not verified here

- The sanitizer runs, the 30-seed holdouts and the fixed-wall comparisons
  of the report, and the grid lookup without Numba (16.98).
- `persistent_workers`, `batch_fitness`, `inline` and `strict_budget`
  beyond the delivered tests, which pass.
- Runs with more than four workers.

## Open points

- CMA-ES users with tuned evaluation budgets, and retry strategies that
  rely on early stops, will see longer runs. `stop_hist` and `accuracy`
  remain the controls.
- The order of archive updates with several processes still depends on
  scheduling; a seed makes single-worker runs repeatable.
