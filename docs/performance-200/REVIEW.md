# Review of the remaining C++ campaign

27 November 2026. The delivery is
`fast-cma-es-remaining-cpp-performance-patch.zip` (833,953 bytes, SHA-256
`92e18d66e6e5d97f0a06b3c034c2289d9c4add57575b6792b78c562f4581454f`). Its
`changes.patch` (SHA-256
`679bcd9c071094049aaedfc15cd37130ffffe3ef765086a78eaa1f52f32e68ef`) applied
unchanged to its base `2cea79f`; that is the first commit of the pull
request. The branch then merged the prepared, untagged 2.3.0, where only
the changelog conflicted. [RESULTS.md](RESULTS.md) is the delivered report,
left as delivered; [PLAN.md](PLAN.md) is the plan it worked from.
[`archive.sha256`](archive.sha256) lists the archive's 497 files.

**The four correctness fixes are real and the speedups of IGD and of the
rank indexes reproduce. One headline needs a caveat and one result was a
regression:** the 74-fold gain of `ParetoArchiveND` applies only to fronts
whose objective sums agree to rounding, and on every other large front of
three to six objectives the delivered archive was 10% slower than 2.2.0.
The review removed that slowdown and made five and six objectives faster
instead.

## What was measured

Three installs were built here from source (GCC 13.3, Eigen 3.4.1, Python
3.12, two cores): the prepared 2.3.0 without this change, the delivered
patch, and the reviewed tree. All comparisons use the public wrappers.

| Check | Delivered | Here |
| --- | --- | --- |
| Python tests on the installed package | 343 | 343 with 2.3.0 merged; 368 after this review |
| Sanitizer, Clang and wheel checks | pass | not repeated |
| 30 paired timings per case | 52 cases | wrapper-level timings below |

### Seeded runs of the optimizers

2,133 seeded runs with a digest of every call: six objectives, among them
one with ties, one with tiny differences and one with its optimum outside
the box; three dimensions, three seeds, budgets of 33, 257, 1,000 and
3,001 at a population of 32.

| Entry point | Runs | Prepared 2.3.0 | With the patch |
| --- | ---: | --- | --- |
| `crfmnescpp.minimize` | 432 | 424 over budget (up to 31 calls); 55 report a value worse than one evaluated | exact budget; none |
| `pgpecpp.minimize` | 432 | 432 over budget (up to 31 calls) | exact budget |
| `dacpp.minimize`, local search | 216 | 51 over budget, up to 21.6 times (714 calls for 33) | exact budget |
| `dacpp.minimize`, no local search | 216 | within budget; `nit` always 0 | the same calls; `nit` counts |
| `jsocpp.minimize` | 432 | | identical |
| `archivede.minimize` | 81 | | identical |
| matrix objective, three wrappers | 324 | not available | equal to the scalar run call for call; rows equal calls |

With the patch every result is the best value evaluated, at the point
that produced it and inside the box. Eight CR-FM-NES runs end early with a
negative stop code in both builds; they now report `success=False`.

The exact budgets cost results at equal settings, because fewer calls are
made: of the 51 Dual Annealing runs that had overrun, 38 end worse. That
is the price of a budget that means what it says; the report's Cassini1
figures (34.95 with 1,183 to 5,096 calls against 41.86 with 1,024) show
the same.

### Front and rank tools

705 generated inputs, compared between the prepared 2.3.0 and the reviewed
tree and, where cheap enough, with a definition-level computation.

| Tool | Inputs | Result |
| --- | ---: | --- |
| `ParetoArchiveND`, 2 to 8 objectives, 10 kinds of stream, mixed `add` and `add_batch` | 150 | identical masks, fronts and payloads; 100 checked against strict dominance by definition |
| `ParetoArchive2D` | 20 | identical |
| `igd`, 1 to 12 dimensions, five scalings | 175 | 122 bit-identical, 53 within a relative 2.5e-15 |
| `BordaBoxIndex`, `GridBoxIndex`, 2 to 8 dimensions, ties and clusters, three box widths | 360 | identical id lists, all equal to a brute-force filter |

### Timings

Best of three rounds on one pinned core.

| Case | Prepared 2.3.0 | Delivered | Reviewed |
| --- | ---: | ---: | ---: |
| `ParetoArchiveND`, 5 objectives, 4,096 points, sums equal to rounding | 70.0 ms | 1.15 ms | 1.15 ms |
| the same, sums within 1e-3 | 53.9 ms | 59.9 ms | 32.1 ms |
| the same, uniform points (front of 281) | 0.70 ms | 0.76 ms | 0.47 ms |
| 3 objectives, sums within 1e-3 | 48.1 ms | 54.5 ms | 44.7 ms |
| 4 objectives, sums within 1e-3 | 52.6 ms | 59.5 ms | 50.4 ms |
| 6 objectives, sums within 1e-3 | 55.6 ms | 62.4 ms | 37.9 ms |
| 8 objectives, sums within 1e-3 | 57.7 ms | 42.1 ms | 37.9 ms |
| `igd`, 1 dimension, 4,096 front and reference points | 21.9 ms | 0.40 ms | |
| `igd`, 4 dimensions, 4,096 and 512 | 2.77 ms | 0.46 ms | |
| `igd`, 8 dimensions, 4,096 and 512 | 4.65 ms | 1.88 ms | |
| `igd`, 8 dimensions, coordinates near 1e150 | 27.5 ms | 0.62 ms | |
| `GridBoxIndex`, 8 dimensions with ties, build / 300 queries | 20.9 / 0.83 ms | 17.1 / 0.65 ms | |
| `BordaBoxIndex`, the same, 300 queries | 7.24 ms | 4.67 ms | |
| PGPE, 128 variables, 8,192 calls | 22.6 ms | 18.2 ms | |
| CR-FM-NES, Dual Annealing, jSO, 16 and 128 variables | | 0.99 to 1.08 | |
| matrix objective against scalar, 16 variables: PGPE, CR-FM-NES, jSO | | 2.5, 2.0, 2.7 times | |

## Findings and changes

1. **"Nearly equal sums" means equal to rounding.** The coordinate index
   of `ParetoArchiveND` engages when all objective sums of the front agree
   within 128 units in the last place, about 3e-14 relative. Points on an
   exact plane qualify. Sums that differ by 1e-9 do not, and get no gain.
   The report's fixture is of the first kind; its label and the changelog
   suggested more. The changelog now states the condition.
2. **The delivered archive was 10% slower on other large fronts.** The
   patch put three kernels behind one `dominates` function, which the
   general scan calls for every pair; choosing the kernel per pair cost
   8% to 12% on fronts of 4,096 mutually nondominated points with three to
   six objectives (table above). The scan now chooses its kernel once and
   the compiler inlines it. Measured alternatives led to the branch-free
   kernel from five objectives on (it lost at four on mostly dominated
   streams) and to keeping the unrolled three-objective kernel, which is no
   faster in a scan but helps the single witness test. Results are
   unchanged.
3. **No test covered the kernels by width.** Added: admission masks
   against strict dominance by definition for 3, 4, 5, 8 and 12 objectives
   on five kinds of stream, including the switch to the coordinate index
   and back.
4. **Three behavior changes were missing from the release notes.** Dual
   Annealing with local search made up to 21 times the requested calls and
   now stops at the budget; CR-FM-NES and PGPE stop up to a population
   short of what they did; `success` follows the stop code. All three are
   now under "Changed behavior to expect".
5. **A failing matrix objective is silent in CR-FM-NES and PGPE.** An
   exception or a wrong result shape ends the run with `status=-1`,
   `nfev=0` and no message, while jSO raises. This follows the wrappers'
   existing handling of scalar objectives and a delivered test pins it, so
   it was documented and not changed.

## Not verified here

- The sanitizer and Clang runs, the allocation counts and the 30-pair
  timing table of the report.
- The 158 trials outside the patch.
- Worker counts above one for the four optimizers.

## Open points

- Whether a matrix objective that raises should raise in every wrapper.
  Today jSO raises, MODE prints the error and returns, and CMA-ES,
  CR-FM-NES and PGPE return `status=-1`.
- The coordinate index could cover sums within a relative tolerance such
  as 1e-9 by widening its band. The argument in the report holds for any
  band width; where the index stops paying was not measured.
