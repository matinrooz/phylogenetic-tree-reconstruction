# Cross-check against ape / phangorn

During development I compared this package against the R packages
[ape](https://cran.r-project.org/package=ape) (5.x) and
[phangorn](https://cran.r-project.org/package=phangorn), running R 4.2.2.
R is **not** needed to use the package. This folder is only here so the
check can be repeated.

```bash
# needs Rscript plus install.packages("phangorn") (which pulls in ape)
python crosscheck/run_crosscheck.py            # or: --rlib /path/to/R/library
```

The script simulates five alignments (8–24 taxa, 150–1000 sites), runs
every Python method, scores the same trees in R, runs phangorn's own searches,
and writes every comparison to `crosscheck_results.csv`.

## What matched

| Check | Result |
|---|---|
| JC69 distance matrix vs `ape::dist.dna(model = "JC69", pairwise.deletion = TRUE)` | identical (max difference ~5e-15) |
| UPGMA vs `phangorn::upgma` | same topology, path lengths equal to ~1e-10 |
| NJ vs `ape::nj` | same topology on all 5 datasets; see the first point below for branch lengths |
| Fitch score of the same tree vs `phangorn::parsimony(method = "fitch")` | identical for all 25 trees |
| JC69 log-likelihood of the same tree with fixed branch lengths vs `phangorn::pml` | equal to ~1e-10 |
| log-likelihood after optimising branch lengths on a fixed topology vs `optim.pml(optEdge = TRUE)` | within 0.001 log-likelihood units (ours is never higher by more than ~1e-5) |
| our parsimony search vs `phangorn::pratchet` | same best score on all 5 datasets; on one dataset a different, equally parsimonious tree |
| our ML search vs `optim.pml(optNni = TRUE)` from an NJ start | same topology on all 5 datasets, log-likelihood within 0.001 |

## Discrepancies found and how they were resolved

1. **NJ branch lengths.** Our NJ originally differed from `ape::nj` by up to
   0.0074 in path lengths on three datasets. The cause: ape keeps negative
   branch estimates, while we set them to zero and add the difference to the
   sister branch. Topologies were identical. I kept zero-clamping as the
   default (negative lengths have no meaning and break the likelihood code)
   and added `neighbor_joining(dm, allow_negative=True)`, which matches ape
   to ~1e-10 on all datasets. Both behaviours are tested.

2. **Branch-length optimisation stopped early.** With a fixed cap of 8
   coordinate-ascent passes, our optimised log-likelihood was up to 0.006
   below phangorn's on a 24-taxon tree. Coordinate ascent converges slowly
   when branches are correlated, so the optimiser now runs until a full pass
   gains less than 1e-5 (at most 100 passes). The remaining differences are
   below 0.001 log-likelihood units.

3. **ML search stuck in a local optimum.** On one 16-taxon dataset our NNI
   search from the NJ tree ended at log-likelihood −6800.78, while phangorn
   reached −6800.56 (a tree 4 RF steps away). Starting from the UPGMA tree
   our search reached the same tree as phangorn. The search now starts from
   the NJ, UPGMA and parsimony trees and keeps the best, which is standard
   practice. After the change our search and phangorn agree on all five
   datasets. That is still a heuristic: neither program guarantees the
   global optimum.

The cross-check was run again after the parsimony search gained SPR moves
(a change made after checking the benchmark output; see `docs/algorithms.md`).
All values in the table above were unchanged. The benchmark results in the
main README were produced after all of these fixes.
