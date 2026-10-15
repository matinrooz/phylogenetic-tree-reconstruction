# Example data

Everything in this folder is small and either made up by hand or simulated.
None of it is real biological data, and nothing here should be read as a
biological result.

| File | What it is | How it was made |
|---|---|---|
| `hand_example.fasta` | 4 taxa, 4 sites | Written by hand so the parsimony scores can be checked on paper: the tree `((A,B),(C,D))` needs 4 changes, `((A,C),(B,D))` needs 6. Used in the tests and the README. |
| `missing_data.fasta` | 5 taxa, 20 sites with gaps, `N`, `?` and one ambiguity code (`R`) | Written by hand to exercise missing-data handling. |
| `simulated_12taxa.fasta` | 12 taxa, 600 sites | `phylo simulate --taxa 12 --sites 600 --height 0.3 --clock-sigma 0.5 --seed 42 ...` (Yule tree, JC69, relaxed clock) |
| `simulated_12taxa_true.nwk` | the tree the alignment above was simulated on | same command |

## Why there is no real dataset

I only wanted to include public data whose source and licence I could check
and cite properly, and the benchmarks need a known true tree anyway, which
real data does not have. The code reads any DNA alignment in FASTA format, so
running it on your own data is one command:

    phylo infer your_alignment.fasta --method ml --out tree.nwk --plot tree.png

The input must already be aligned (all sequences the same length).
