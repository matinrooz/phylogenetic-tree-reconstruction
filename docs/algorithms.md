# How the algorithms work

This is the longer version of the method descriptions in the README. Each
section points to the file with the implementation. The docstrings at the
top of those files say the same things next to the code.

## Tree representation (`src/phylo/tree.py`)

A tree is a set of `Node` objects with `children`, a `parent` pointer, an
optional `name` and `length` (the branch to the parent). Every method
returns a rooted binary tree.

Most of the methods actually infer **unrooted** trees (NJ, parsimony, ML
under a reversible model). For those the root is just a storage detail: NJ
puts it on the last branch it creates, and the searches keep it wherever it
ended up. Because of that:

- tree comparison uses **splits** (bipartitions of the leaf set), which do
  not depend on the root;
- the two branches below the root count as one unrooted branch. That
  matters for NNI moves and for stepwise addition, and both handle it
  explicitly (`nni_moves`, `unique_insertion_branches`).

Newick input and output (`newick.py`) supports quoted labels, comments in
square brackets, internal labels and scientific notation, and a round trip
keeps topology and branch lengths.

## Distances (`src/phylo/distance.py`)

For each pair of sequences, only sites where both have an unambiguous base
are compared. The p-distance is the fraction of those sites that differ. The
Jukes–Cantor correction

    d = -3/4 ln(1 - 4p/3)

accounts for multiple substitutions at the same site, and it is the
maximum-likelihood distance under JC69. If p ≥ 3/4 the logarithm is
undefined, so the distance is capped (default 5 substitutions/site) and the
number of capped pairs is reported. Capped distances carry little
information, which is part of why distance methods degrade on very
divergent data.

## UPGMA (`src/phylo/methods/upgma.py`)

Agglomerative clustering with average linkage: join the closest pair of
clusters, put the new node at half their distance, and update the distances
as size-weighted averages. The result is ultrametric. UPGMA is consistent
only if the data follow a molecular clock. O(n³).

## Neighbor-joining (`src/phylo/methods/nj.py`)

At each step NJ joins the pair minimising

    Q(i,j) = (r-2) d(i,j) - R_i - R_j

which corrects for the fact that a taxon on a long branch is far from
everything. Branch lengths come from the standard formulas. NJ is
consistent for additive distances and does not need a clock. Negative
branch estimates are set to zero by default (`allow_negative=True` keeps
them, which matches ape). O(n³).

## Maximum parsimony (`src/phylo/methods/parsimony.py`)

**Scoring.** Fitch's algorithm goes once from the leaves to the root.
Each node gets the intersection of its children's state sets if they
overlap, otherwise their union plus one change. State sets are bitmasks, so
all sites are processed at once with NumPy, and identical alignment columns
are compressed into weighted patterns first. Missing data is the full set.
The tests compare Fitch against brute-force enumeration of every
ancestral-state assignment on small trees.

**Search.** Finding the best tree is NP-hard. The search is:

1. random-order stepwise addition (each taxon goes on the branch where it
   adds the fewest changes, with ties broken at random);
2. NNI hill climbing with best improvement;
3. SPR hill climbing (cut every subtree, try it on every other branch,
   take the best improvement, repeat);
4. repeated from several random orders, plus the NJ tree as an extra start
   when called through `infer_tree` or the lineage module, keeping the best.

The first version stopped after step 2. Checking the benchmark output showed
that on the lineage-tracing data (few site patterns, many ties) NNI alone
ended worse than the NJ tree in 40 of 210 datasets. SPR fixed 35 of them,
and with the NJ start the search can no longer end worse than that tree. On
the DNA examples I timed, SPR found no better trees but cost about 6x more
time. I kept it on by default anyway (`use_spr=False` turns it off).

For ≤ 8 taxa `exhaustive_parsimony` enumerates all (2n−5)!! unrooted trees.
The tests use it to check that the heuristic reaches the exact optimum on
small problems. It did in 12 of 12 simulated 8-taxon datasets; the test
allows one miss, since a local search is not guaranteed to succeed.

## Maximum likelihood (`src/phylo/methods/likelihood.py`)

**Model.** JC69: equal base frequencies and one substitution rate. The
transition probability for a branch of length t is
P(same) = 1/4 + 3/4·e^(−4t/3), P(each other base) = 1/4 − 1/4·e^(−4t/3).

**Pruning.** Felsenstein's algorithm computes, for every node and base, the
probability of the data below it. Tip vectors are 1 for observed bases
(all four for missing or ambiguous data). The site likelihood is the
average of the root vector over the four bases. Per-node rescaling with
stored log factors prevents underflow. The tests check it against a
pure log-space implementation on a 300-taxon tree and against brute-force
summation over internal states on small trees.

**Branch lengths.** For each branch we also keep the likelihood vector of
the rest of the tree, at the branch's upper end ("outside" vectors). The
likelihood as a function of that one branch length is then cheap to
evaluate, and Brent's method finds its maximum on [1e-8, 10]. Repeating
over all branches until the log-likelihood stops improving (< 1e-5 per
pass) gives a local optimum. A useful check (also a test): for two
sequences, the ML branch length equals the JC69 distance.

**Search.** NNI hill climbing, scoring each candidate after re-optimising
only the branch the move acts on. It starts from the NJ, UPGMA and a quick
parsimony tree (2 random-addition starts) and keeps the best result. In
the benchmark this ended in a worse local optimum than another method's tree
in 1 of 200 datasets (by 0.25 log-likelihood units). This is a local search, not a guarantee of
the ML tree. See `crosscheck/README.md` for how it compares with phangorn.

## Robinson–Foulds distance (`src/phylo/metrics.py`)

The number of splits found in one tree but not the other. Normalising by
2(n−3) gives 0 for identical topologies and 1 for trees with no internal
split in common. It ignores branch lengths and the root. It is also a
fairly harsh measure: moving one leaf a long way across the tree can change
many splits at once.

## Simulation (`src/phylo/simulate.py`)

- Yule (pure birth) trees, rescaled so the root-to-tip depth equals
  `tree_height` substitutions per site.
- Optional relaxed clock: each branch is multiplied by
  exp(N(−σ²/2, σ)), so its mean factor is 1.
- JC69 sequence evolution from a uniform random root sequence.

Every simulated dataset is synthetic and labelled as such in the outputs.
