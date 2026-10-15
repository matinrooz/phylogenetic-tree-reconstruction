# CRISPR lineage-tracing extension

`src/phylo/lineage/` is a separate, experimental module. It asks a narrow
question: **how well do standard phylogenetic algorithms recover a cell
lineage tree from CRISPR target-site data, on a deliberately simple
simulator where the true tree is known?**

## Background

In CRISPR lineage tracing, cells carry an array of target sites. Cas9
creates heritable insertions or deletions (indels) at those sites while the
cells divide, so related cells share edits. Reading the sites out at the end
gives a character matrix: rows are cells, columns are sites, values are
"unedited", a particular indel, or missing. Reconstructing the cell tree
from that matrix is a phylogenetics problem, but not quite the textbook one:

- characters start in a known state (unedited) and change at most once;
- there are many possible outcomes per site, with very unequal frequencies;
- a lot of data is missing, often for technical reasons.

## The simulator (`simulator.py`)

| Parameter | Meaning (base value in the benchmark) |
|---|---|
| `n_generations` | synchronous divisions; 2^g cells at the end (7 → 128 cells) |
| `n_cells` | cells sampled for the experiment (32) |
| `n_sites` | target sites per cell (20) |
| `edit_rate` | probability that an unedited site is edited in one generation (0.05) |
| `n_outcomes` | possible indel alleles per site (30) |
| `outcome_skew` | power-law exponent for allele probabilities; 0 = all equally likely (1.5) |
| `dropout_rate` | probability that an observed value is missing (0.1) |

Edits are irreversible. The true tree is the sampled cells' part of the
complete division tree, with unary nodes removed.

### Assumptions and what is left out

The simulator is much simpler than a real experiment. It does **not**
include:

- large deletions that remove several neighbouring sites at once. In real
  data these cause correlated, heritable dropout and can remove earlier
  edits;
- site-specific or time-varying editing rates (editing often slows down as
  Cas9 or guide expression changes);
- asynchronous divisions, cell death, or differences in growth between
  lineages;
- sequencing errors, doublets, or uncertainty in allele calling;
- indel probabilities that have to be estimated from the data. Here they
  are known and fixed.

The results show how the algorithms behave on the basic problem. They do
not tell you how accurate any method would be on a real dataset.

## Reconstruction (`reconstruct.py`)

| Method | What it does | What it ignores |
|---|---|---|
| `parsimony` | Fitch parsimony (multi-state, missing = any state) + the same random-addition/NNI search as for DNA | that edits are irreversible and start from "unedited" (Camin–Sokal-style parsimony would use this) |
| `nj`, `upgma` | on a lineage distance: per site 0 if equal, 1 if one cell is unedited, 2 if both are edited differently; averaged over sites observed in both cells | allele frequencies; it is a heuristic weighting, not a model-based distance |
| `random` | a uniformly random binary tree | everything. It shows what "no information" looks like on the RF scale |

JC69 likelihood is deliberately **not** used here. Its assumptions (four
states, reversible changes) do not fit irreversible edits with dozens of
outcomes. A proper likelihood would need a model of the editing process.

## Things to keep in mind when reading the results

- **Unresolvable splits.** If no site was edited on a branch of the true
  tree, no method can recover that split from the data. Part of the RF
  error at low editing rates is therefore unavoidable, and the RF distance
  does not separate "wrong" from "impossible to know".
- **Saturation.** At high editing rates most sites are edited in the first
  few generations, which leaves little signal for the later divisions.
- **Homoplasy.** With a skewed allele distribution, the most common indels
  appear independently in unrelated cells and mislead every method.
- **UPGMA's clock assumption** is actually satisfied by this simulator
  (synchronous divisions, every leaf at the same depth), so UPGMA does
  better here than it would on asynchronous real data.

## Possible extensions

- Camin–Sokal (irreversible) parsimony, or Dollo-style constraints.
- A likelihood model of the editing process with allele frequencies.
- Heritable large-deletion dropout in the simulator.
- Measuring accuracy only on the splits that are resolvable in principle,
  for example triplet accuracy split by depth.
