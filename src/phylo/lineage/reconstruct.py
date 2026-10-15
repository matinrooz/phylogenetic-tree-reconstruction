"""Reconstructing cell lineage trees from simulated target-site data.

We apply the general-purpose algorithms from ``phylo.methods`` to the
lineage character matrix, plus a random-tree baseline:

* parsimony - Fitch parsimony with the same heuristic search used for DNA
  (random addition + NNI + SPR, plus the NJ tree as one extra start).
  Each target site is an unordered multi-state character and missing values
  are "any state". This ignores two things we know about the data: that
  every site starts unedited, and that edits cannot be undone. A
  Camin-Sokal style (irreversible) parsimony would use that information.
* nj / upgma - on the lineage distance below.
* random - a uniformly random binary topology, to show what "no signal"
  looks like on the Robinson-Foulds scale.

Lineage distance
----------------
For two cells, look only at sites observed in both:

* same state (both unedited, or the same indel)           -> 0
* one unedited, the other edited                           -> 1
* both edited, but with different indels                   -> 2

and average over those sites. Two different edits at the same site mean
the cells diverged before either edit happened, so they count as further
apart than "one edited, one not". This is a simple heuristic weighting,
not a model-based distance. Pairs with no site in common get the maximum
value (2) and are counted, so the caller can see when that happens.

Likelihood under JC69 is not used here: that model describes four
nucleotides that can change back and forth, which is the wrong model for
irreversible edits with many possible outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..distance import DistanceMatrix
from ..methods.nj import neighbor_joining
from ..methods.parsimony import CharacterData, fitch_score, parsimony_search
from ..methods.upgma import upgma
from ..tree import Node, Tree
from .simulator import LineageData, MISSING, UNEDITED

LINEAGE_METHODS = ("parsimony", "nj", "upgma", "random")


def lineage_distance(names, states: np.ndarray) -> DistanceMatrix:
    states = np.asarray(states)
    n = states.shape[0]
    values = np.zeros((n, n))
    n_empty = 0
    observed = states != MISSING
    for i in range(n):
        for j in range(i + 1, n):
            both = observed[i] & observed[j]
            if not both.any():
                values[i, j] = values[j, i] = 2.0
                n_empty += 1
                continue
            a, b = states[i, both], states[j, both]
            differ = a != b
            one_unedited = (a == UNEDITED) | (b == UNEDITED)
            score = np.where(~differ, 0.0, np.where(one_unedited, 1.0, 2.0))
            values[i, j] = values[j, i] = score.mean()
    return DistanceMatrix(list(names), values, n_saturated=n_empty)


def random_tree(names, rng: np.random.Generator) -> Tree:
    """Uniformly random unrooted binary topology (rooted arbitrarily)."""
    from ..tree import insert_leaf, unique_insertion_branches

    order = list(rng.permutation(list(names)))
    root = Node(None)
    root.add_child(Node(order[0], 1.0))
    inner = root.add_child(Node(None, 1.0))
    inner.add_child(Node(order[1], 1.0))
    inner.add_child(Node(order[2], 1.0))
    tree = Tree(root)
    for name in order[3:]:
        branches = unique_insertion_branches(tree)
        insert_leaf(branches[int(rng.integers(len(branches)))], Node(name, 1.0))
    return tree


@dataclass
class LineageInference:
    method: str
    tree: Tree
    parsimony_score: float


def reconstruct(data: LineageData, method: str, seed: int = 0,
                parsimony_starts: int = 3) -> LineageInference:
    chars = CharacterData.from_states(data.names, data.states, n_states=int(data.outcome_probs.size) + 1)
    if method == "parsimony":
        nj_start = neighbor_joining(lineage_distance(data.names, data.states))
        res = parsimony_search(chars, n_starts=parsimony_starts, seed=seed, start_trees=[nj_start])
        tree = res.tree
    elif method == "nj":
        tree = neighbor_joining(lineage_distance(data.names, data.states))
    elif method == "upgma":
        tree = upgma(lineage_distance(data.names, data.states))
    elif method == "random":
        tree = random_tree(data.names, np.random.default_rng(seed))
    else:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(LINEAGE_METHODS)}")
    return LineageInference(method, tree, fitch_score(tree, chars))
