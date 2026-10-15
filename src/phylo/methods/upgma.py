"""UPGMA (Unweighted Pair Group Method with Arithmetic mean).

Algorithm
---------
Start with every taxon as its own cluster. Repeatedly:

1. find the two clusters i, j with the smallest distance d(i, j);
2. join them under a new node placed at height d(i, j) / 2;
3. the distance from the new cluster to every other cluster k is the
   average over all taxon pairs, which can be computed from the old values
   weighted by cluster size:  (n_i d(i,k) + n_j d(j,k)) / (n_i + n_j).

Branch lengths are the height differences between a node and its children,
so the result is ultrametric (all leaves equally far from the root).

UPGMA assumes a molecular clock. When rates differ between lineages it can
group taxa by similarity rather than by ancestry, which is exactly what the
benchmark's relaxed-clock condition is meant to show.

Ties are broken deterministically by taking the first pair in row-major
order, so the same input always gives the same tree. Runtime is O(n^3).
"""

from __future__ import annotations

import numpy as np

from ..distance import DistanceMatrix
from ..tree import Node, Tree


def upgma(dm: DistanceMatrix) -> Tree:
    n = len(dm.names)
    if n < 2:
        raise ValueError("UPGMA needs at least two taxa")
    d = dm.values.astype(float).copy()
    np.fill_diagonal(d, np.inf)
    clusters = [Node(name) for name in dm.names]
    heights = [0.0] * n
    sizes = [1] * n
    active = list(range(n))

    while len(active) > 1:
        sub = d[np.ix_(active, active)]
        flat = int(np.argmin(sub))
        a, b = divmod(flat, len(active))
        i, j = active[min(a, b)], active[max(a, b)]

        height = d[i, j] / 2.0
        parent = Node(None)
        for k in (i, j):
            child = clusters[k]
            # with non-negative input UPGMA never produces negative lengths,
            # but rounding can give tiny negatives
            child.length = max(height - heights[k], 0.0)
            parent.add_child(child)

        # reuse slot i for the merged cluster, drop slot j
        merged = (sizes[i] * d[i] + sizes[j] * d[j]) / (sizes[i] + sizes[j])
        d[i, :] = merged
        d[:, i] = merged
        d[i, i] = np.inf
        clusters[i] = parent
        heights[i] = height
        sizes[i] += sizes[j]
        active.remove(j)

    return Tree(clusters[active[0]])
