"""Neighbor-joining (Saitou and Nei, 1987).

Algorithm
---------
With r active nodes and distances d, compute for every pair

    Q(i, j) = (r - 2) d(i, j) - R_i - R_j,   where R_i = sum_k d(i, k)

and join the pair with the smallest Q. The new node u gets branch lengths

    d(i, u) = d(i, j) / 2 + (R_i - R_j) / (2 (r - 2))
    d(j, u) = d(i, j) - d(i, u)

and distances d(u, k) = (d(i, k) + d(j, k) - d(i, j)) / 2 to the others.
This repeats until two nodes remain, which are connected directly.

If the input distances are additive (they fit some tree exactly), NJ
recovers that tree and its branch lengths. With real data the estimated
branch lengths can come out negative. By default we set those to zero and
move the difference to the sister branch so the path length between the
pair is kept. That is a common convention, but it means our lengths can
differ slightly from programs that keep negative values (ape's ``nj`` does);
``allow_negative=True`` reproduces that behaviour. The topology is the same
either way.

NJ produces an unrooted tree. We store it rooted on the last branch that
was created, with the length split equally; that root has no biological
meaning. Runtime is O(n^3).
"""

from __future__ import annotations

import numpy as np

from ..distance import DistanceMatrix
from ..tree import Node, Tree


def _nonnegative(a: float, b: float) -> tuple:
    """Clamp a negative length to zero and give the excess to its sister."""
    if a < 0:
        return 0.0, b + a
    if b < 0:
        return a + b, 0.0
    return a, b


def neighbor_joining(dm: DistanceMatrix, allow_negative: bool = False) -> Tree:
    n = len(dm.names)
    if n < 3:
        raise ValueError("neighbor-joining needs at least three taxa")
    d = dm.values.astype(float).copy()
    nodes = [Node(name) for name in dm.names]
    active = list(range(n))

    while len(active) > 2:
        r = len(active)
        sub = d[np.ix_(active, active)]
        row_sums = sub.sum(axis=1)
        q = (r - 2) * sub - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(q, np.inf)
        flat = int(np.argmin(q))
        a, b = divmod(flat, r)
        a, b = min(a, b), max(a, b)
        i, j = active[a], active[b]

        dij = d[i, j]
        li = dij / 2.0 + (row_sums[a] - row_sums[b]) / (2.0 * (r - 2))
        lj = dij - li
        if not allow_negative:
            li, lj = _nonnegative(li, lj)

        u = Node(None)
        nodes[i].length = li
        nodes[j].length = lj
        u.add_child(nodes[i])
        u.add_child(nodes[j])

        new_dist = (d[i] + d[j] - dij) / 2.0
        d[i, :] = new_dist
        d[:, i] = new_dist
        d[i, i] = 0.0
        nodes[i] = u
        active.remove(j)

    i, j = active
    last = d[i, j] if allow_negative else max(d[i, j], 0.0)
    root = Node(None)
    nodes[i].length = last / 2.0
    nodes[j].length = last / 2.0
    root.add_child(nodes[i])
    root.add_child(nodes[j])
    return Tree(root)
