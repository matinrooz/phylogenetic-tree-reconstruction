"""Simulated trees and DNA alignments with a known true tree.

Everything here is synthetic. It exists so we can measure how well each
method recovers a tree we actually know, which is impossible with real data.

Trees come from a pure-birth (Yule) process: lineages split at a constant
rate, so the tree is ultrametric. ``tree_height`` rescales it so that the
expected number of substitutions per site from the root to any tip equals
that value. It plays the role of "mutation rate x time": small values give
few differences between sequences, large values push towards saturation.

The optional relaxed clock multiplies every branch by an independent
log-normal factor with mean 1. That breaks the molecular clock, which is
the assumption UPGMA relies on.

Sequences evolve under JC69 along the tree, starting from a uniformly random
root sequence.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .alignment import Alignment
from .tree import Node, Tree


def yule_tree(n_taxa: int, rng: np.random.Generator, tree_height: float = 0.5,
              prefix: str = "t") -> Tree:
    """Random ultrametric tree from a Yule process, scaled to ``tree_height``."""
    if n_taxa < 2:
        raise ValueError("need at least two taxa")
    if tree_height <= 0:
        raise ValueError("tree_height must be positive")
    root = Node(None)
    active: List[Node] = [root.add_child(Node(None)), root.add_child(Node(None))]
    while len(active) < n_taxa:
        wait = rng.exponential(1.0 / len(active))
        for node in active:
            node.length += wait
        k = int(rng.integers(len(active)))
        splitting = active.pop(k)
        active.append(splitting.add_child(Node(None)))
        active.append(splitting.add_child(Node(None)))
    wait = rng.exponential(1.0 / len(active))
    for node in active:
        node.length += wait
    # name leaves in a random order so names carry no information
    order = rng.permutation(len(active))
    for idx, node in zip(order, active):
        node.name = f"{prefix}{idx + 1}"
    tree = Tree(root)
    height = max(tree.root_to_tip().values())
    for node in tree.branches():
        node.length *= tree_height / height
    return tree


def relax_clock(tree: Tree, rng: np.random.Generator, sigma: float) -> Tree:
    """Copy of ``tree`` with log-normal rate variation across branches."""
    out = tree.copy()
    if sigma <= 0:
        return out
    for node in out.branches():
        node.length *= math.exp(rng.normal(-sigma ** 2 / 2.0, sigma))
    return out


def simulate_jc69(tree: Tree, n_sites: int, rng: np.random.Generator) -> Alignment:
    """Evolve sequences down ``tree`` under the Jukes-Cantor model."""
    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    states = {id(tree.root): rng.integers(0, 4, size=n_sites)}
    names: List[str] = []
    rows: List[np.ndarray] = []
    for node in tree.preorder():
        if node.parent is None:
            continue
        parent_seq = states[id(node.parent)]
        p_change = 0.75 * (1.0 - math.exp(-4.0 * node.length / 3.0))
        change = rng.random(n_sites) < p_change
        shift = rng.integers(1, 4, size=n_sites)  # move to one of the other 3 bases
        seq = np.where(change, (parent_seq + shift) % 4, parent_seq)
        states[id(node)] = seq
        if node.is_leaf:
            names.append(str(node.name))
            rows.append(seq)
    order = np.argsort(names)
    return Alignment.from_states([names[i] for i in order], np.array([rows[i] for i in order]))


def simulate_dataset(n_taxa: int, n_sites: int, tree_height: float, seed: int,
                     clock_sigma: float = 0.0, missing_fraction: float = 0.0):
    """Convenience wrapper: (true_tree, alignment) for one replicate.

    ``missing_fraction`` blanks out that fraction of cells at random, to
    test how methods cope with missing data.
    """
    rng = np.random.default_rng(seed)
    tree = yule_tree(n_taxa, rng, tree_height)
    tree = relax_clock(tree, rng, clock_sigma)
    aln = simulate_jc69(tree, n_sites, rng)
    if missing_fraction > 0:
        holes = rng.random(aln.masks.shape) < missing_fraction
        aln.masks[holes] = 15
    return tree, aln
