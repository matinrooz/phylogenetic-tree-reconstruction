"""Maximum parsimony: Fitch scoring and a heuristic tree search.

Fitch's algorithm (1971)
------------------------
For one site on a binary tree, go from the leaves to the root. A leaf's set
is its observed state(s). For an internal node with child sets S1 and S2:

* if S1 and S2 intersect, the node's set is the intersection (no change
  needed here);
* otherwise it is the union, and one change is counted.

The number of counted changes is the minimum number of state changes the
tree needs to explain that site. The total score is the sum over sites.

States are stored as bitmasks, so every site of the alignment is processed
at once with NumPy operations. Missing data is the full set, which can
never force a change. The same code works for DNA (4 states) and for the
multi-state characters of the lineage-tracing module (up to 64 states).

Fitch scores unordered, reversible characters on a binary tree. The score
does not depend on where the tree is rooted, so trees are treated as
unrooted here.

Tree search
-----------
Finding the most parsimonious tree is NP-hard, so for anything but tiny
data we use a heuristic, the same basic strategy as most programs:

1. Random stepwise addition: start with three taxa, then add the remaining
   taxa one at a time, in random order, on the branch where they increase
   the score the least.
2. NNI hill climbing: try every nearest-neighbour interchange, take the
   best improvement, and repeat until no move lowers the score.
3. SPR hill climbing: cut off every subtree and try it on every other
   branch, again taking the best improvement until nothing helps. NNI
   first is cheaper; SPR then escapes the local optima NNI gets stuck in.
4. Do this from several random addition orders and keep the best tree.

This finds a local optimum, not a guaranteed global one. For up to eight
taxa ``exhaustive_parsimony`` scores every possible unrooted tree, which the
tests use to check how often the heuristic reaches the true optimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..alignment import Alignment
from ..tree import (Node, Tree, TreeError, insert_leaf, nni_moves,
                    remove_inserted_leaf, swap_subtrees, unique_insertion_branches)


# ---------------------------------------------------------------------- #
# character data
# ---------------------------------------------------------------------- #
@dataclass
class CharacterData:
    """Per-taxon state sets (bitmasks) for compressed site patterns."""

    names: List[str]
    masks: Dict[str, np.ndarray]  # name -> uint64 array, one entry per pattern
    weights: np.ndarray            # how often each pattern occurs

    @classmethod
    def from_alignment(cls, alignment: Alignment) -> "CharacterData":
        patterns, counts = alignment.site_patterns()
        masks = {name: patterns[i].astype(np.uint64) for i, name in enumerate(alignment.names)}
        return cls(list(alignment.names), masks, counts)

    @classmethod
    def from_states(cls, names: Sequence[str], states: np.ndarray, n_states: Optional[int] = None) -> "CharacterData":
        """Integer states 0..k-1 per taxon and site; negative means missing."""
        states = np.asarray(states, dtype=np.int64)
        if states.ndim != 2 or states.shape[0] != len(names):
            raise ValueError("states must be a (taxa x sites) matrix")
        k = int(states.max()) + 1 if n_states is None else n_states
        if k > 64:
            raise ValueError("Fitch bitmasks support at most 64 states per character")
        full = np.uint64((1 << k) - 1) if k < 64 else np.uint64(np.iinfo(np.uint64).max)
        one = np.uint64(1)
        safe = np.clip(states, 0, None).astype(np.uint64)
        masks_all = np.where(states < 0, full, np.left_shift(one, safe))
        # compress identical columns
        patterns, counts = np.unique(masks_all, axis=1, return_counts=True)
        masks = {name: patterns[i] for i, name in enumerate(names)}
        return cls(list(names), masks, counts.astype(float))


# ---------------------------------------------------------------------- #
# scoring
# ---------------------------------------------------------------------- #
def fitch_score(tree: Tree, data: CharacterData) -> float:
    """Parsimony score of a tree (minimum number of changes)."""
    sets: Dict[int, np.ndarray] = {}
    score = 0.0
    weights = data.weights
    for node in tree.postorder():
        if node.is_leaf:
            try:
                sets[id(node)] = data.masks[str(node.name)]
            except KeyError:
                raise TreeError(f"leaf {node.name!r} is not in the data") from None
            continue
        children = node.children
        current = sets.pop(id(children[0]))
        # multifurcations are resolved by combining children one by one;
        # that scores one particular binary resolution (an upper bound)
        for child in children[1:]:
            other = sets.pop(id(child))
            inter = current & other
            empty = inter == 0
            if empty.any():
                score += float(weights[empty].sum())
                current = np.where(empty, current | other, inter)
            else:
                current = inter
        sets[id(node)] = current
    return score


def site_scores(tree: Tree, data: CharacterData) -> np.ndarray:
    """Changes per *pattern* (not weighted). Mostly useful for tests."""
    sets: Dict[int, np.ndarray] = {}
    changes = np.zeros(len(data.weights))
    for node in tree.postorder():
        if node.is_leaf:
            sets[id(node)] = data.masks[str(node.name)]
            continue
        current = sets.pop(id(node.children[0]))
        for child in node.children[1:]:
            other = sets.pop(id(child))
            inter = current & other
            empty = inter == 0
            changes += empty
            current = np.where(empty, current | other, inter)
        sets[id(node)] = current
    return changes


# ---------------------------------------------------------------------- #
# search
# ---------------------------------------------------------------------- #
@dataclass
class ParsimonyResult:
    tree: Tree
    score: float
    n_starts: int
    start_scores: List[float] = field(default_factory=list)
    n_nni_rounds: int = 0
    seconds: float = 0.0


def _three_taxon_tree(names: Sequence[str]) -> Tree:
    root = Node(None)
    root.add_child(Node(names[0], 1.0))
    inner = root.add_child(Node(None, 1.0))
    inner.add_child(Node(names[1], 1.0))
    inner.add_child(Node(names[2], 1.0))
    return Tree(root)


def stepwise_addition(data: CharacterData, order: Sequence[str],
                      rng: Optional[np.random.Generator] = None) -> Tree:
    """Greedy stepwise addition in the given taxon order."""
    if len(order) < 3:
        raise ValueError("parsimony search needs at least three taxa")
    tree = _three_taxon_tree(order)
    for name in order[3:]:
        leaf = Node(name, 1.0)
        best_score = np.inf
        best: List[Node] = []
        for branch in unique_insertion_branches(tree):
            middle = insert_leaf(branch, leaf)
            score = fitch_score(tree, data)
            remove_inserted_leaf(middle)
            if score < best_score - 1e-9:
                best_score, best = score, [branch]
            elif abs(score - best_score) <= 1e-9:
                best.append(branch)
        choice = best[int(rng.integers(len(best)))] if rng is not None else best[0]
        insert_leaf(choice, leaf)
    return tree


def nni_hill_climb(tree: Tree, data: CharacterData, max_rounds: int = 1000) -> tuple:
    """Apply best-improvement NNI moves until none lowers the score.

    Works in place and returns (score, number of accepted moves).
    """
    score = fitch_score(tree, data)
    rounds = 0
    while rounds < max_rounds:
        nodes = list(tree.preorder())
        best_move = None
        best_score = score
        for i, j in nni_moves(tree):
            x, y = nodes[i], nodes[j]
            swap_subtrees(x, y)
            s = fitch_score(tree, data)
            swap_subtrees(x, y)  # undo
            if s < best_score - 1e-9:
                best_score, best_move = s, (x, y)
        if best_move is None:
            break
        swap_subtrees(*best_move)
        score = best_score
        rounds += 1
    return score, rounds


def _prune(tree: Tree, subtree: Node) -> Tree:
    """Detach ``subtree`` and suppress the node it hung from (in place).

    Returns the remaining tree (its root may change if the subtree was a
    child of the root).
    """
    parent = subtree.parent
    assert parent is not None and len(parent.children) == 2
    parent.remove_child(subtree)
    sister = parent.children[0]
    grand = parent.parent
    parent.remove_child(sister)
    if grand is None:
        sister.length = 0.0
        return Tree(sister)
    idx = grand.children.index(parent)
    grand.children[idx] = sister
    sister.parent = grand
    sister.length += parent.length
    parent.parent = None
    return tree


def spr_hill_climb(tree: Tree, data: CharacterData, max_rounds: int = 1000) -> tuple:
    """Best-improvement SPR (subtree pruning and regrafting) hill climbing.

    Every subtree is cut off and re-attached on every branch of the rest of
    the tree; the best strictly improving move is applied and the process
    repeats. SPR includes all NNI moves and many more, so it gets out of
    local optima where NNI is stuck. Each round costs O(n^2) Fitch scorings.
    Returns (tree, score, number of accepted moves); the tree object can
    change because pruning may move the root.
    """
    score = fitch_score(tree, data)
    rounds = 0
    while rounds < max_rounds:
        best_score = score
        best_tree: Optional[Tree] = None
        n_nodes = sum(1 for _ in tree.preorder())
        for k in range(1, n_nodes):  # 0 is the root
            work = tree.copy()
            node = list(work.preorder())[k]
            remaining = _prune(work, node)
            if remaining.n_leaves < 2:
                continue
            for branch in unique_insertion_branches(remaining):
                middle = insert_leaf(branch, node)
                s = fitch_score(remaining, data)
                if s < best_score - 1e-9:
                    best_score, best_tree = s, remaining.copy()
                remove_inserted_leaf(middle)
        if best_tree is None:
            break
        tree, score = best_tree, best_score
        rounds += 1
    return tree, score, rounds


def parsimony_search(data: CharacterData, n_starts: int = 10, seed: Optional[int] = 0,
                     use_spr: bool = True, start_trees: Sequence[Tree] = ()) -> ParsimonyResult:
    """Heuristic maximum-parsimony search.

    From each of ``n_starts`` random addition orders (and from any trees in
    ``start_trees``, e.g. a neighbor-joining tree): NNI hill climbing, then
    by default SPR hill climbing. The best tree over all starts is returned.

    History: the first version used NNI only. Checking the benchmark output
    showed that on the lineage-tracing data (few site patterns, many ties)
    it ended 1-2 steps worse than the NJ tree in 40 of 210 datasets. SPR
    fixed 35 of those; allowing the NJ tree as an extra start means the
    search can no longer end worse than that tree.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    names = list(data.names)
    if len(names) < 3:
        raise ValueError("parsimony search needs at least three taxa")
    best_tree: Optional[Tree] = None
    best_score = np.inf
    start_scores: List[float] = []
    total_rounds = 0

    starts: List[Tree] = []
    for tree in start_trees:
        if sorted(tree.leaf_names()) != sorted(names):
            raise TreeError("start tree has different taxa from the data")
        if not tree.is_binary():
            raise TreeError("start trees must be binary")
        starts.append(tree.copy())
    for start in range(max(1, n_starts)):
        order = names if start == 0 else list(rng.permutation(names))
        starts.append(stepwise_addition(data, order, rng))

    for tree in starts:
        score, rounds = nni_hill_climb(tree, data)
        total_rounds += rounds
        if use_spr and len(names) >= 5:
            tree, score, rounds = spr_hill_climb(tree, data)
            total_rounds += rounds
        start_scores.append(score)
        if score < best_score:
            best_score, best_tree = score, tree
    assert best_tree is not None
    return ParsimonyResult(best_tree, best_score, len(starts), start_scores,
                           total_rounds, time.perf_counter() - t0)


def all_unrooted_trees(names: Sequence[str]):
    """Yield every unrooted binary topology on ``names`` (as rooted trees).

    There are (2n - 5)!! of them, so this is only practical for small n.
    """
    names = list(names)
    if len(names) < 3:
        raise ValueError("need at least three taxa")

    def extend(tree: Tree, k: int):
        if k == len(names):
            yield tree
            return
        leaf = Node(names[k], 1.0)
        for branch in unique_insertion_branches(tree):
            middle = insert_leaf(branch, leaf)
            yield from extend(tree, k + 1)
            remove_inserted_leaf(middle)

    yield from extend(_three_taxon_tree(names), 3)


def exhaustive_parsimony(data: CharacterData, max_taxa: int = 8) -> ParsimonyResult:
    """Score every topology and return a most parsimonious tree (exact)."""
    if len(data.names) > max_taxa:
        raise ValueError(
            f"exhaustive search over {len(data.names)} taxa is too slow; "
            f"the limit is {max_taxa} (use parsimony_search instead)"
        )
    t0 = time.perf_counter()
    best_score = np.inf
    best_tree: Optional[Tree] = None
    for tree in all_unrooted_trees(data.names):
        s = fitch_score(tree, data)
        if s < best_score:
            best_score, best_tree = s, tree.copy()
    assert best_tree is not None
    return ParsimonyResult(best_tree, best_score, 0, [], 0, time.perf_counter() - t0)
