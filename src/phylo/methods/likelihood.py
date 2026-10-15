"""Maximum likelihood under the Jukes-Cantor (JC69) model.

Model
-----
JC69 assumes equal base frequencies (1/4) and the same rate for every
substitution. For a branch of length t (expected substitutions per site):

    P(same base)      = 1/4 + 3/4 exp(-4t/3)
    P(specific other) = 1/4 - 1/4 exp(-4t/3)

Sites evolve independently and with the same rate (no rate variation).

Felsenstein's pruning algorithm
-------------------------------
For each node v and base x, the partial likelihood L_v(x) is the
probability of the data below v given that v has base x. Leaves get 1 for
the observed base(s) and 0 otherwise (missing data: 1 everywhere). For an
internal node,

    L_v(x) = prod over children c of  sum_y P_xy(t_c) L_c(y),

and the site likelihood is sum_x 1/4 L_root(x). Because JC69 is reversible
the likelihood does not depend on where the tree is rooted.

Numerical stability
-------------------
Partial likelihoods shrink towards zero as the tree grows, and would
underflow for large trees. After computing each node we divide its vectors
by their per-site maximum and keep the logarithm of that factor. The log
factors are added back at the end, so the result is exact.

Branch lengths and topology search
----------------------------------
Branch lengths are optimised one at a time (Brent's method on a bounded
interval) while the rest of the tree is fixed. To do that cheaply we also
compute, for each branch, the likelihood vector of everything *outside* the
subtree below it. The likelihood as a function of a single branch length t
is then  sum_x O(x) sum_y P_xy(t) L(y), which costs O(sites) per evaluation.
A few passes over all branches converge to a local optimum.

The topology search runs best-improvement NNI hill climbing from several
starting trees (by default the neighbor-joining, UPGMA and parsimony
trees) and keeps the best result. Each NNI candidate is scored after
re-optimising only the branch it acts on (a common shortcut). Accepted moves
are followed by two branch-length passes, and the winning tree gets a final
optimisation until the log-likelihood stops changing. Like the parsimony
search, this finds a local optimum; it is not an exhaustive search. During
development a single NJ start sometimes stopped at a worse local optimum
than phangorn's search, which is why several starts are used.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from ..alignment import Alignment
from ..distance import distance_matrix
from ..tree import Node, Tree, TreeError, nni_moves, nni_pivot, swap_subtrees
from .nj import neighbor_joining

MIN_BRANCH = 1e-8
MAX_BRANCH = 10.0
_FREQ = np.full(4, 0.25)


def jc69_matrix(t: float) -> np.ndarray:
    """4x4 JC69 transition probability matrix for branch length t."""
    if t < 0:
        raise ValueError("branch length must be non-negative")
    e = math.exp(-4.0 * t / 3.0)
    same = 0.25 + 0.75 * e
    diff = 0.25 - 0.25 * e
    m = np.full((4, 4), diff)
    np.fill_diagonal(m, same)
    return m


@dataclass
class PatternData:
    """Tip vectors for compressed site patterns."""

    names: List[str]
    tips: Dict[str, np.ndarray]  # name -> (n_patterns, 4) array
    weights: np.ndarray

    @classmethod
    def from_alignment(cls, alignment: Alignment) -> "PatternData":
        patterns, counts = alignment.site_patterns()
        bits = np.array([1, 2, 4, 8], dtype=np.uint8)
        tips = {}
        for i, name in enumerate(alignment.names):
            tips[name] = ((patterns[i][:, None] & bits[None, :]) > 0).astype(float)
        return cls(list(alignment.names), tips, counts)


class LikelihoodEngine:
    """Computes and optimises the JC69 likelihood of one tree."""

    def __init__(self, tree: Tree, data: PatternData) -> None:
        self.tree = tree
        self.data = data
        missing = set(tree.leaf_names()) ^ set(data.names)
        if missing:
            raise TreeError(f"tree and alignment have different taxa: {sorted(missing)[:5]}")
        self.down: Dict[int, np.ndarray] = {}
        self.down_scale: Dict[int, np.ndarray] = {}
        self.outside: Dict[int, np.ndarray] = {}
        self.outside_scale: Dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- #
    def _compute_down(self) -> None:
        n_pat = len(self.data.weights)
        self.down.clear()
        self.down_scale.clear()
        for node in self.tree.postorder():
            if node.is_leaf:
                self.down[id(node)] = self.data.tips[str(node.name)]
                self.down_scale[id(node)] = np.zeros(n_pat)
                continue
            vec = np.ones((n_pat, 4))
            scale = np.zeros(n_pat)
            for child in node.children:
                vec = vec * (self.down[id(child)] @ jc69_matrix(child.length))
                scale = scale + self.down_scale[id(child)]
            vec, scale = _rescale(vec, scale)
            self.down[id(node)] = vec
            self.down_scale[id(node)] = scale

    def _compute_outside(self) -> None:
        """Vectors for 'everything except the subtree below this branch'.

        ``outside[v]`` lives at the *parent* end of the branch above v.
        """
        n_pat = len(self.data.weights)
        self.outside.clear()
        self.outside_scale.clear()
        above: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
            id(self.tree.root): (np.tile(_FREQ, (n_pat, 1)), np.zeros(n_pat))
        }
        for node in self.tree.preorder():
            if node.is_leaf:
                continue
            a_vec, a_scale = above[id(node)]
            # messages from each child, up through its branch
            msgs = [self.down[id(c)] @ jc69_matrix(c.length) for c in node.children]
            for k, child in enumerate(node.children):
                vec = a_vec.copy()
                scale = a_scale.copy()
                for m, other in enumerate(node.children):
                    if m != k:
                        vec = vec * msgs[m]
                        scale = scale + self.down_scale[id(other)]
                vec, scale = _rescale(vec, scale)
                self.outside[id(child)] = vec
                self.outside_scale[id(child)] = scale
                if not child.is_leaf:
                    up_vec, up_scale = _rescale(vec @ jc69_matrix(child.length), scale.copy())
                    above[id(child)] = (up_vec, up_scale)

    def update(self) -> None:
        self._compute_down()
        self._compute_outside()

    # ---------------------------------------------------------------- #
    def log_likelihood(self) -> float:
        self._compute_down()
        root = self.tree.root
        site = self.down[id(root)] @ _FREQ
        return _weighted_log(site, self.down_scale[id(root)], self.data.weights)

    def branch_log_likelihood(self, node: Node, t: float) -> float:
        """Log-likelihood with the branch above ``node`` set to t.

        Requires ``update()`` to have been called for the current tree.
        """
        o = self.outside[id(node)]
        site = np.einsum("pi,pi->p", o, self.down[id(node)] @ jc69_matrix(t))
        scale = self.outside_scale[id(node)] + self.down_scale[id(node)]
        return _weighted_log(site, scale, self.data.weights)

    def optimize_branch(self, node: Node, tol: float = 1e-6) -> float:
        """Set the branch above ``node`` to its conditional optimum."""
        if node.parent is None:
            raise TreeError("the root has no branch")

        def negative(t: float) -> float:
            value = self.branch_log_likelihood(node, t)
            return -value if math.isfinite(value) else 1e300

        current = min(max(node.length, MIN_BRANCH), MAX_BRANCH)
        res = minimize_scalar(negative, bounds=(MIN_BRANCH, MAX_BRANCH),
                              method="bounded", options={"xatol": tol})
        # keep the old value if Brent did not find anything better
        if res.fun <= negative(current):
            node.length = float(res.x)
            return -float(res.fun)
        node.length = current
        return -negative(current)

    def optimize_branch_lengths(self, max_passes: int = 100, tol: float = 1e-5) -> float:
        """Coordinate ascent over all branch lengths.

        Stops when a full pass improves the log-likelihood by less than
        ``tol`` or after ``max_passes`` passes. Coordinate ascent can need
        10-20 passes on larger trees to settle to 1e-5.
        """
        previous = -np.inf
        for _ in range(max_passes):
            for node in self.tree.branches():
                self.update()
                self.optimize_branch(node)
            current = self.log_likelihood()
            if current - previous < tol:
                return current
            previous = current
        return self.log_likelihood()


def _rescale(vec: np.ndarray, scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = vec.max(axis=1)
    m_safe = np.where(m > 0, m, 1.0)
    return vec / m_safe[:, None], scale + np.log(m_safe)


def _weighted_log(site: np.ndarray, scale: np.ndarray, weights: np.ndarray) -> float:
    if np.any(site <= 0):
        # a site with probability zero (e.g. two different bases joined by
        # branches of length exactly zero)
        return -math.inf
    return float(np.dot(weights, np.log(site) + scale))


# ---------------------------------------------------------------------- #
# public helpers
# ---------------------------------------------------------------------- #
def log_likelihood(tree: Tree, alignment: Alignment) -> float:
    """JC69 log-likelihood of a tree with its current branch lengths."""
    return LikelihoodEngine(tree, PatternData.from_alignment(alignment)).log_likelihood()


def optimize_branch_lengths(tree: Tree, alignment: Alignment, max_passes: int = 100) -> float:
    """Optimise branch lengths in place; returns the final log-likelihood."""
    engine = LikelihoodEngine(tree, PatternData.from_alignment(alignment))
    return engine.optimize_branch_lengths(max_passes=max_passes)


@dataclass
class MLResult:
    tree: Tree
    log_likelihood: float
    start_log_likelihood: float
    n_nni_accepted: int = 0
    seconds: float = 0.0
    history: List[float] = field(default_factory=list)
    start_name: str = ""
    per_start: Dict[str, float] = field(default_factory=dict)


def _hill_climb(tree: Tree, data: PatternData, max_rounds: int) -> Tuple[float, float, int, List[float]]:
    """NNI hill climbing from one starting tree (in place)."""
    for node in tree.branches():
        node.length = min(max(node.length, 0.01), MAX_BRANCH)
    engine = LikelihoodEngine(tree, data)
    current = engine.optimize_branch_lengths(max_passes=3)
    start_ll = current
    history = [current]
    accepted = 0
    if tree.n_leaves >= 4:
        for _ in range(max_rounds):
            nodes = list(tree.preorder())
            best_ll = current
            best_move = None
            for i, j in nni_moves(tree):
                x, y = nodes[i], nodes[j]
                pivot = nni_pivot(tree, x, y)
                old_len = pivot.length
                swap_subtrees(x, y)
                engine.update()
                ll = engine.optimize_branch(pivot)
                swap_subtrees(x, y)
                pivot.length = old_len
                if ll > best_ll + 1e-6:
                    best_ll, best_move = ll, (x, y)
            if best_move is None:
                break
            swap_subtrees(*best_move)
            current = engine.optimize_branch_lengths(max_passes=2)
            history.append(current)
            accepted += 1
    return current, start_ll, accepted, history


def ml_search(alignment: Alignment, start_tree: Optional[Tree] = None,
              starts: Sequence[str] = ("nj", "upgma", "parsimony"),
              max_rounds: int = 200, seed: int = 0) -> MLResult:
    """Heuristic ML tree search: NNI hill climbing from several start trees.

    If ``start_tree`` is given, only that tree is used as a start.
    Otherwise ``starts`` names the starting trees ('nj', 'upgma',
    'parsimony'); the best final tree is returned.
    """
    from .parsimony import CharacterData, parsimony_search
    from .upgma import upgma

    t0 = time.perf_counter()
    if alignment.n_taxa < 3:
        raise ValueError("ML search needs at least three taxa")
    data = PatternData.from_alignment(alignment)
    candidates: List[Tuple[str, Tree]] = []
    if start_tree is not None:
        candidates.append(("given", start_tree.copy()))
    else:
        if not starts:
            raise ValueError("need at least one starting tree")
        dm = None
        for name in starts:
            if name in ("nj", "upgma") and dm is None:
                dm = distance_matrix(alignment)
            if name == "nj":
                candidates.append((name, neighbor_joining(dm)))
            elif name == "upgma":
                candidates.append((name, upgma(dm)))
            elif name == "parsimony":
                res = parsimony_search(CharacterData.from_alignment(alignment), n_starts=2, seed=seed)
                candidates.append((name, res.tree))
            else:
                raise ValueError(f"unknown starting tree {name!r}")

    best = None
    per_start: Dict[str, float] = {}
    seen_topologies: Dict[frozenset, float] = {}
    for name, tree in candidates:
        key = frozenset(tree.splits())
        if key in seen_topologies:  # identical start topology; same result as before
            per_start[name] = seen_topologies[key]
            continue
        ll, start_ll, accepted, history = _hill_climb(tree, data, max_rounds)
        seen_topologies[key] = ll
        per_start[name] = ll
        if best is None or ll > best[1] + 1e-9:
            best = (name, ll, start_ll, accepted, history, tree)
    assert best is not None
    name, _, start_ll, accepted, history, tree = best
    final = LikelihoodEngine(tree, data).optimize_branch_lengths()
    history.append(final)
    return MLResult(tree, final, start_ll, accepted, time.perf_counter() - t0, history, name, per_start)
