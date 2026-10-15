"""Tree comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .tree import Tree, TreeError


@dataclass
class RFResult:
    distance: int          # number of splits in one tree but not the other
    max_distance: int      # 2 (n - 3) for two binary trees on n taxa
    false_positives: int   # splits in the inferred tree that are wrong
    false_negatives: int   # true splits the inferred tree missed

    @property
    def normalized(self) -> float:
        return self.distance / self.max_distance if self.max_distance else 0.0


def robinson_foulds(true_tree: Tree, inferred: Tree) -> RFResult:
    """Unrooted Robinson-Foulds distance between two trees on the same taxa.

    The root of either tree is ignored, so this compares topologies only.
    Normalising by 2(n - 3) gives 0 for identical trees and 1 for trees
    that share no internal split.
    """
    a, b = set(true_tree.leaf_names()), set(inferred.leaf_names())
    if a != b:
        raise TreeError(f"trees have different taxa: {sorted(a ^ b)[:5]}")
    sa, sb = true_tree.splits(), inferred.splits()
    fn = len(sa - sb)
    fp = len(sb - sa)
    n = len(a)
    return RFResult(fn + fp, max(2 * (n - 3), 0), fp, fn)
