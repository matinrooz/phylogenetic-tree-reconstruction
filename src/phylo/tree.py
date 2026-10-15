"""Tree data structure used by every algorithm in the package.

Trees are stored as rooted trees made of ``Node`` objects. Most methods in
this package produce *binary* rooted trees. For methods that really infer an
unrooted tree (neighbor-joining, parsimony, likelihood under a reversible
model) the root position is arbitrary and carries no meaning. Comparisons
between trees (Robinson-Foulds) are therefore done on unrooted splits.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple


class TreeError(ValueError):
    """Raised when a tree is malformed or an operation does not apply."""


class Node:
    """A node in a phylogenetic tree.

    ``length`` is the length of the branch that connects this node to its
    parent. It is ignored for the root.
    """

    __slots__ = ("name", "length", "children", "parent")

    def __init__(self, name: Optional[str] = None, length: float = 0.0) -> None:
        self.name = name
        self.length = float(length)
        self.children: List["Node"] = []
        self.parent: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: "Node") -> "Node":
        if child.parent is not None:
            raise TreeError("node already has a parent; detach it first")
        child.parent = self
        self.children.append(child)
        return child

    def remove_child(self, child: "Node") -> None:
        self.children.remove(child)
        child.parent = None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        kind = "leaf" if self.is_leaf else f"{len(self.children)} children"
        return f"Node({self.name!r}, length={self.length:.4g}, {kind})"


class Tree:
    """A rooted tree with named leaves."""

    def __init__(self, root: Node) -> None:
        if root.parent is not None:
            raise TreeError("the root node must not have a parent")
        self.root = root

    # ------------------------------------------------------------------ #
    # traversal
    # ------------------------------------------------------------------ #
    def preorder(self) -> Iterator[Node]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def postorder(self) -> Iterator[Node]:
        # iterative so deep (caterpillar-like) trees do not hit recursion limits
        stack: List[Tuple[Node, bool]] = [(self.root, False)]
        while stack:
            node, visited = stack.pop()
            if visited or node.is_leaf:
                yield node
            else:
                stack.append((node, True))
                for child in reversed(node.children):
                    stack.append((child, False))

    def leaves(self) -> List[Node]:
        return [n for n in self.preorder() if n.is_leaf]

    def leaf_names(self) -> List[str]:
        return [str(n.name) for n in self.leaves()]

    def internal_nodes(self) -> List[Node]:
        return [n for n in self.preorder() if not n.is_leaf]

    def branches(self) -> List[Node]:
        """All non-root nodes; each one stands for the branch above it."""
        return [n for n in self.preorder() if not n.is_root]

    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self.preorder() if n.is_leaf)

    # ------------------------------------------------------------------ #
    # basic properties
    # ------------------------------------------------------------------ #
    def is_binary(self) -> bool:
        return all(len(n.children) in (0, 2) for n in self.preorder())

    def total_length(self) -> float:
        return sum(n.length for n in self.branches())

    def root_to_tip(self) -> Dict[str, float]:
        depth: Dict[int, float] = {id(self.root): 0.0}
        out: Dict[str, float] = {}
        for node in self.preorder():
            if node.parent is not None:
                depth[id(node)] = depth[id(node.parent)] + node.length
            if node.is_leaf:
                out[str(node.name)] = depth[id(node)]
        return out

    def path_length(self, a: str, b: str) -> float:
        """Sum of branch lengths on the path between two leaves."""
        nodes = {str(n.name): n for n in self.leaves()}
        if a not in nodes or b not in nodes:
            raise TreeError(f"unknown leaf {a!r} or {b!r}")
        ancestors: Dict[int, float] = {}
        dist = 0.0
        node: Optional[Node] = nodes[a]
        while node is not None:
            ancestors[id(node)] = dist
            dist += node.length
            node = node.parent
        dist = 0.0
        node = nodes[b]
        while id(node) not in ancestors:
            dist += node.length
            node = node.parent
            if node is None:  # pragma: no cover - only for broken trees
                raise TreeError("leaves are not in the same tree")
        return dist + ancestors[id(node)]

    def validate(self) -> None:
        """Check parent pointers, leaf names and branch lengths."""
        seen: Set[str] = set()
        for node in self.preorder():
            for child in node.children:
                if child.parent is not node:
                    raise TreeError("inconsistent parent pointer")
            if node.is_leaf:
                if node.name is None or node.name == "":
                    raise TreeError("every leaf needs a name")
                if node.name in seen:
                    raise TreeError(f"duplicate leaf name {node.name!r}")
                seen.add(node.name)
            elif len(node.children) == 1:
                raise TreeError("internal node with a single child")
            if not node.is_root and not math.isfinite(node.length):
                raise TreeError("branch lengths must be finite")

    # ------------------------------------------------------------------ #
    # copying
    # ------------------------------------------------------------------ #
    def copy(self) -> "Tree":
        """Deep copy. Preorder positions of nodes are preserved."""
        mapping: Dict[int, Node] = {}
        for node in self.preorder():
            new = Node(node.name, node.length)
            mapping[id(node)] = new
            if node.parent is not None:
                mapping[id(node.parent)].add_child(new)
        return Tree(mapping[id(self.root)])

    # ------------------------------------------------------------------ #
    # splits
    # ------------------------------------------------------------------ #
    def splits(self) -> Set[FrozenSet[str]]:
        """Non-trivial bipartitions of the leaf set (unrooted view).

        Each split is stored as the side that does *not* contain the
        alphabetically first leaf, so the two orientations of the same
        bipartition compare equal. Splits that separate a single leaf (or
        nothing) are left out because every tree on the same taxa has them.
        """
        all_leaves = frozenset(self.leaf_names())
        n = len(all_leaves)
        anchor = min(all_leaves)
        below: Dict[int, FrozenSet[str]] = {}
        result: Set[FrozenSet[str]] = set()
        for node in self.postorder():
            if node.is_leaf:
                below[id(node)] = frozenset([str(node.name)])
            else:
                below[id(node)] = frozenset().union(*(below[id(c)] for c in node.children))
            side = below[id(node)]
            if anchor in side:
                side = all_leaves - side
            if 2 <= len(side) <= n - 2:
                result.add(side)
        return result

    def clades(self) -> Set[FrozenSet[str]]:
        """Leaf sets below each internal non-root node (rooted view)."""
        below: Dict[int, FrozenSet[str]] = {}
        result: Set[FrozenSet[str]] = set()
        for node in self.postorder():
            if node.is_leaf:
                below[id(node)] = frozenset([str(node.name)])
            else:
                below[id(node)] = frozenset().union(*(below[id(c)] for c in node.children))
                if not node.is_root:
                    result.add(below[id(node)])
        return result

    # ------------------------------------------------------------------ #
    # Newick (thin wrappers so users do not need a second import)
    # ------------------------------------------------------------------ #
    def to_newick(self, precision: int = 6, lengths: bool = True) -> str:
        from .newick import write_newick

        return write_newick(self, precision=precision, lengths=lengths)

    @classmethod
    def from_newick(cls, text: str) -> "Tree":
        from .newick import parse_newick

        return parse_newick(text)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Tree({self.n_leaves} leaves)"


# ---------------------------------------------------------------------- #
# helpers shared by the search algorithms
# ---------------------------------------------------------------------- #
def swap_subtrees(x: Node, y: Node) -> None:
    """Exchange the positions of two subtrees (with their branch lengths).

    Neither node may be an ancestor of the other. Calling the function twice
    with the same arguments restores the original tree.
    """
    px, py = x.parent, y.parent
    if px is None or py is None:
        raise TreeError("cannot swap the root")
    ix, iy = px.children.index(x), py.children.index(y)
    px.children[ix], py.children[iy] = y, x
    x.parent, y.parent = py, px


def nni_moves(tree: Tree) -> List[Tuple[int, int]]:
    """List the nearest-neighbour-interchange moves of a binary tree.

    A move is a pair of preorder indices of two subtrees to exchange. Every
    move produces a different unrooted topology, and a binary tree with n
    leaves has exactly 2(n - 3) of them.
    """
    if not tree.is_binary():
        raise TreeError("NNI needs a binary tree")
    nodes = list(tree.preorder())
    index = {id(n): i for i, n in enumerate(nodes)}
    root = tree.root
    moves: List[Tuple[int, int]] = []
    for v in nodes:
        if v.is_root or v.is_leaf:
            continue
        p = v.parent
        assert p is not None
        sibling = p.children[1] if p.children[0] is v else p.children[0]
        a = v.children[0]
        if p is not root:
            # internal branch p-v: the subtree hanging off p (its other child)
            # is swapped with either child of v
            for child in v.children:
                moves.append((index[id(sibling)], index[id(child)]))
        else:
            # The two branches below the root form a single unrooted branch.
            # It is internal only if both root children are internal, and we
            # handle it once (from the first root child).
            if sibling.is_leaf or v is not p.children[0]:
                continue
            for other in sibling.children:
                moves.append((index[id(a)], index[id(other)]))
    return moves


def nni_pivot(tree: Tree, x: Node, y: Node) -> Node:
    """The node whose parent branch is the internal branch an NNI acts on.

    Call this *before* applying the move. For a move from ``nni_moves``:
    in the ordinary case ``y`` is a child of the pivot; for the branch
    below the root ``x`` is a child of the first root child, which we use
    as the pivot (its sibling's branch is the other half of the same
    unrooted branch).
    """
    if y.parent is None or x.parent is None:
        raise TreeError("not an NNI move")
    if y.parent.parent is tree.root and x.parent.parent is tree.root:
        return x.parent
    return y.parent


def unique_insertion_branches(tree: Tree) -> List[Node]:
    """Branches where a new leaf can be attached, one per unrooted branch.

    In a rooted binary tree the two branches below the root are a single
    branch of the unrooted tree, so the second root child is skipped.
    """
    root = tree.root
    skip = root.children[1] if len(root.children) == 2 else None
    return [n for n in tree.branches() if n is not skip]


def insert_leaf(branch: Node, leaf: Node) -> Node:
    """Attach ``leaf`` to the middle of the branch above ``branch``.

    Returns the newly created internal node, which ``remove_inserted_leaf``
    uses to undo the insertion.
    """
    parent = branch.parent
    if parent is None:
        raise TreeError("cannot insert above the root")
    half = branch.length / 2.0
    middle = Node(None, half)
    idx = parent.children.index(branch)
    parent.children[idx] = middle
    middle.parent = parent
    branch.parent = None
    branch.length = half
    middle.add_child(branch)
    middle.add_child(leaf)
    return middle


def remove_inserted_leaf(middle: Node) -> None:
    """Undo ``insert_leaf``."""
    parent = middle.parent
    if parent is None or len(middle.children) != 2:
        raise TreeError("not an inserted node")
    branch, leaf = middle.children
    middle.remove_child(leaf)
    middle.remove_child(branch)
    branch.length += middle.length
    idx = parent.children.index(middle)
    parent.children[idx] = branch
    branch.parent = parent
    middle.parent = None


def prune_to_leaves(tree: Tree, keep: Set[str]) -> Tree:
    """Return the subtree induced by ``keep``.

    Nodes left with a single child are removed and their branch lengths are
    added up, so the result has no unary nodes.
    """
    work = tree.copy()
    missing = keep - set(work.leaf_names())
    if missing:
        raise TreeError(f"unknown leaves: {sorted(missing)[:5]}")
    for node in list(work.postorder()):
        if node.is_leaf and node.name not in keep and node.parent is not None:
            node.parent.remove_child(node)
        elif not node.is_leaf and not node.children and node.parent is not None:
            node.parent.remove_child(node)
    # suppress unary nodes
    changed = True
    while changed:
        changed = False
        for node in list(work.postorder()):
            if len(node.children) == 1:
                child = node.children[0]
                if node.parent is None:
                    node.remove_child(child)
                    work = Tree(child)
                    child.length = 0.0
                else:
                    parent = node.parent
                    idx = parent.children.index(node)
                    node.remove_child(child)
                    child.length += node.length
                    parent.children[idx] = child
                    child.parent = parent
                    node.parent = None
                changed = True
                break
    return work
