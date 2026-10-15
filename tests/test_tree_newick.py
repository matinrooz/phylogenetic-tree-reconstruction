import itertools

import numpy as np
import pytest

from phylo.newick import NewickError, parse_newick, write_newick
from phylo.simulate import yule_tree
from phylo.tree import (Node, Tree, TreeError, insert_leaf, nni_moves, prune_to_leaves,
                        remove_inserted_leaf, swap_subtrees, unique_insertion_branches)


def test_parse_basic_structure_and_lengths():
    tree = parse_newick("((A:0.1,B:0.2)ab:0.3,C:0.4);")
    assert sorted(tree.leaf_names()) == ["A", "B", "C"]
    assert tree.is_binary()
    assert tree.path_length("A", "B") == pytest.approx(0.3)
    assert tree.path_length("A", "C") == pytest.approx(0.8)
    assert {n.name for n in tree.internal_nodes()} == {None, "ab"}


@pytest.mark.parametrize("text", [
    "((A:0.1,B:0.2):0.3,(C:1e-3,D:2.5E-1):0);",
    "(A,B,(C,D));",
    "('taxon one':1,'it''s':2,C:3);",
    "((A:1,B:1)[a comment]:1,C:2);",
])
def test_newick_round_trip(text):
    tree = parse_newick(text)
    again = parse_newick(write_newick(tree))
    assert sorted(again.leaf_names()) == sorted(tree.leaf_names())
    assert again.splits() == tree.splits()
    for a, b in itertools.combinations(tree.leaf_names(), 2):
        assert again.path_length(a, b) == pytest.approx(tree.path_length(a, b))


def test_round_trip_random_trees():
    rng = np.random.default_rng(3)
    for n in (3, 5, 20, 60):
        tree = yule_tree(n, rng, 0.4)
        again = parse_newick(tree.to_newick(precision=17))
        assert again.splits() == tree.splits()
        assert again.total_length() == pytest.approx(tree.total_length())


def test_quoted_names_survive():
    tree = parse_newick("('a b':1,'c,d':1,e:1);")
    assert set(tree.leaf_names()) == {"a b", "c,d", "e"}
    assert set(parse_newick(tree.to_newick()).leaf_names()) == {"a b", "c,d", "e"}


@pytest.mark.parametrize("bad", ["", "((A,B);", "(A,B));", "(A:x,B);", "(A,A);", "(A,B)C D;", "('A,B);"])
def test_bad_newick_raises(bad):
    with pytest.raises((NewickError, TreeError)):
        parse_newick(bad)


def test_splits_ignore_root_position():
    # the same unrooted tree, rooted in two different places
    a = parse_newick("((A,B),(C,(D,E)));")
    b = parse_newick("(A,(B,((D,E),C)));")
    assert a.splits() == b.splits()
    assert len(a.splits()) == 2  # n - 3 internal branches


def test_copy_is_independent():
    tree = parse_newick("((A:1,B:1):1,C:1);")
    clone = tree.copy()
    clone.leaves()[0].length = 99
    assert tree.leaves()[0].length == 1


def test_validate_catches_problems():
    root = Node()
    root.add_child(Node("A"))
    root.add_child(Node(None))
    with pytest.raises(TreeError):
        Tree(root).validate()
    with pytest.raises(TreeError):
        Node("x").add_child(root.children[0])  # already has a parent


def test_nni_moves_count_and_distinct_topologies():
    rng = np.random.default_rng(0)
    for n in (4, 5, 8, 15):
        for _ in range(5):
            tree = yule_tree(n, rng, 0.3)
            moves = nni_moves(tree)
            assert len(moves) == 2 * (n - 3)
            nodes = list(tree.preorder())
            original = tree.splits()
            seen = set()
            for i, j in moves:
                swap_subtrees(nodes[i], nodes[j])
                new = frozenset(tree.splits())
                tree.validate()
                swap_subtrees(nodes[i], nodes[j])
                assert new != frozenset(original)
                # NNI changes exactly one split
                assert len(set(new) - original) == 1
                seen.add(new)
            assert len(seen) == len(moves)
            assert tree.splits() == original  # all moves were undone


def test_insert_and_remove_leaf_restores_tree():
    tree = parse_newick("((A:2,B:2):1,C:3);")
    before = tree.to_newick()
    leaf = Node("D", 1.0)
    for branch in unique_insertion_branches(tree):
        middle = insert_leaf(branch, leaf)
        tree.validate()
        assert tree.n_leaves == 4
        remove_inserted_leaf(middle)
        assert tree.to_newick() == before


def test_unique_insertion_branches_count():
    rng = np.random.default_rng(1)
    for n in (3, 6, 10):
        tree = yule_tree(n, rng, 0.2)
        assert len(unique_insertion_branches(tree)) == 2 * n - 3


def test_prune_keeps_path_lengths():
    rng = np.random.default_rng(5)
    tree = yule_tree(20, rng, 1.0)
    keep = set(tree.leaf_names()[:7])
    pruned = prune_to_leaves(tree, keep)
    pruned.validate()
    assert set(pruned.leaf_names()) == keep and pruned.is_binary()
    for a, b in itertools.combinations(sorted(keep), 2):
        assert pruned.path_length(a, b) == pytest.approx(tree.path_length(a, b))
