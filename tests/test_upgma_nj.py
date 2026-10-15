import itertools

import numpy as np
import pytest

from phylo.distance import DistanceMatrix
from phylo.methods import neighbor_joining, upgma
from phylo.metrics import robinson_foulds
from phylo.newick import parse_newick
from phylo.simulate import relax_clock, yule_tree


def tree_distances(tree):
    names = sorted(tree.leaf_names())
    values = np.array([[tree.path_length(a, b) if a != b else 0.0 for b in names] for a in names])
    return DistanceMatrix(names, values)


def test_upgma_textbook_example():
    # classic 5-taxon example (a.k.a. the Wikipedia UPGMA example)
    names = list("abcde")
    d = np.array([[0, 17, 21, 31, 23],
                  [17, 0, 30, 34, 21],
                  [21, 30, 0, 28, 39],
                  [31, 34, 28, 0, 43],
                  [23, 21, 39, 43, 0]], dtype=float)
    tree = upgma(DistanceMatrix(names, d))
    expected = parse_newick("(((a:8.5,b:8.5):2.5,e:11):5.5,(c:14,d:14):2.5);")
    assert tree.clades() == expected.clades()
    for x, y in itertools.combinations(names, 2):
        assert tree.path_length(x, y) == pytest.approx(expected.path_length(x, y))
    depths = tree.root_to_tip()
    assert all(v == pytest.approx(16.5) for v in depths.values())


def test_upgma_recovers_ultrametric_tree():
    rng = np.random.default_rng(10)
    for n in (4, 9, 25):
        true = yule_tree(n, rng, 1.0)
        inferred = upgma(tree_distances(true))
        assert inferred.clades() == true.clades()  # even the root is right
        for a, b in itertools.combinations(true.leaf_names(), 2):
            assert inferred.path_length(a, b) == pytest.approx(true.path_length(a, b))


def test_nj_textbook_example():
    names = list("abcde")
    d = np.array([[0, 5, 9, 9, 8],
                  [5, 0, 10, 10, 9],
                  [9, 10, 0, 8, 7],
                  [9, 10, 8, 0, 3],
                  [8, 9, 7, 3, 0]], dtype=float)
    tree = neighbor_joining(DistanceMatrix(names, d))
    # the matrix is additive, so NJ must reproduce it exactly
    for i, j in itertools.combinations(range(5), 2):
        assert tree.path_length(names[i], names[j]) == pytest.approx(d[i, j])
    assert tree.splits() == {frozenset("de"), frozenset("cde")} or tree.splits() == {frozenset("de"), frozenset("ab")}


def test_nj_recovers_non_clock_tree():
    rng = np.random.default_rng(11)
    for n in (5, 12, 40):
        true = relax_clock(yule_tree(n, rng, 1.0), rng, sigma=1.0)
        inferred = neighbor_joining(tree_distances(true))
        assert robinson_foulds(true, inferred).distance == 0
        for a, b in itertools.combinations(true.leaf_names()[:10], 2):
            assert inferred.path_length(a, b) == pytest.approx(true.path_length(a, b), abs=1e-9)


def test_nj_branch_lengths_never_negative():
    rng = np.random.default_rng(12)
    names = [f"x{i}" for i in range(12)]
    noise = rng.random((12, 12))
    d = (noise + noise.T)
    np.fill_diagonal(d, 0)
    tree = neighbor_joining(DistanceMatrix(names, d))
    tree.validate()
    assert all(node.length >= 0 for node in tree.branches())


def test_identical_taxa_and_ties_are_deterministic():
    names = ["a", "b", "c", "d"]
    d = np.zeros((4, 4))
    t1 = upgma(DistanceMatrix(names, d))
    t2 = upgma(DistanceMatrix(names, d))
    assert t1.to_newick() == t2.to_newick()
    t3 = neighbor_joining(DistanceMatrix(names, d))
    assert all(node.length == 0 for node in t3.branches())


def test_too_few_taxa():
    with pytest.raises(ValueError):
        neighbor_joining(DistanceMatrix(["a", "b"], np.array([[0, 1], [1, 0]])))
    tree = upgma(DistanceMatrix(["a", "b"], np.array([[0, 1.0], [1.0, 0]])))
    assert tree.path_length("a", "b") == pytest.approx(1.0)


def test_nj_negative_lengths_option():
    # this simulated dataset (also used in the phangorn cross-check) gives
    # NJ a negative branch estimate
    from phylo.distance import distance_matrix
    from phylo.simulate import simulate_dataset

    _, aln = simulate_dataset(10, 150, 1.0, 105, clock_sigma=0.5)
    dm = distance_matrix(aln)
    clamped = neighbor_joining(dm)
    raw = neighbor_joining(dm, allow_negative=True)
    assert clamped.splits() == raw.splits()  # same topology either way
    assert any(node.length < 0 for node in raw.branches())
    assert all(node.length >= 0 for node in clamped.branches())
