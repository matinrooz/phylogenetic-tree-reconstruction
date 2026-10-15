import itertools
import os

import numpy as np
import pytest

from phylo.alignment import Alignment
from phylo.methods import CharacterData, exhaustive_parsimony, fitch_score, parsimony_search
from phylo.methods.parsimony import all_unrooted_trees, site_scores
from phylo.newick import parse_newick
from phylo.simulate import simulate_dataset, yule_tree


def double_factorial(k):
    return 1 if k <= 1 else k * double_factorial(k - 2)


def brute_force_changes(tree, leaf_states):
    """Minimum number of changes for one site by trying every ancestral assignment.

    ``leaf_states`` maps leaf name -> set of allowed states. Exponential, only
    for checking Fitch on small trees.
    """
    internal = [n for n in tree.preorder() if not n.is_leaf]
    leaves = tree.leaves()
    leaf_options = [sorted(leaf_states[str(l.name)]) for l in leaves]
    best = np.inf
    for anc in itertools.product(range(4), repeat=len(internal)):
        state = {id(n): s for n, s in zip(internal, anc)}
        for choice in itertools.product(*leaf_options):
            for leaf, s in zip(leaves, choice):
                state[id(leaf)] = s
            changes = sum(state[id(n)] != state[id(n.parent)] for n in tree.branches())
            best = min(best, changes)
    return best


def test_hand_example(examples_dir):
    aln = Alignment.from_fasta(os.path.join(examples_dir, "hand_example.fasta"))
    data = CharacterData.from_alignment(aln)
    assert fitch_score(parse_newick("((A,B),(C,D));"), data) == 4
    assert fitch_score(parse_newick("((A,C),(B,D));"), data) == 6
    assert fitch_score(parse_newick("((A,D),(B,C));"), data) == 6
    assert exhaustive_parsimony(data).score == 4


def test_fitch_matches_brute_force_on_random_trees():
    rng = np.random.default_rng(0)
    for trial in range(15):
        n = int(rng.integers(3, 6))
        tree = yule_tree(n, rng, 0.5)
        sites = rng.integers(0, 4, size=(n, 6))
        names = sorted(tree.leaf_names())
        aln = Alignment.from_states(names, sites)
        data = CharacterData.from_alignment(aln)
        brute = 0
        for k in range(sites.shape[1]):
            brute += brute_force_changes(tree, {name: {int(sites[i, k])} for i, name in enumerate(names)})
        assert fitch_score(tree, data) == brute


def test_fitch_with_ambiguity_matches_brute_force():
    tree = parse_newick("((a,b),(c,(d,e)));")
    aln = Alignment.from_sequences([("a", "AR"), ("b", "GN"), ("c", "CA"), ("d", "YA"), ("e", "TG")])
    data = CharacterData.from_alignment(aln)
    bits = {"A": {0}, "C": {1}, "G": {2}, "T": {3}, "R": {0, 2}, "Y": {1, 3}, "N": {0, 1, 2, 3}}
    seqs = dict(aln.sequences())
    expected = sum(brute_force_changes(tree, {n: bits[s[k]] for n, s in seqs.items()}) for k in range(2))
    assert fitch_score(tree, data) == expected


def test_missing_data_and_identical_sequences_cost_nothing():
    aln = Alignment.from_sequences([("a", "NNNN"), ("b", "ACGT"), ("c", "ACGT"), ("d", "ACGT")])
    data = CharacterData.from_alignment(aln)
    for tree in all_unrooted_trees(["a", "b", "c", "d"]):
        assert fitch_score(tree, data) == 0


def test_score_does_not_depend_on_root():
    _, aln = simulate_dataset(6, 200, 0.6, seed=4)
    data = CharacterData.from_alignment(aln)
    n = aln.names
    a = parse_newick(f"(({n[0]},{n[1]}),({n[2]},({n[3]},({n[4]},{n[5]}))));")
    b = parse_newick(f"({n[0]},({n[1]},({n[2]},({n[3]},({n[4]},{n[5]})))));")
    c = parse_newick(f"(({n[4]},{n[5]}),({n[3]},({n[2]},({n[0]},{n[1]}))));")
    assert a.splits() == b.splits() == c.splits()
    assert fitch_score(a, data) == fitch_score(b, data) == fitch_score(c, data)


def test_pattern_compression_gives_same_score():
    true, aln = simulate_dataset(8, 300, 0.3, seed=9)
    data = CharacterData.from_alignment(aln)
    assert data.weights.sum() == aln.n_sites
    per_pattern = site_scores(true, data)
    assert float(np.dot(per_pattern, data.weights)) == fitch_score(true, data)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_number_of_unrooted_trees(n):
    trees = list(t.to_newick() for t in all_unrooted_trees([f"x{i}" for i in range(n)]))
    assert len(trees) == double_factorial(2 * n - 5)
    topologies = {frozenset(parse_newick(t).splits()) for t in trees}
    assert len(topologies) == len(trees)  # all distinct


def test_heuristic_finds_exact_optimum_on_small_data():
    hits = 0
    for seed in range(12):
        _, aln = simulate_dataset(8, 150, 0.6, seed=seed, clock_sigma=0.5)
        data = CharacterData.from_alignment(aln)
        exact = exhaustive_parsimony(data).score
        heuristic = parsimony_search(data, n_starts=5, seed=seed)
        assert heuristic.score >= exact  # can never beat the exact optimum
        assert heuristic.score == fitch_score(heuristic.tree, data)
        hits += heuristic.score == exact
    assert hits >= 11  # the search is a heuristic, but on 8 taxa it almost always succeeds


def test_search_is_reproducible():
    _, aln = simulate_dataset(14, 300, 0.4, seed=3)
    data = CharacterData.from_alignment(aln)
    r1 = parsimony_search(data, n_starts=3, seed=5)
    r2 = parsimony_search(data, n_starts=3, seed=5)
    assert r1.score == r2.score and r1.tree.splits() == r2.tree.splits()
    r1.tree.validate()
    assert r1.tree.is_binary() and sorted(r1.tree.leaf_names()) == sorted(aln.names)


def test_multistate_characters():
    names = ["a", "b", "c", "d"]
    states = np.array([[5, 0, -1], [5, 0, 7], [9, 3, 7], [9, 3, -1]])
    data = CharacterData.from_states(names, states)
    assert fitch_score(parse_newick("((a,b),(c,d));"), data) == 2
    assert fitch_score(parse_newick("((a,c),(b,d));"), data) == 4
    with pytest.raises(ValueError):
        CharacterData.from_states(names, np.full((4, 2), 70))


def test_exhaustive_refuses_large_input():
    _, aln = simulate_dataset(10, 50, 0.3, seed=1)
    with pytest.raises(ValueError):
        exhaustive_parsimony(CharacterData.from_alignment(aln))


def test_prune_removes_subtree_and_keeps_tree_valid():
    from phylo.methods.parsimony import _prune

    rng = np.random.default_rng(21)
    tree = yule_tree(12, rng, 0.5)
    for k in range(1, sum(1 for _ in tree.preorder())):
        work = tree.copy()
        node = list(work.preorder())[k]
        below = set(work.leaf_names())
        moved = [n for n in Tree_preorder(node) if n.is_leaf]
        rest = _prune(work, node)
        if rest.n_leaves >= 2:
            rest.validate()
        assert set(rest.leaf_names()) == below - {n.name for n in moved}


def Tree_preorder(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def test_spr_never_worsens_score_and_returns_valid_tree():
    from phylo.methods.parsimony import spr_hill_climb, stepwise_addition

    for seed in range(4):
        _, aln = simulate_dataset(15, 200, 0.8, seed=seed, clock_sigma=0.8)
        data = CharacterData.from_alignment(aln)
        start = stepwise_addition(data, list(aln.names), np.random.default_rng(seed))
        before = fitch_score(start, data)
        tree, score, _ = spr_hill_climb(start, data)
        tree.validate()
        assert tree.is_binary() and sorted(tree.leaf_names()) == sorted(aln.names)
        assert score <= before and score == fitch_score(tree, data)


def test_search_never_ends_worse_than_a_start_tree():
    from phylo.distance import distance_matrix
    from phylo.methods import neighbor_joining

    for seed in range(4):
        _, aln = simulate_dataset(14, 150, 1.0, seed=seed, clock_sigma=1.0)
        data = CharacterData.from_alignment(aln)
        nj = neighbor_joining(distance_matrix(aln))
        res = parsimony_search(data, n_starts=1, seed=seed, start_trees=[nj])
        assert res.score <= fitch_score(nj, data)
