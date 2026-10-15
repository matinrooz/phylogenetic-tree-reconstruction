import numpy as np
import pytest

from phylo.metrics import robinson_foulds
from phylo.newick import parse_newick
from phylo.simulate import relax_clock, simulate_dataset, simulate_jc69, yule_tree
from phylo.tree import TreeError


def test_yule_tree_shape_and_height():
    rng = np.random.default_rng(0)
    tree = yule_tree(25, rng, tree_height=0.7)
    tree.validate()
    assert tree.n_leaves == 25 and tree.is_binary()
    depths = list(tree.root_to_tip().values())
    assert np.allclose(depths, 0.7)


def test_relaxed_clock_breaks_ultrametricity_but_keeps_topology():
    rng = np.random.default_rng(1)
    tree = yule_tree(20, rng, 0.5)
    relaxed = relax_clock(tree, rng, 0.8)
    assert relaxed.splits() == tree.splits()
    assert np.std(list(relaxed.root_to_tip().values())) > 0.01
    assert relax_clock(tree, rng, 0.0).to_newick() == tree.to_newick()


def test_simulation_is_reproducible():
    t1, a1 = simulate_dataset(9, 100, 0.3, seed=123, clock_sigma=0.4)
    t2, a2 = simulate_dataset(9, 100, 0.3, seed=123, clock_sigma=0.4)
    assert t1.to_newick() == t2.to_newick()
    assert np.array_equal(a1.masks, a2.masks)
    _, a3 = simulate_dataset(9, 100, 0.3, seed=124)
    assert not np.array_equal(a1.masks, a3.masks)


def test_jc_simulation_matches_expected_divergence():
    # two leaves at distance t: the expected proportion of differences is 3/4 (1 - exp(-4t/3))
    rng = np.random.default_rng(2)
    tree = parse_newick("(a:0.15,b:0.15);")
    aln = simulate_jc69(tree, 200_000, rng)
    p = float(np.mean(aln.masks[0] != aln.masks[1]))
    assert p == pytest.approx(0.75 * (1 - np.exp(-4 * 0.3 / 3)), abs=0.005)


def test_missing_fraction():
    _, aln = simulate_dataset(6, 2000, 0.3, seed=5, missing_fraction=0.2)
    assert np.mean(aln.masks == 15) == pytest.approx(0.2, abs=0.02)


def test_robinson_foulds_basics():
    a = parse_newick("((A,B),(C,(D,E)));")
    assert robinson_foulds(a, a).distance == 0
    rerooted = parse_newick("(E,(D,(C,(A,B))));")
    assert robinson_foulds(a, rerooted).distance == 0  # root is ignored
    b = parse_newick("((A,C),(B,(D,E)));")
    rf = robinson_foulds(a, b)
    assert (rf.distance, rf.max_distance, rf.false_positives, rf.false_negatives) == (2, 4, 1, 1)
    assert rf.normalized == pytest.approx(0.5)
    star = parse_newick("(A,B,C,D,E);")
    assert robinson_foulds(a, star).false_negatives == 2
    with pytest.raises(TreeError):
        robinson_foulds(a, parse_newick("((A,B),(C,(D,F)));"))


def test_comparison_highlighting_matches_rf():
    # the two root branches of a rooted tree are one unrooted split, so we
    # count highlighted *splits*, which must equal the RF false negatives/positives
    from phylo.methods import upgma
    from phylo.distance import distance_matrix
    from phylo.plotting import _rooted_clades_as_splits, conflicting_clades

    for seed in range(6):
        true, aln = simulate_dataset(14, 300, 0.3, seed=seed, clock_sigma=1.0)
        inferred = upgma(distance_matrix(aln))
        missed, wrong = conflicting_clades(true, inferred)
        rf = robinson_foulds(true, inferred)
        true_map, inf_map = _rooted_clades_as_splits(true), _rooted_clades_as_splits(inferred)
        assert len({true_map[c] for c in missed}) == rf.false_negatives
        assert len({inf_map[c] for c in wrong}) == rf.false_positives
