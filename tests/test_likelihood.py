import itertools
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from phylo.alignment import Alignment
from phylo.distance import distance_matrix
from phylo.methods import (LikelihoodEngine, PatternData, jc69_matrix, log_likelihood, ml_search,
                           neighbor_joining, optimize_branch_lengths)
from phylo.metrics import robinson_foulds
from phylo.newick import parse_newick
from phylo.simulate import simulate_dataset, yule_tree


def brute_force_loglik(tree, aln):
    """Sum over every assignment of bases to internal nodes (tiny trees only)."""
    internal = [n for n in tree.preorder() if not n.is_leaf]
    bases = dict(zip(aln.names, aln.single_base_index()))
    total = 0.0
    for k in range(aln.n_sites):
        site = 0.0
        for anc in itertools.product(range(4), repeat=len(internal)):
            state = {id(n): s for n, s in zip(internal, anc)}
            for leaf in tree.leaves():
                state[id(leaf)] = int(bases[str(leaf.name)][k])
            p = 0.25
            for node in tree.branches():
                p *= jc69_matrix(node.length)[state[id(node.parent)], state[id(node)]]
            site += p
        total += math.log(site)
    return total


def reference_log_space(tree, aln):
    """Pruning done entirely in log space; slow but independent of the scaling trick."""
    data = PatternData.from_alignment(aln)
    with np.errstate(divide="ignore"):
        logs = {}
        for node in tree.postorder():
            if node.is_leaf:
                logs[id(node)] = np.log(data.tips[str(node.name)])
                continue
            acc = 0.0
            for c in node.children:
                logp = np.log(jc69_matrix(c.length))
                acc = acc + logsumexp(logp[None, :, :] + logs[id(c)][:, None, :], axis=2)
            logs[id(node)] = acc
        site = logsumexp(logs[id(tree.root)] + math.log(0.25), axis=1)
    return float(np.dot(data.weights, site))


def test_jc_matrix_properties():
    assert np.allclose(jc69_matrix(0.0), np.eye(4))
    for t in (0.01, 0.5, 3.0, 50.0):
        m = jc69_matrix(t)
        assert np.allclose(m.sum(axis=1), 1.0) and np.allclose(m, m.T)
    assert np.allclose(jc69_matrix(1e6), 0.25)
    with pytest.raises(ValueError):
        jc69_matrix(-1)


def test_pruning_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(4):
        true, aln = simulate_dataset(5, 12, 0.5, seed=int(rng.integers(1000)))
        assert log_likelihood(true, aln) == pytest.approx(brute_force_loglik(true, aln), rel=1e-10)


def test_likelihood_does_not_depend_on_root():
    # same unrooted tree, root moved along a branch and to another branch
    a = parse_newick("((A:0.1,B:0.2):0.05,(C:0.3,D:0.15):0.25);")
    b = parse_newick("((A:0.1,B:0.2):0.3,(C:0.3,D:0.15):0.0);")
    c = parse_newick("(A:0.1,(B:0.2,(C:0.3,D:0.15):0.3):0.0);")
    _, aln = simulate_dataset(4, 50, 0.3, seed=1)
    aln.names = ["A", "B", "C", "D"]
    assert log_likelihood(a, aln) == pytest.approx(log_likelihood(b, aln))
    assert log_likelihood(a, aln) == pytest.approx(log_likelihood(c, aln))


def test_scaling_on_large_tree():
    rng = np.random.default_rng(1)
    tree = yule_tree(300, rng, 2.0)
    from phylo.simulate import simulate_jc69
    aln = simulate_jc69(tree, 200, rng)
    value = log_likelihood(tree, aln)
    assert math.isfinite(value)
    assert value == pytest.approx(reference_log_space(tree, aln), rel=1e-9)


def test_two_taxa_ml_branch_length_is_jc_distance():
    aln = Alignment.from_sequences([("a", "ACGTACGTACGTACGTACGT"), ("b", "ACGTTCGTACGAACGTACCT")])
    tree = parse_newick("(a:0.1,b:0.1);")
    optimize_branch_lengths(tree, aln)
    expected = distance_matrix(aln).values[0, 1]
    assert tree.path_length("a", "b") == pytest.approx(expected, rel=1e-3)


def test_zero_length_branches():
    same = Alignment.from_sequences([("a", "ACGT"), ("b", "ACGT"), ("c", "ACGA")])
    assert math.isfinite(log_likelihood(parse_newick("((a:0,b:0):0.1,c:0.2);"), same))
    diff = Alignment.from_sequences([("a", "ACGT"), ("b", "ACGA"), ("c", "ACGA")])
    assert log_likelihood(parse_newick("((a:0,b:0):0.1,c:0.2);"), diff) == -math.inf
    # optimisation moves the branches away from zero and gives a finite value
    tree = parse_newick("((a:0,b:0):0.1,c:0.2);")
    assert math.isfinite(optimize_branch_lengths(tree, diff))


def test_missing_data_handled():
    aln = Alignment.from_sequences([("a", "ACGTN"), ("b", "AC-TA"), ("c", "ACGTA"), ("d", "TCGTA")])
    tree = parse_newick("((a:0.1,b:0.1):0.1,(c:0.1,d:0.1):0.1);")
    full = Alignment.from_sequences([("a", "ACGT"), ("b", "ACTT"), ("c", "ACGT"), ("d", "TCGT")])
    assert math.isfinite(log_likelihood(tree, aln))
    # a column of all-missing data contributes log(1) = 0
    blank = Alignment.from_sequences([("a", "ACGTN"), ("b", "ACGTN"), ("c", "ACGTN"), ("d", "TCGTN")])
    four = Alignment.from_sequences([("a", "ACGT"), ("b", "ACGT"), ("c", "ACGT"), ("d", "TCGT")])
    assert log_likelihood(tree, blank) == pytest.approx(log_likelihood(tree, four))
    del full


def test_branch_optimisation_improves_likelihood():
    true, aln = simulate_dataset(10, 400, 0.4, seed=6)
    tree = true.copy()
    for node in tree.branches():
        node.length = 0.2
    before = log_likelihood(tree, aln)
    after = optimize_branch_lengths(tree, aln)
    assert after > before
    assert after == pytest.approx(log_likelihood(tree, aln))


def test_branch_likelihood_agrees_with_full_computation():
    true, aln = simulate_dataset(7, 100, 0.4, seed=8)
    engine = LikelihoodEngine(true, PatternData.from_alignment(aln))
    engine.update()
    full = engine.log_likelihood()
    for node in true.branches():
        assert engine.branch_log_likelihood(node, node.length) == pytest.approx(full, rel=1e-10)


def test_ml_search_improves_on_nj_and_recovers_easy_tree():
    true, aln = simulate_dataset(12, 2000, 0.3, seed=3, clock_sigma=0.3)
    result = ml_search(aln)
    nj = neighbor_joining(distance_matrix(aln))
    nj_ll = optimize_branch_lengths(nj, aln)
    assert result.log_likelihood >= nj_ll - 1e-6
    assert result.log_likelihood >= result.start_log_likelihood - 1e-6
    assert robinson_foulds(true, result.tree).distance == 0
    true_ll = optimize_branch_lengths(true.copy(), aln)
    assert result.log_likelihood >= true_ll - 1e-3
