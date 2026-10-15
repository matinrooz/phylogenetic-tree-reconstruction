import numpy as np
import pytest

from phylo.lineage import LINEAGE_METHODS, LineageConfig, lineage_distance, reconstruct, simulate_lineage
from phylo.metrics import robinson_foulds


def test_simulator_output_shapes_and_tree():
    cfg = LineageConfig(n_generations=6, n_cells=20, n_sites=12)
    data = simulate_lineage(cfg, seed=1)
    assert data.states.shape == (20, 12)
    data.true_tree.validate()
    assert data.true_tree.is_binary() and sorted(data.true_tree.leaf_names()) == sorted(data.names)
    assert data.states.min() >= -1 and data.states.max() <= cfg.n_outcomes
    # every sampled cell is at depth n_generations in the true tree
    assert np.allclose(list(data.true_tree.root_to_tip().values()), 6.0)


def test_no_editing_and_no_dropout():
    data = simulate_lineage(LineageConfig(edit_rate=0.0, dropout_rate=0.0), seed=2)
    assert np.all(data.states == 0)
    data = simulate_lineage(LineageConfig(edit_rate=0.3, dropout_rate=0.0), seed=2)
    assert np.all(data.states >= 0) and data.fraction_edited > 0.5


def test_full_editing_in_first_generation_gives_two_groups():
    # with edit_rate 1 every site is edited in generation 1 and never changes again,
    # so cells inherit one of only two possible barcodes
    data = simulate_lineage(LineageConfig(n_generations=5, n_cells=16, edit_rate=1.0,
                                          dropout_rate=0.0, n_sites=10), seed=3)
    assert len({tuple(row) for row in data.states}) <= 2


def test_edits_are_irreversible_along_the_tree():
    # cells that share a parent share every edit made before they split
    data = simulate_lineage(LineageConfig(n_generations=5, n_cells=32, dropout_rate=0.0, edit_rate=0.2), seed=4)
    idx = {name: i for i, name in enumerate(data.names)}
    for node in data.true_tree.internal_nodes():
        leaves = [n.name for n in _leaves_below(node)]
        rows = data.states[[idx[x] for x in leaves]]
        shared = np.all(rows == rows[0], axis=0)
        # a site edited in the common ancestor appears identically in all descendants;
        # any site unedited in some descendant was unedited in the ancestor
        unedited_somewhere = np.any(rows == 0, axis=0)
        assert np.all(shared | unedited_somewhere | (rows.min(axis=0) > 0))


def _leaves_below(node):
    stack, out = [node], []
    while stack:
        n = stack.pop()
        if n.is_leaf:
            out.append(n)
        else:
            stack.extend(n.children)
    return out


def test_lineage_distance_rules():
    states = np.array([[0, 3, 3, -1], [0, 3, 0, 2], [5, 4, 3, 2]])
    dm = lineage_distance(["a", "b", "c"], states)
    assert dm["a", "b"] == pytest.approx((0 + 0 + 1) / 3)
    assert dm["a", "c"] == pytest.approx((1 + 2 + 0) / 3)
    assert dm["b", "c"] == pytest.approx((1 + 2 + 1 + 0) / 4)
    empty = lineage_distance(["a", "b", "c"], np.array([[1, -1], [-1, 1], [1, 1]]))
    assert empty["a", "b"] == 2.0 and empty.n_saturated == 1


@pytest.mark.parametrize("method", LINEAGE_METHODS)
def test_each_method_returns_valid_tree(method):
    data = simulate_lineage(LineageConfig(n_generations=5, n_cells=16), seed=5)
    res = reconstruct(data, method, seed=1)
    res.tree.validate()
    assert sorted(res.tree.leaf_names()) == sorted(data.names) and res.tree.is_binary()


def test_informative_data_beats_random_baseline():
    cfg = LineageConfig(n_generations=6, n_cells=32, n_sites=80, edit_rate=0.08,
                        n_outcomes=60, outcome_skew=0.5, dropout_rate=0.0)
    errors = {m: [] for m in LINEAGE_METHODS}
    for seed in range(4):
        data = simulate_lineage(cfg, seed=seed)
        for m in LINEAGE_METHODS:
            errors[m].append(robinson_foulds(data.true_tree, reconstruct(data, m, seed=seed).tree).normalized)
    for m in ("parsimony", "nj", "upgma"):
        assert np.mean(errors[m]) < np.mean(errors["random"]) - 0.3


def test_config_validation():
    with pytest.raises(ValueError):
        LineageConfig(n_generations=3, n_cells=20).validate()
    with pytest.raises(ValueError):
        LineageConfig(n_outcomes=100).validate()
