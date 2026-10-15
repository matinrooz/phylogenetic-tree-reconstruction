"""A simple simulator for CRISPR-based lineage-tracing data.

What it models
--------------
In lineage-tracing experiments (GESTALT, scGESTALT, the Cassiopeia-style
target-site arrays, and others) cells carry an array of CRISPR target
sites. Cas9 editing creates heritable insertions/deletions at those sites
over time, and the pattern of edits shared between cells is used to
reconstruct how the cells are related.

This simulator keeps only the core of that idea:

* Cells divide synchronously for ``n_generations`` generations, giving a
  complete binary tree with 2**n_generations cells. Every branch is one
  generation long.
* Each cell has ``n_sites`` target sites. A site starts unedited (state 0).
  In each generation, every still-unedited site is edited independently
  with probability ``edit_rate``.
* An edit picks one of ``n_outcomes`` indel alleles (states 1..n_outcomes).
  Outcome probabilities follow a power law, so a few alleles are common and
  most are rare. Common alleles can arise independently in unrelated
  cells (homoplasy), which is one of the real difficulties of this data.
* Edits are irreversible: an edited site never changes again.
* ``n_cells`` cells are sampled at the end (we never observe every cell),
  and the true tree is the subtree that connects them.
* Each observed value is lost with probability ``dropout_rate``
  (independent, non-heritable missing data).

What it leaves out (on purpose)
-------------------------------
Real data has things this simulator does not: large deletions that remove
several neighbouring sites at once (heritable, correlated dropout), editing
rates that differ between sites and change over time, asynchronous
divisions, cell death, sequencing errors, and allele probabilities that
are estimated rather than known. Results on this simulator show how the
algorithms behave on the basic problem; they do not predict how well any
method works on a real experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..tree import Node, Tree, prune_to_leaves

UNEDITED = 0
MISSING = -1


@dataclass
class LineageConfig:
    n_generations: int = 7
    n_cells: int = 32
    n_sites: int = 20
    edit_rate: float = 0.05
    n_outcomes: int = 30
    outcome_skew: float = 1.5   # exponent of the power law over indel alleles
    dropout_rate: float = 0.1

    def validate(self) -> None:
        if self.n_generations < 1:
            raise ValueError("n_generations must be at least 1")
        if not 3 <= self.n_cells <= 2 ** self.n_generations:
            raise ValueError("n_cells must be between 3 and 2**n_generations")
        if self.n_sites < 1 or self.n_outcomes < 1:
            raise ValueError("n_sites and n_outcomes must be positive")
        if self.n_outcomes > 63:
            raise ValueError("at most 63 outcomes (Fitch bitmasks hold 64 states)")
        if not 0 <= self.edit_rate <= 1 or not 0 <= self.dropout_rate < 1:
            raise ValueError("edit_rate must be in [0, 1] and dropout_rate in [0, 1)")

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LineageData:
    true_tree: Tree
    names: List[str]
    states: np.ndarray            # (cells x sites), 0 = unedited, -1 = missing
    outcome_probs: np.ndarray     # probability of each indel allele 1..K

    @property
    def fraction_edited(self) -> float:
        observed = self.states >= 0
        return float((self.states[observed] > 0).mean()) if observed.any() else 0.0


def outcome_distribution(n_outcomes: int, skew: float) -> np.ndarray:
    ranks = np.arange(1, n_outcomes + 1, dtype=float)
    p = ranks ** (-skew)
    return p / p.sum()


def simulate_lineage(config: LineageConfig, seed: int) -> LineageData:
    config.validate()
    rng = np.random.default_rng(seed)
    probs = outcome_distribution(config.n_outcomes, config.outcome_skew)

    root = Node(None)
    root_state = np.zeros(config.n_sites, dtype=np.int64)
    frontier: List[Tuple[Node, np.ndarray]] = [(root, root_state)]
    for _generation in range(config.n_generations):
        next_frontier = []
        for parent, state in frontier:
            for _ in range(2):
                child = parent.add_child(Node(None, 1.0))
                new_state = state.copy()
                unedited = new_state == UNEDITED
                hits = unedited & (rng.random(config.n_sites) < config.edit_rate)
                if hits.any():
                    new_state[hits] = rng.choice(config.n_outcomes, size=int(hits.sum()), p=probs) + 1
                next_frontier.append((child, new_state))
        frontier = next_frontier

    for i, (leaf, _) in enumerate(frontier):
        leaf.name = f"cell{i + 1}"
    full_tree = Tree(root)

    chosen = sorted(rng.choice(len(frontier), size=config.n_cells, replace=False))
    names = [f"cell{i + 1}" for i in chosen]
    states = np.array([frontier[i][1] for i in chosen])
    if config.dropout_rate > 0:
        drop = rng.random(states.shape) < config.dropout_rate
        states = np.where(drop, MISSING, states)

    true_tree = prune_to_leaves(full_tree, set(names))
    return LineageData(true_tree, names, states, probs)
