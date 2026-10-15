"""Tree inference methods and a single entry point, ``infer_tree``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..alignment import Alignment
from ..distance import distance_matrix
from ..tree import Tree
from .likelihood import (LikelihoodEngine, MLResult, PatternData, jc69_matrix,
                         log_likelihood, ml_search, optimize_branch_lengths)
from .nj import neighbor_joining
from .parsimony import (CharacterData, ParsimonyResult, exhaustive_parsimony,
                        fitch_score, parsimony_search)
from .upgma import upgma

METHODS = ("upgma", "nj", "parsimony", "ml")


@dataclass
class Inference:
    method: str
    tree: Tree
    score: Optional[float] = None  # parsimony score or log-likelihood, if the method has one


def infer_tree(alignment: Alignment, method: str, seed: int = 0,
               parsimony_starts: int = 10) -> Inference:
    """Infer a tree with one of: 'upgma', 'nj', 'parsimony', 'ml'."""
    method = method.lower()
    if method == "upgma":
        return Inference(method, upgma(distance_matrix(alignment)))
    if method == "nj":
        return Inference(method, neighbor_joining(distance_matrix(alignment)))
    if method == "parsimony":
        nj_start = neighbor_joining(distance_matrix(alignment))
        res = parsimony_search(CharacterData.from_alignment(alignment), n_starts=parsimony_starts, seed=seed,
                               start_trees=[nj_start])
        return Inference(method, res.tree, res.score)
    if method == "ml":
        res = ml_search(alignment, seed=seed)
        return Inference(method, res.tree, res.log_likelihood)
    raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")


__all__ = [
    "METHODS", "Inference", "infer_tree", "upgma", "neighbor_joining",
    "CharacterData", "ParsimonyResult", "fitch_score", "parsimony_search", "exhaustive_parsimony",
    "LikelihoodEngine", "MLResult", "PatternData", "jc69_matrix", "log_likelihood",
    "ml_search", "optimize_branch_lengths",
]
