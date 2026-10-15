"""Pairwise evolutionary distances.

For every pair of sequences only the sites where both sequences have an
unambiguous base are compared ("pairwise deletion"). The proportion of
differing sites p is then corrected for multiple hits with the Jukes-Cantor
formula

    d = -3/4 * ln(1 - 4/3 * p)

which is the maximum-likelihood distance under the JC69 model. When
p >= 3/4 the formula has no finite value (the sequences look no more similar
than random sequences). Instead of returning infinity we cap the distance at
``saturation_cap`` and count how often that happened, so the caller can warn
about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .alignment import Alignment, AlignmentError

DEFAULT_SATURATION_CAP = 5.0


@dataclass
class DistanceMatrix:
    names: List[str]
    values: np.ndarray
    n_saturated: int = 0

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        n = len(self.names)
        if self.values.shape != (n, n):
            raise ValueError("distance matrix shape does not match the number of names")
        if not np.allclose(self.values, self.values.T):
            raise ValueError("distance matrix must be symmetric")
        if np.any(np.diag(self.values) != 0):
            raise ValueError("distance matrix must have a zero diagonal")
        if np.any(self.values < 0) or not np.all(np.isfinite(self.values)):
            raise ValueError("distances must be finite and non-negative")

    def __getitem__(self, pair):
        a, b = pair
        return self.values[self.names.index(a), self.names.index(b)]


def p_distance_matrix(alignment: Alignment) -> tuple:
    """Proportion of differing sites and number of compared sites per pair."""
    bases = alignment.single_base_index()
    observed = bases >= 0
    n = alignment.n_taxa
    p = np.zeros((n, n))
    compared = np.zeros((n, n), dtype=int)
    for i in range(n):
        both = observed[i] & observed
        diff = (bases[i] != bases) & both
        compared[i] = both.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p[i] = diff.sum(axis=1) / compared[i]
    np.fill_diagonal(p, 0.0)
    np.fill_diagonal(compared, alignment.n_sites)
    no_overlap = [(alignment.names[i], alignment.names[j])
                  for i in range(n) for j in range(i + 1, n) if compared[i, j] == 0]
    if no_overlap:
        raise AlignmentError(
            f"no comparable sites between {no_overlap[0][0]} and {no_overlap[0][1]} "
            "(every site is missing or ambiguous in at least one of them)"
        )
    return p, compared


def jc69_distance(p: np.ndarray, saturation_cap: float = DEFAULT_SATURATION_CAP) -> np.ndarray:
    """Jukes-Cantor correction of p-distances, capped at ``saturation_cap``."""
    p = np.asarray(p, dtype=float)
    inside = 1.0 - (4.0 / 3.0) * p
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -0.75 * np.log(inside)
    d = np.where(inside <= 0, saturation_cap, d)
    return np.minimum(d, saturation_cap)


def distance_matrix(alignment: Alignment, model: str = "jc69",
                    saturation_cap: float = DEFAULT_SATURATION_CAP) -> DistanceMatrix:
    """Pairwise distances under ``model`` ('jc69' or 'p')."""
    p, _ = p_distance_matrix(alignment)
    if model == "p":
        values = p
        n_sat = 0
    elif model == "jc69":
        values = jc69_distance(p, saturation_cap)
        n_sat = int(np.triu(values >= saturation_cap, k=1).sum())
    else:
        raise ValueError(f"unknown distance model {model!r}; use 'jc69' or 'p'")
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    return DistanceMatrix(list(alignment.names), values, n_sat)
