"""FASTA input and the ``Alignment`` class.

Nucleotides are stored as 4-bit masks (A=1, C=2, G=4, T=8). A site that is
ambiguous holds several bits, and missing data (N, ?, gaps) holds all four.
The same masks are used directly by Fitch parsimony and turned into tip
vectors for the likelihood code, so ambiguity is handled the same way
everywhere.

Gaps are treated as missing data. That is the usual simplification in
parsimony and likelihood programs, but it throws away indel information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

A, C, G, T = 1, 2, 4, 8
ALL = 15

IUPAC: Dict[str, int] = {
    "A": A, "C": C, "G": G, "T": T, "U": T,
    "R": A | G, "Y": C | T, "S": C | G, "W": A | T, "K": G | T, "M": A | C,
    "B": C | G | T, "D": A | G | T, "H": A | C | T, "V": A | C | G,
    "N": ALL, "?": ALL, "-": ALL, ".": ALL,
}
BASES = "ACGT"
_MASK_TO_INDEX = {A: 0, C: 1, G: 2, T: 3}


class AlignmentError(ValueError):
    pass


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read (name, sequence) records from a FASTA file.

    The name is the first word after '>'. Sequences can span several lines.
    """
    records: List[Tuple[str, str]] = []
    name = None
    chunks: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                header = line[1:].strip()
                if not header:
                    raise AlignmentError(f"empty FASTA header on line {line_no}")
                name = header.split()[0]
                chunks = []
            else:
                if name is None:
                    raise AlignmentError(f"sequence data before the first header (line {line_no})")
                chunks.append("".join(line.split()))
    if name is not None:
        records.append((name, "".join(chunks)))
    if not records:
        raise AlignmentError(f"no sequences found in {path}")
    return records


def write_fasta(path: str, records: Iterable[Tuple[str, str]], width: int = 70) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(f">{name}\n")
            for i in range(0, len(seq), width):
                handle.write(seq[i:i + width] + "\n")


@dataclass
class Alignment:
    """A DNA multiple sequence alignment."""

    names: List[str]
    masks: np.ndarray  # shape (n_taxa, n_sites), dtype uint8

    @classmethod
    def from_sequences(cls, records: Sequence[Tuple[str, str]]) -> "Alignment":
        if len(records) < 2:
            raise AlignmentError("an alignment needs at least two sequences")
        names = [name for name, _ in records]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise AlignmentError(f"duplicate sequence names: {sorted(duplicates)}")
        lengths = {len(seq) for _, seq in records}
        if len(lengths) != 1:
            detail = ", ".join(f"{n}={len(s)}" for n, s in records[:6])
            raise AlignmentError(
                "sequences have different lengths, so this is not an alignment "
                f"({detail}{', ...' if len(records) > 6 else ''})"
            )
        n_sites = lengths.pop()
        if n_sites == 0:
            raise AlignmentError("sequences are empty")
        masks = np.zeros((len(records), n_sites), dtype=np.uint8)
        for i, (name, seq) in enumerate(records):
            seq = seq.upper()
            bad = sorted(set(seq) - set(IUPAC))
            if bad:
                raise AlignmentError(f"sequence {name!r} contains invalid characters: {bad}")
            masks[i] = [IUPAC[ch] for ch in seq]
        return cls(names, masks)

    @classmethod
    def from_fasta(cls, path: str) -> "Alignment":
        return cls.from_sequences(read_fasta(path))

    @classmethod
    def from_states(cls, names: Sequence[str], states: np.ndarray) -> "Alignment":
        """Build from integer states 0-3 (A, C, G, T); -1 means missing."""
        states = np.asarray(states)
        masks = np.where(states < 0, ALL, np.left_shift(1, np.clip(states, 0, 3))).astype(np.uint8)
        return cls(list(names), masks)

    @property
    def n_taxa(self) -> int:
        return self.masks.shape[0]

    @property
    def n_sites(self) -> int:
        return self.masks.shape[1]

    def sequences(self) -> List[Tuple[str, str]]:
        reverse = {v: k for k, v in IUPAC.items() if k not in "U?-."}
        return [(name, "".join(reverse[int(m)] for m in row)) for name, row in zip(self.names, self.masks)]

    def to_fasta(self, path: str) -> None:
        write_fasta(path, self.sequences())

    def subset(self, names: Sequence[str]) -> "Alignment":
        idx = [self.names.index(n) for n in names]
        return Alignment(list(names), self.masks[idx].copy())

    def site_patterns(self) -> Tuple[np.ndarray, np.ndarray]:
        """Unique alignment columns and how often each occurs.

        Parsimony and likelihood only depend on the column patterns, so
        working on the compressed alignment gives identical results faster.
        """
        patterns, counts = np.unique(self.masks, axis=1, return_counts=True)
        return patterns, counts.astype(float)

    def single_base_index(self) -> np.ndarray:
        """Base index 0-3 per cell, -1 where the cell is ambiguous or missing."""
        out = np.full(self.masks.shape, -1, dtype=np.int8)
        for mask, idx in _MASK_TO_INDEX.items():
            out[self.masks == mask] = idx
        return out
