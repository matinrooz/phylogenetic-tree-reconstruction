import math
import os

import numpy as np
import pytest

from phylo.alignment import Alignment, AlignmentError, read_fasta
from phylo.distance import DistanceMatrix, distance_matrix, jc69_distance
from phylo.simulate import simulate_dataset

from conftest import write_fasta


def test_read_fasta_multiline_and_headers(tmp_path):
    path = tmp_path / "x.fasta"
    path.write_text(">seq1 some description\nACGT\nAC\n\n>seq2\nACGTAA\n")
    assert read_fasta(str(path)) == [("seq1", "ACGTAC"), ("seq2", "ACGTAA")]


def test_unequal_lengths_rejected(tmp_path):
    path = write_fasta(tmp_path / "x.fasta", [("a", "ACGT"), ("b", "ACG")])
    with pytest.raises(AlignmentError, match="different lengths"):
        Alignment.from_fasta(path)


@pytest.mark.parametrize("records,msg", [
    ([("a", "ACGX"), ("b", "ACGT")], "invalid characters"),
    ([("a", "ACGT"), ("a", "ACGT")], "duplicate"),
    ([("a", "ACGT")], "at least two"),
    ([("a", ""), ("b", "")], "empty"),
])
def test_invalid_alignments(records, msg):
    with pytest.raises(AlignmentError, match=msg):
        Alignment.from_sequences(records)


def test_sequence_before_header(tmp_path):
    path = tmp_path / "x.fasta"
    path.write_text("ACGT\n>a\nACGT\n")
    with pytest.raises(AlignmentError):
        read_fasta(str(path))


def test_lowercase_rna_and_ambiguity_codes():
    aln = Alignment.from_sequences([("a", "acgu"), ("b", "RYN-")])
    assert list(aln.masks[0]) == [1, 2, 4, 8]
    assert list(aln.masks[1]) == [5, 10, 15, 15]


def test_distance_matrix_symmetric_zero_diagonal():
    _, aln = simulate_dataset(10, 400, 0.4, seed=2)
    dm = distance_matrix(aln)
    assert np.allclose(dm.values, dm.values.T)
    assert np.all(np.diag(dm.values) == 0)
    assert np.all(dm.values >= 0)


def test_jc69_formula_values():
    assert jc69_distance(np.array(0.0)) == 0.0
    p = 0.3
    assert float(jc69_distance(np.array(p))) == pytest.approx(-0.75 * math.log(1 - 4 * p / 3))
    # saturated pairs are capped instead of becoming infinite
    assert float(jc69_distance(np.array(0.75), saturation_cap=5.0)) == 5.0
    assert float(jc69_distance(np.array(0.9), saturation_cap=5.0)) == 5.0


def test_identical_sequences_have_zero_distance():
    aln = Alignment.from_sequences([("a", "ACGTACGT"), ("b", "ACGTACGT"), ("c", "TTTTACGT")])
    dm = distance_matrix(aln)
    assert dm["a", "b"] == 0.0
    assert dm["a", "c"] > 0


def test_missing_data_uses_pairwise_deletion():
    aln = Alignment.from_sequences([("a", "ACGTNN"), ("b", "ACGAAC"), ("c", "AC-TAC")])
    dm = distance_matrix(aln, model="p")
    assert dm["a", "b"] == pytest.approx(1 / 4)  # only 4 shared observed sites, 1 differs
    assert dm["a", "c"] == pytest.approx(0.0)    # 3 comparable sites, all equal


def test_no_overlap_raises():
    aln = Alignment.from_sequences([("a", "ACNN"), ("b", "NNGT"), ("c", "ACGT")])
    with pytest.raises(AlignmentError, match="no comparable sites"):
        distance_matrix(aln)


def test_saturation_is_counted():
    aln = Alignment.from_sequences([("a", "AAAA"), ("b", "CCCC"), ("c", "AACC")])
    dm = distance_matrix(aln)
    assert dm.n_saturated == 1


def test_distance_matrix_validation():
    with pytest.raises(ValueError):
        DistanceMatrix(["a", "b"], np.array([[0, 1], [2, 0]]))
    with pytest.raises(ValueError):
        DistanceMatrix(["a", "b"], np.array([[1, 1], [1, 0]]))


def test_example_files_load(examples_dir):
    for name in ("hand_example.fasta", "missing_data.fasta", "simulated_12taxa.fasta"):
        aln = Alignment.from_fasta(os.path.join(examples_dir, name))
        assert aln.n_taxa >= 4
