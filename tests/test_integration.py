"""End-to-end checks: CLI, benchmark pipeline and figures."""

import csv
import json
import os
import subprocess
import sys

import pytest

from phylo.benchmark import load_config, read_csv, run_dna_benchmark, run_lineage_benchmark, summarize
from phylo.cli import main
from phylo.newick import read_newick_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_full_round(tmp_path, capsys):
    fasta, true_nwk = str(tmp_path / "sim.fasta"), str(tmp_path / "true.nwk")
    assert main(["simulate", "--taxa", "8", "--sites", "400", "--seed", "3",
                 "--out-fasta", fasta, "--out-tree", true_nwk]) == 0
    for method in ("upgma", "nj", "parsimony", "ml"):
        out = str(tmp_path / f"{method}.nwk")
        plot = str(tmp_path / f"{method}.png")
        assert main(["infer", fasta, "--method", method, "--out", out, "--plot", plot, "--starts", "2"]) == 0
        tree = read_newick_file(out)
        assert tree.n_leaves == 8 and os.path.getsize(plot) > 0
        assert main(["compare", true_nwk, out, "--plot", str(tmp_path / f"cmp_{method}.png")]) == 0
    assert main(["distance", fasta, "--out", str(tmp_path / "d.csv")]) == 0
    assert main(["score", true_nwk, fasta]) == 0
    printed = capsys.readouterr().out
    assert "RF distance" in printed and "parsimony score" in printed


def test_cli_reports_bad_input(tmp_path, capsys):
    bad = tmp_path / "bad.fasta"
    bad.write_text(">a\nACGT\n>b\nACG\n")
    assert main(["infer", str(bad)]) == 1
    assert "different lengths" in capsys.readouterr().err


def test_python_dash_m_entry_point(tmp_path):
    out = subprocess.run([sys.executable, "-m", "phylo", "infer",
                          os.path.join(ROOT, "data", "examples", "hand_example.fasta"), "--method", "parsimony"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().endswith(";")
    assert "parsimony score: 4" in out.stderr


def test_dna_benchmark_pipeline(tmp_path):
    config = {"name": "t", "seed": 1, "replicates": 2, "parsimony_starts": 1,
              "methods": ["upgma", "nj", "parsimony", "ml"],
              "base": {"n_taxa": 6, "n_sites": 100, "tree_height": 0.3, "clock_sigma": 0.3},
              "sweeps": {"n_taxa": [5, 6], "tree_height": [0.1]}}
    out = str(tmp_path / "raw.csv")
    rows = run_dna_benchmark(config, out, workers=1, progress=False)
    assert len(rows) == (2 + 1) * 2 * 4
    again = run_dna_benchmark(config, str(tmp_path / "raw2.csv"), workers=1, progress=False)
    key = lambda r: (r["axis"], r["value"], r["replicate"], r["method"])
    for a, b in zip(sorted(rows, key=key), sorted(again, key=key)):
        assert a["rf"] == b["rf"] and a["parsimony_score"] == b["parsimony_score"]
    loaded = read_csv(out)
    assert {r["method"] for r in loaded} == {"upgma", "nj", "parsimony", "ml"}
    for r in loaded:
        assert 0.0 <= r["rf_norm"] <= 1.0
        # the parsimony search can only beat or match the other methods' trees on its own criterion
    summary = summarize(loaded, "rf_norm")
    assert set(summary) == {"n_taxa", "tree_height"}


def test_lineage_benchmark_pipeline(tmp_path):
    config = load_config(os.path.join(ROOT, "experiments", "configs", "lineage_quick.json"))
    config["replicates"] = 1
    rows = run_lineage_benchmark(config, str(tmp_path / "raw.csv"), workers=1, progress=False)
    assert len(rows) == 4 * 4
    assert all(0.0 <= r["rf_norm"] <= 1.0 for r in rows)


def test_configs_are_valid():
    folder = os.path.join(ROOT, "experiments", "configs")
    for name in os.listdir(folder):
        config = load_config(os.path.join(folder, name))
        for axis in config["sweeps"]:
            assert axis in config["base"]


def test_make_figures_script(tmp_path):
    dna_dir, lin_dir, fig_dir = tmp_path / "dna", tmp_path / "lin", tmp_path / "fig"
    config = {"name": "t", "seed": 1, "replicates": 2, "parsimony_starts": 1,
              "methods": ["upgma", "nj", "parsimony", "ml"],
              "base": {"n_taxa": 6, "n_sites": 100, "tree_height": 0.3, "clock_sigma": 0.3},
              "sweeps": {"n_taxa": [5, 6], "n_sites": [50, 100], "tree_height": [0.1, 0.3]}}
    os.makedirs(dna_dir)
    run_dna_benchmark(config, str(dna_dir / "raw.csv"), progress=False)
    lin_cfg = load_config(os.path.join(ROOT, "experiments", "configs", "lineage_quick.json"))
    os.makedirs(lin_dir)
    run_lineage_benchmark(lin_cfg, str(lin_dir / "raw.csv"), progress=False)
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "make_figures.py"),
                    "--dna", str(dna_dir), "--lineage", str(lin_dir), "--figures", str(fig_dir)],
                   check=True, capture_output=True)
    made = set(os.listdir(fig_dir))
    for expected in ("dna_rf_distance.png", "dna_runtime.png", "dna_tradeoff.png",
                     "lineage_rf_distance.png", "example_true_vs_upgma.png", "example_lineage_parsimony.png"):
        assert expected in made


def test_parsimony_tree_written_without_branch_lengths(tmp_path):
    out = str(tmp_path / "mp.nwk")
    assert main(["infer", os.path.join(ROOT, "data", "examples", "hand_example.fasta"),
                 "--method", "parsimony", "--out", out]) == 0
    with open(out) as handle:
        text = handle.read()
    assert ":" not in text
    assert read_newick_file(out).splits() == {frozenset({"C", "D"})}
