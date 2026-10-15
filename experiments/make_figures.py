"""Make every figure in figures/ from the CSV files in results/.

    python experiments/make_figures.py [--dna results/dna_benchmark] [--lineage results/lineage_benchmark]

The example tree figures are drawn from freshly simulated datasets with
fixed seeds, so they do not depend on the benchmark output.
"""

import argparse
import os
import sys

from phylo.benchmark import read_csv, summarize
from phylo.lineage import LineageConfig, reconstruct, simulate_lineage
from phylo.methods import infer_tree
from phylo.metrics import robinson_foulds
from phylo.plotting import plot_comparison, plot_metric_panels, plot_tradeoff, plot_tree
from phylo.simulate import simulate_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

DNA_AXES = [
    ("n_taxa", "Number of taxa", "number of taxa", False),
    ("n_sites", "Alignment length", "sites (log scale)", True),
    ("tree_height", "Tree height", "root-to-tip subst./site (log)", True),
    ("clock_sigma", "Clock violation", "log-normal rate SD", False),
]
LINEAGE_AXES = [
    ("edit_rate", "Editing rate", "edits per site per generation (log)", True),
    ("n_sites", "Target sites", "number of sites (log)", True),
    ("dropout_rate", "Dropout", "fraction of missing values", False),
    ("outcome_skew", "Allele skew", "power-law exponent", False),
]


def _groups(rows, metric, axes):
    summary = summarize(rows, metric)
    groups = []
    for axis, title, xlabel, log_x in axes:
        if axis not in summary:
            continue
        series = {m: (means, sems) for m, (values, means, sems) in summary[axis].items()}
        x = next(iter(summary[axis].values()))[0]
        groups.append({"title": title, "xlabel": xlabel, "x": x, "series": series, "log_x": log_x})
    return groups


def dna_figures(result_dir: str, fig_dir: str) -> None:
    raw = os.path.join(result_dir, "raw.csv")
    if not os.path.exists(raw):
        print(f"skipping DNA benchmark figures: {raw} not found")
        return
    rows = read_csv(raw)
    for r in rows:
        r["delta_parsimony"] = r["parsimony_score"] - r["true_parsimony_score"]
        r["delta_loglik"] = r["loglik"] - r["true_loglik"]

    plot_metric_panels(_groups(rows, "rf_norm", DNA_AXES), os.path.join(fig_dir, "dna_rf_distance.png"),
                       "normalised RF distance to true tree")
    runtime_axes = [a for a in DNA_AXES if a[0] in ("n_taxa", "n_sites")]
    plot_metric_panels(_groups(rows, "runtime_s", runtime_axes), os.path.join(fig_dir, "dna_runtime.png"),
                       "runtime per tree (s, log scale)", log_y=True)
    crit_axes = [a for a in DNA_AXES if a[0] in ("n_sites", "tree_height")]
    plot_metric_panels(_groups(rows, "delta_parsimony", crit_axes),
                       os.path.join(fig_dir, "dna_delta_parsimony.png"),
                       "parsimony score minus true tree's")
    plot_metric_panels(_groups(rows, "delta_loglik", crit_axes),
                       os.path.join(fig_dir, "dna_delta_loglik.png"),
                       "log-likelihood minus true tree's")

    # runtime/accuracy trade-off over every dataset in the benchmark
    import numpy as np
    points = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        sub = [r for r in rows if r["method"] == method]
        rt = np.array([r["runtime_s"] for r in sub])
        err = np.array([r["rf_norm"] for r in sub])
        n = len(sub)
        points[method] = (rt.mean(), rt.std(ddof=1) / np.sqrt(n), err.mean(), err.std(ddof=1) / np.sqrt(n))
    plot_tradeoff(points, os.path.join(fig_dir, "dna_tradeoff.png"))
    print("DNA benchmark figures written")


def lineage_figures(result_dir: str, fig_dir: str) -> None:
    raw = os.path.join(result_dir, "raw.csv")
    if not os.path.exists(raw):
        print(f"skipping lineage figures: {raw} not found")
        return
    rows = read_csv(raw)
    plot_metric_panels(_groups(rows, "rf_norm", LINEAGE_AXES), os.path.join(fig_dir, "lineage_rf_distance.png"),
                       "normalised RF distance to true tree")
    print("lineage benchmark figures written")


def example_figures(fig_dir: str) -> None:
    # one dataset with a strong clock violation, where UPGMA's assumption fails
    true_tree, aln = simulate_dataset(16, 1000, 0.3, seed=7, clock_sigma=1.0)
    for method in ("upgma", "ml"):
        inferred = infer_tree(aln, method).tree
        rf = robinson_foulds(true_tree, inferred)
        plot_comparison(true_tree, inferred, os.path.join(fig_dir, f"example_true_vs_{method}.png"),
                        "True tree (simulated)", f"{method.upper()} (RF = {rf.distance}/{rf.max_distance})")
    ml_tree = infer_tree(aln, "ml").tree
    plot_tree(ml_tree, os.path.join(fig_dir, "example_ml_tree.png"),
              title="ML tree, 16 simulated taxa, 1000 sites")

    data = simulate_lineage(LineageConfig(), seed=11)
    inferred = reconstruct(data, "parsimony").tree
    rf = robinson_foulds(data.true_tree, inferred)
    plot_comparison(data.true_tree, inferred, os.path.join(fig_dir, "example_lineage_parsimony.png"),
                    "True cell lineage (simulated)", f"Parsimony (RF = {rf.distance}/{rf.max_distance})")
    print("example figures written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dna", default=os.path.join(ROOT, "results", "dna_benchmark"))
    parser.add_argument("--lineage", default=os.path.join(ROOT, "results", "lineage_benchmark"))
    parser.add_argument("--figures", default=os.path.join(ROOT, "figures"))
    parser.add_argument("--skip-examples", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    dna_figures(args.dna, args.figures)
    lineage_figures(args.lineage, args.figures)
    if not args.skip_examples:
        example_figures(args.figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
