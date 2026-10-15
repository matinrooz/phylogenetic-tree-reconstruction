"""Compare this package against ape/phangorn on simulated datasets.

    python crosscheck/run_crosscheck.py [--rlib PATH]

Steps: simulate a few alignments, run our methods, call
phangorn_reference.R, then compare distances, UPGMA/NJ trees, parsimony
scores and log-likelihoods. The comparison is written to
crosscheck/crosscheck_results.csv and summarised in crosscheck/README.md
(by hand, from that CSV).

R is optional for the project; this script just stops with a message if
Rscript or phangorn is missing.
"""

import argparse
import csv
import itertools
import os
import shutil
import subprocess
import sys

import numpy as np

from phylo.distance import distance_matrix
from phylo.methods import CharacterData, fitch_score, infer_tree, log_likelihood, optimize_branch_lengths
from phylo.metrics import robinson_foulds
from phylo.newick import read_newick_file
from phylo.simulate import simulate_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = [  # (name, taxa, sites, height, clock_sigma, seed)
    ("ds1", 8, 300, 0.2, 0.3, 101),
    ("ds2", 12, 500, 0.4, 0.5, 102),
    ("ds3", 16, 800, 0.3, 1.0, 103),
    ("ds4", 24, 1000, 0.6, 0.5, 104),
    ("ds5", 10, 150, 1.0, 0.5, 105),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlib", default=None, help="extra R library path (sets R_LIBS)")
    parser.add_argument("--work", default=os.path.join(HERE, "work"))
    args = parser.parse_args()
    if shutil.which("Rscript") is None:
        print("Rscript not found; the cross-check needs R with ape and phangorn installed.")
        return 1
    os.makedirs(args.work, exist_ok=True)

    ours = {}
    for name, n, sites, height, sigma, seed in DATASETS:
        true, aln = simulate_dataset(n, sites, height, seed, clock_sigma=sigma)
        aln.to_fasta(os.path.join(args.work, f"{name}.fasta"))
        trees = {"true": true}
        for method in ("upgma", "nj", "parsimony", "ml"):
            trees[method] = infer_tree(aln, method, seed=seed, parsimony_starts=10).tree
        for method, tree in trees.items():
            with open(os.path.join(args.work, f"{name}_py_{method}.nwk"), "w") as handle:
                handle.write(tree.to_newick(precision=12) + "\n")
        ours[name] = (aln, trees)

    env = dict(os.environ)
    if args.rlib:
        env["R_LIBS"] = args.rlib
    proc = subprocess.run(["Rscript", os.path.join(HERE, "phangorn_reference.R"), args.work],
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        print("R step failed (is phangorn installed?)")
        return 1

    with open(os.path.join(args.work, "R_scores.csv")) as handle:
        r_rows = list(csv.DictReader(handle))

    out_rows = []
    for name, (aln, trees) in ours.items():
        chars = CharacterData.from_alignment(aln)
        # distances
        dm = distance_matrix(aln)
        with open(os.path.join(args.work, f"{name}_R_dist.csv")) as handle:
            reader = list(csv.reader(handle))
        r_names = reader[0][1:]
        r_vals = np.array([[float(x) for x in row[1:]] for row in reader[1:]])
        order = [dm.names.index(x) for x in r_names]
        dist_diff = float(np.max(np.abs(dm.values[np.ix_(order, order)] - r_vals)))
        out_rows.append({"dataset": name, "check": "JC69 distance matrix (max abs diff)", "ours": "", "reference": "",
                         "difference": f"{dist_diff:.3g}"})
        # UPGMA / NJ topologies and path lengths
        for method in ("upgma", "nj"):
            r_tree = read_newick_file(os.path.join(args.work, f"{name}_R_{method}.nwk"))
            rf = robinson_foulds(r_tree, trees[method]).distance
            names = trees[method].leaf_names()
            diff = max(abs(r_tree.path_length(a, b) - trees[method].path_length(a, b))
                       for a, b in itertools.combinations(names, 2))
            out_rows.append({"dataset": name, "check": f"{method.upper()} tree vs R (RF; max path-length diff)",
                             "ours": "", "reference": "", "difference": f"RF={rf}; {diff:.3g}"})
        # NJ again, keeping negative branch lengths like ape does
        from phylo.methods import neighbor_joining
        nj_neg = neighbor_joining(dm, allow_negative=True)
        r_tree = read_newick_file(os.path.join(args.work, f"{name}_R_nj.nwk"))
        diff = max(abs(r_tree.path_length(a, b) - nj_neg.path_length(a, b))
                   for a, b in itertools.combinations(nj_neg.leaf_names(), 2))
        out_rows.append({"dataset": name, "check": "NJ allow_negative=True vs R (RF; max path-length diff)",
                         "ours": "", "reference": "",
                         "difference": f"RF={robinson_foulds(r_tree, nj_neg).distance}; {diff:.3g}"})
        # scores of the same trees
        for method in ("true", "upgma", "nj", "parsimony", "ml"):
            ref = next(r for r in r_rows if r["dataset"] == name and r["tree"] == method)
            tree = trees[method].copy()
            for node in tree.branches():
                node.length = max(node.length, 0.0)
            ps = fitch_score(tree, chars)
            ll_given = log_likelihood(tree, aln)
            ll_opt = optimize_branch_lengths(tree.copy(), aln)
            out_rows.append({"dataset": name, "check": f"Fitch score, {method} tree", "ours": f"{ps:.0f}",
                             "reference": ref["R_parsimony"], "difference": f"{ps - float(ref['R_parsimony']):.3g}"})
            out_rows.append({"dataset": name, "check": f"logL given lengths, {method} tree", "ours": f"{ll_given:.6f}",
                             "reference": ref["R_loglik_given"],
                             "difference": f"{ll_given - float(ref['R_loglik_given']):.3g}"})
            out_rows.append({"dataset": name, "check": f"logL optimised lengths, {method} tree", "ours": f"{ll_opt:.4f}",
                             "reference": ref["R_loglik_opt_edges"],
                             "difference": f"{ll_opt - float(ref['R_loglik_opt_edges']):.3g}"})
        # searches
        ref = next(r for r in r_rows if r["dataset"] == name and r["tree"] == "R_search")
        ours_ps = fitch_score(trees["parsimony"], chars)
        ours_ml = optimize_branch_lengths(trees["ml"].copy(), aln)
        r_mp = read_newick_file(os.path.join(args.work, f"{name}_R_parsimony.nwk"))
        r_ml = read_newick_file(os.path.join(args.work, f"{name}_R_ml.nwk"))
        out_rows.append({"dataset": name, "check": "best parsimony score (our search vs pratchet)",
                         "ours": f"{ours_ps:.0f}", "reference": ref["R_parsimony"],
                         "difference": f"{ours_ps - float(ref['R_parsimony']):.3g}; RF={robinson_foulds(r_mp, trees['parsimony']).distance}"})
        out_rows.append({"dataset": name, "check": "best logL (our NNI search vs optim.pml NNI)",
                         "ours": f"{ours_ml:.4f}", "reference": ref["R_loglik_opt_edges"],
                         "difference": f"{ours_ml - float(ref['R_loglik_opt_edges']):.3g}; RF={robinson_foulds(r_ml, trees['ml']).distance}"})

    out_path = os.path.join(HERE, "crosscheck_results.csv")
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "check", "ours", "reference", "difference"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out_path}")
    for row in out_rows:
        print(f"{row['dataset']:4s} {row['check']:55s} {row['ours']:>14s} {row['reference']:>16s}  {row['difference']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
