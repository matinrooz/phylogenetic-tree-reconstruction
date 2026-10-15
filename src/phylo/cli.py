"""Command-line interface: ``phylo <command> ...`` (or ``python -m phylo``)."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional



from .alignment import Alignment, AlignmentError
from .distance import distance_matrix
from .metrics import robinson_foulds
from .newick import read_newick_file
from .tree import TreeError


def _cmd_infer(args: argparse.Namespace) -> int:
    from .methods import infer_tree

    aln = Alignment.from_fasta(args.input)
    result = infer_tree(aln, args.method, seed=args.seed, parsimony_starts=args.starts)
    # parsimony does not estimate branch lengths, so its tree is written as a
    # topology only (any lengths on it are leftovers from the starting tree)
    newick = result.tree.to_newick(lengths=args.method != "parsimony")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(newick + "\n")
    else:
        print(newick)
    if result.score is not None:
        label = "parsimony score" if args.method == "parsimony" else "log-likelihood"
        print(f"{label}: {result.score:.4f}", file=sys.stderr)
    if args.plot:
        from .plotting import plot_tree

        plot_tree(result.tree, args.plot, title=f"{args.method} tree",
                  use_lengths=args.method != "parsimony")
    return 0


def _cmd_distance(args: argparse.Namespace) -> int:
    aln = Alignment.from_fasta(args.input)
    dm = distance_matrix(aln, model=args.model)
    lines = ["," + ",".join(dm.names)]
    for name, row in zip(dm.names, dm.values):
        lines.append(name + "," + ",".join(f"{v:.6f}" for v in row))
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    if dm.n_saturated:
        print(f"warning: {dm.n_saturated} pair(s) were saturated and capped", file=sys.stderr)
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .simulate import simulate_dataset

    tree, aln = simulate_dataset(args.taxa, args.sites, args.height, args.seed,
                                 clock_sigma=args.clock_sigma)
    aln.to_fasta(args.out_fasta)
    with open(args.out_tree, "w", encoding="utf-8") as handle:
        handle.write(tree.to_newick() + "\n")
    print(f"wrote {args.out_fasta} ({aln.n_taxa} taxa x {aln.n_sites} sites) and {args.out_tree}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    a, b = read_newick_file(args.reference), read_newick_file(args.other)
    rf = robinson_foulds(a, b)
    print(f"RF distance: {rf.distance} of {rf.max_distance} (normalised {rf.normalized:.4f})")
    if args.plot:
        from .plotting import plot_comparison

        plot_comparison(a, b, args.plot, "Reference", "Other")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .methods import CharacterData, fitch_score, log_likelihood, optimize_branch_lengths

    tree = read_newick_file(args.tree)
    aln = Alignment.from_fasta(args.input)
    print(f"parsimony score: {fitch_score(tree, CharacterData.from_alignment(aln)):.0f}")
    print(f"log-likelihood (given branch lengths): {log_likelihood(tree, aln):.4f}")
    print(f"log-likelihood (optimised branch lengths): {optimize_branch_lengths(tree, aln):.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phylo", description="Phylogenetic inference from scratch.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("infer", help="infer a tree from a FASTA alignment")
    p.add_argument("input")
    p.add_argument("--method", choices=["upgma", "nj", "parsimony", "ml"], default="nj")
    p.add_argument("--out", help="write the Newick tree here (default: stdout)")
    p.add_argument("--plot", help="also save a drawing of the tree (PNG/PDF)")
    p.add_argument("--seed", type=int, default=0, help="seed for the parsimony search")
    p.add_argument("--starts", type=int, default=10, help="random addition starts for parsimony")
    p.set_defaults(func=_cmd_infer)

    p = sub.add_parser("distance", help="pairwise distance matrix as CSV")
    p.add_argument("input")
    p.add_argument("--model", choices=["jc69", "p"], default="jc69")
    p.add_argument("--out")
    p.set_defaults(func=_cmd_distance)

    p = sub.add_parser("simulate", help="simulate a tree and a JC69 alignment")
    p.add_argument("--taxa", type=int, default=10)
    p.add_argument("--sites", type=int, default=500)
    p.add_argument("--height", type=float, default=0.3, help="root-to-tip substitutions per site")
    p.add_argument("--clock-sigma", type=float, default=0.0, help="log-normal relaxed clock SD")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-fasta", required=True)
    p.add_argument("--out-tree", required=True)
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("compare", help="Robinson-Foulds distance between two Newick trees")
    p.add_argument("reference")
    p.add_argument("other")
    p.add_argument("--plot")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("score", help="parsimony score and log-likelihood of a given tree")
    p.add_argument("tree")
    p.add_argument("input")
    p.set_defaults(func=_cmd_score)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (AlignmentError, TreeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
