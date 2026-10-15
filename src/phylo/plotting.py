"""Figures: tree drawings and benchmark plots (matplotlib only)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

import matplotlib

matplotlib.use("Agg")  # file output only; works without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .tree import Node, Tree  # noqa: E402

# colour-blind friendly palette (Okabe-Ito)
METHOD_COLORS = {
    "upgma": "#E69F00",
    "nj": "#56B4E9",
    "parsimony": "#009E73",
    "ml": "#CC79A7",
    "random": "#999999",
}
METHOD_LABELS = {"upgma": "UPGMA", "nj": "Neighbor-joining", "parsimony": "Parsimony",
                 "ml": "Max. likelihood", "random": "Random tree"}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 200, "figure.dpi": 100,
    })


def _ladderize(tree: Tree) -> None:
    """Order children so smaller clades come first (tidier drawings)."""
    size: Dict[int, int] = {}
    for node in tree.postorder():
        size[id(node)] = 1 if node.is_leaf else sum(size[id(c)] for c in node.children)
        node.children.sort(key=lambda c: (size[id(c)], str(c.name)))


def draw_tree(tree: Tree, ax, use_lengths: bool = True, highlight: Optional[Set[frozenset]] = None,
              highlight_color: str = "#D55E00", title: str = "", leaf_order: Optional[List[str]] = None,
              label_size: float = 7.0) -> List[str]:
    """Draw a rectangular cladogram/phylogram on ``ax``.

    ``highlight`` is a set of clades (frozensets of leaf names); the branch
    above any node whose clade is in it is coloured. Returns the leaf order
    used, so a second tree can be drawn with the same order.
    """
    tree = tree.copy()
    if leaf_order is None:
        _ladderize(tree)
    depth: Dict[int, float] = {}
    for node in tree.preorder():
        if node.parent is None:
            depth[id(node)] = 0.0
        else:
            step = node.length if use_lengths else 1.0
            depth[id(node)] = depth[id(node.parent)] + step
    if not use_lengths:  # align leaves on the right
        max_depth = max(depth.values())
        for node in tree.leaves():
            depth[id(node)] = max_depth

    leaves = tree.leaves()
    if leaf_order is not None:
        position = {name: i for i, name in enumerate(leaf_order)}
        y = {id(leaf): float(position[str(leaf.name)]) for leaf in leaves}
    else:
        y = {id(leaf): float(i) for i, leaf in enumerate(leaves)}
    below: Dict[int, frozenset] = {}
    for node in tree.postorder():
        if node.is_leaf:
            below[id(node)] = frozenset([str(node.name)])
        else:
            y[id(node)] = float(np.mean([y[id(c)] for c in node.children]))
            below[id(node)] = frozenset().union(*(below[id(c)] for c in node.children))

    for node in tree.preorder():
        x0 = depth[id(node)]
        if node.parent is not None:
            color = "black"
            if highlight and below[id(node)] in highlight:
                color = highlight_color
            ax.plot([depth[id(node.parent)], x0], [y[id(node)]] * 2, color=color, lw=1.2,
                    solid_capstyle="butt")
        if not node.is_leaf:
            ys = [y[id(c)] for c in node.children]
            ax.plot([x0, x0], [min(ys), max(ys)], color="black", lw=1.2)
        else:
            ax.text(x0, y[id(node)], " " + str(node.name), va="center", fontsize=label_size)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if use_lengths:
        ax.set_xlabel("substitutions per site")
    else:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    max_x = max(depth.values()) or 1.0
    ax.set_xlim(-0.02 * max_x, max_x * 1.25)
    ax.set_ylim(-1, len(leaves))
    if title:
        ax.set_title(title, loc="left")
    order = sorted(leaves, key=lambda leaf: y[id(leaf)])
    return [str(leaf.name) for leaf in order]


def plot_tree(tree: Tree, path: str, title: str = "", use_lengths: bool = True) -> None:
    _style()
    n = tree.n_leaves
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.22 * n + 0.8)))
    draw_tree(tree, ax, use_lengths=use_lengths, title=title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _rooted_clades_as_splits(tree: Tree) -> Dict[frozenset, frozenset]:
    """Map each clade to its normalised unrooted split (for highlighting).

    Clades whose split is trivial (one leaf, or all but one leaf) are left
    out. Every tree has those splits, so they can never be "wrong", and the
    Robinson-Foulds distance ignores them too.
    """
    leaves = frozenset(tree.leaf_names())
    anchor = min(leaves)
    n = len(leaves)
    out = {}
    for clade in tree.clades():
        side = leaves - clade if anchor in clade else clade
        if 2 <= len(side) <= n - 2:
            out[clade] = side
    return out


def conflicting_clades(true_tree: Tree, inferred: Tree) -> tuple:
    """(clades of the true tree the inferred tree missed,
        clades of the inferred tree that are not in the true tree)"""
    true_splits, inf_splits = true_tree.splits(), inferred.splits()
    missed = {c for c, s in _rooted_clades_as_splits(true_tree).items() if s not in inf_splits}
    wrong = {c for c, s in _rooted_clades_as_splits(inferred).items() if s not in true_splits}
    return missed, wrong


def plot_comparison(true_tree: Tree, inferred: Tree, path: str, title_true: str = "True tree",
                    title_inferred: str = "Inferred tree", use_lengths: bool = False) -> None:
    """Side-by-side drawing; branches whose split is missing from the other tree are highlighted."""
    _style()
    missed_in_true, wrong_in_inferred = conflicting_clades(true_tree, inferred)
    n = true_tree.n_leaves
    fig, axes = plt.subplots(1, 2, figsize=(9, max(2.5, 0.22 * n + 1.0)))
    order = draw_tree(true_tree, axes[0], use_lengths=use_lengths, highlight=missed_in_true,
                      highlight_color="#0072B2", title=title_true)
    draw_tree(inferred, axes[1], use_lengths=use_lengths, highlight=wrong_in_inferred,
              title=title_inferred)
    del order
    fig.text(0.5, 0.005, "blue: true splits the inferred tree missed    orange: inferred splits not in the true tree",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_panels(groups: Sequence[dict], path: str, ylabel: str, log_y: bool = False,
                       methods: Optional[Iterable[str]] = None) -> None:
    """One panel per experiment axis: mean +/- standard error per method.

    Each element of ``groups`` is a dict with keys 'title', 'xlabel',
    'x' (list) and 'series' (method -> (means, sems)).
    """
    _style()
    fig, axes = plt.subplots(1, len(groups), figsize=(3.1 * len(groups), 2.8), squeeze=False)
    for ax, group in zip(axes[0], groups):
        x = np.asarray(group["x"], dtype=float)
        for method, (mean, sem) in group["series"].items():
            if methods is not None and method not in methods:
                continue
            mean, sem = np.asarray(mean), np.asarray(sem)
            color = METHOD_COLORS.get(method, None)
            ax.plot(x, mean, marker="o", ms=3.5, lw=1.4, color=color, label=METHOD_LABELS.get(method, method))
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
        ax.set_title(group["title"], loc="left")
        ax.set_xlabel(group["xlabel"])
        if group.get("log_x"):
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(points: Dict[str, tuple], path: str, xlabel: str = "mean runtime per tree (s)",
                  ylabel: str = "mean normalised RF distance") -> None:
    """Scatter of runtime vs. error, one point per method (with SE bars)."""
    _style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for method, (rt, rt_sem, err, err_sem) in points.items():
        color = METHOD_COLORS.get(method)
        ax.errorbar(rt, err, xerr=rt_sem, yerr=err_sem, fmt="o", color=color, ms=6, capsize=2,
                    label=METHOD_LABELS.get(method, method))
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
