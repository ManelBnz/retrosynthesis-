"""
Generates the result images to show (figures/ folder).

  figures/scaling.png       : nodes expanded vs depth (log scale)
  figures/comparison.png    : nodes expanded per algorithm on the two drugs
  figures/tree_*.png        : the synthesis route found, as a tree

Run: python figures.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # no on-screen display, we just save files
import matplotlib.pyplot as plt

from benchmark import measure
from reactions import CATALOG, name
from retro_problem import RetrosynthesisProblem
from search import astar, ida_star

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
TARGETS = {
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen":   "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
}


def figure_scaling():
    """The plot that shows the exponential gap between Dijkstra and A*/IDA*."""
    data = measure(range(2, 16))
    depths = [d for d, *_ in data]
    dij = [x[1] for x in data]
    ast = [x[2] for x in data]
    ida = [x[3] for x in data]

    plt.figure(figsize=(8, 5))
    plt.semilogy(depths, dij, "o-", label="Dijkstra (h = 0)", color="#c0392b")
    plt.semilogy(depths, ast, "s-", label="A* (heuristic)", color="#27ae60")
    plt.semilogy(depths, ida, "^--", label="IDA*", color="#2980b9")
    plt.xlabel("synthesis depth")
    plt.ylabel("nodes expanded (log scale)")
    plt.title("Scaling: the heuristic avoids the combinatorial explosion")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTDIR, "scaling.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print("  wrote", path)


def figure_comparison():
    """Bars: nodes expanded per algorithm on the real targets."""
    algos = ["Dijkstra", "A* (hsum)", "IDA*"]
    colors = {"Paracetamol": "#4c72b0", "Aspirin": "#dd8452", "Ibuprofen": "#55a868"}
    values = {label: [] for label in TARGETS}

    for label, smiles in TARGETS.items():
        d = astar(RetrosynthesisProblem(smiles, "h0"), use_heuristic=False)
        a = astar(RetrosynthesisProblem(smiles, "hsum"), use_heuristic=True)
        i = ida_star(RetrosynthesisProblem(smiles, "hsum"))
        values[label] = [d.nodes_expanded, a.nodes_expanded, i.nodes_expanded]

    x = range(len(algos))
    n = len(values)
    width = 0.8 / n
    plt.figure(figsize=(8, 5))
    for k, (label, vals) in enumerate(values.items()):
        pos = [xi + (k - (n - 1) / 2) * width for xi in x]
        bars = plt.bar(pos, vals, width, label=label,
                       color=colors[label], edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
                     ha="center", va="bottom", fontsize=9)

    plt.xticks(list(x), algos)
    plt.ylabel("nodes expanded")
    plt.title("Search effort on the real targets (less is better)")
    plt.legend(title="target")
    plt.tight_layout()
    path = os.path.join(OUTDIR, "comparison.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print("  wrote", path)


def _positions(root, children):
    """Small tree layout: leaves spread out, parents centered over their children."""
    pos = {}
    counter = [0]

    def place(node, depth):
        kids = children.get(node, [])
        if not kids:
            x = counter[0]
            counter[0] += 1
        else:
            xs = [place(c, depth + 1) for c in kids]
            x = sum(xs) / len(xs)
        pos[node] = (x, -depth)
        return x

    place(root, 0)
    return pos


def figure_tree(label, smiles):
    """Draw the synthesis route found, as a retrosynthetic tree."""
    res = astar(RetrosynthesisProblem(smiles, "hsum"), use_heuristic=True)
    by_product = {r.product: r for r in res.plan}

    children = {}
    labels = {}
    colors = {}

    def explore(mol):
        labels[mol] = name(mol).replace(" (target)", "")
        if mol in CATALOG:
            colors[mol] = "#a9dfbf"   # light green: we buy it
            return
        if mol == smiles:
            colors[mol] = "#f5b7b1"   # light red: the target
        else:
            colors[mol] = "#aed6f1"   # light blue: intermediate
        r = by_product.get(mol)
        if r:
            children[mol] = list(r.precursors)
            for p in r.precursors:
                explore(p)

    explore(smiles)
    pos = _positions(smiles, children)

    plt.figure(figsize=(9, 5.5))
    # the arrows (from product to its precursors) with the reaction name
    for parent, kids in children.items():
        xp, yp = pos[parent]
        reaction_name = by_product[parent].name
        for c in kids:
            xc, yc = pos[c]
            plt.annotate("", xy=(xc, yc + 0.18), xytext=(xp, yp - 0.18),
                         arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.4))
        plt.text(xp + 0.08, yp - 0.5, reaction_name, fontsize=8,
                 style="italic", color="#555")

    # the molecules
    for mol, (x, y) in pos.items():
        plt.scatter([x], [y], s=2600, c=colors[mol],
                    edgecolors="black", zorder=3)
        plt.text(x, y, labels[mol], ha="center", va="center",
                 fontsize=8, zorder=4, wrap=True)

    plt.title(f"Synthesis route found by A*: {label} ({int(res.cost)} reactions)")
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTDIR, f"tree_{label.lower()}.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print("  wrote", path)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Generating the figures:")
    figure_scaling()
    figure_comparison()
    for label, smiles in TARGETS.items():
        figure_tree(label, smiles)
    print("Done. The images are in the figures/ folder.")


if __name__ == "__main__":
    main()
