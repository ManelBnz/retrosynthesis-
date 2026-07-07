"""
Generates the result images (figures/ folder).

  figures/scaling.png       : nodes expanded vs depth (log scale)
  figures/comparison.png    : nodes expanded per algorithm on the drugs
  figures/tree_*.png        : the synthesis route, drawn as a chemistry scheme

The synthesis trees show the real molecular structures (drawn from the SMILES
with RDKit) linked by retrosynthetic arrows. If RDKit is missing we fall back to
a plain text scheme so the script still runs.

Run: python figures.py
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

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


def _build_tree(smiles):
    """From the optimal plan, build children map, depth and leaf positions."""
    res = astar(RetrosynthesisProblem(smiles, "hsum"), use_heuristic=True)
    by_product = {r.product: r for r in res.plan}

    children, reaction_of = {}, {}

    def explore(mol):
        r = by_product.get(mol)
        if mol in CATALOG or r is None:
            children[mol] = []
        else:
            children[mol] = list(r.precursors)
            reaction_of[mol] = r.name
            for p in r.precursors:
                explore(p)

    explore(smiles)
    pos = _positions(smiles, children)
    return res, children, reaction_of, pos


def _positions(root, children):
    """Tree layout: leaves spread out, parents centered over their children."""
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


def _mol_image(smiles, size=(360, 260)):
    """Render a molecule to a PIL image, with a fixed bond length so every
    molecule is drawn at the same scale (a big one just takes more room)."""
    import io
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.fixedBondLength = 24
    opts.padding = 0.15
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def _retro_arrow(ax, p_from, p_to, off=0.045, head=0.15):
    """Draw a retrosynthetic arrow (double open shaft) from p_from to p_to."""
    p0, p1 = np.array(p_from, float), np.array(p_to, float)
    d = p1 - p0
    length = np.hypot(*d)
    if length == 0:
        return
    u = d / length
    perp = np.array([-u[1], u[0]])
    # the two parallel shafts stop a bit before the tip so the head stays open
    for s in (1, -1):
        a = p0 + s * off * perp
        b = p1 - u * head * 0.55 + s * off * perp
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.1, zorder=3)
    # wide open V-shaped head at the tip
    ang = np.deg2rad(32)
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    for t in (ang, -ang):
        v = rot(t) @ (-u)
        ax.plot([p1[0], p1[0] + v[0] * head], [p1[1], p1[1] + v[1] * head],
                color="black", lw=1.1, zorder=3)


def _figure_tree_rdkit(label, smiles):
    res, children, reaction_of, pos = _build_tree(smiles)

    width = max(x for x, _ in pos.values()) + 1
    depth = -min(y for _, y in pos.values())

    fig, ax = plt.subplots(figsize=(3.6 * width, 3.0 * (depth + 1)), dpi=130)

    # molecule structures
    for mol, (x, y) in pos.items():
        img = _mol_image(mol)
        ab = AnnotationBbox(OffsetImage(np.asarray(img), zoom=0.6), (x, y),
                            frameon=False, zorder=2)
        ax.add_artist(ab)
        tag = name(mol).replace(" (target)", "")
        if mol in CATALOG:
            tag += "  (commercial)"
        ax.text(x, y - 0.32, tag, ha="center", va="top", fontsize=8, color="#333")

    # one retrosynthetic arrow pointing down, precursors joined by a "+"
    for parent, kids in children.items():
        if not kids:
            continue
        xp, yp = pos[parent]
        _retro_arrow(ax, (xp, yp - 0.40), (xp, yp - 0.70))
        ax.text(xp + 0.12, yp - 0.55, reaction_of[parent], fontsize=8,
                style="italic", color="#555", ha="left", va="center")
        kid_pos = sorted(pos[c] for c in kids)
        for (x1, y1), (x2, _) in zip(kid_pos, kid_pos[1:]):
            ax.text((x1 + x2) / 2, y1, "+", fontsize=15, ha="center", va="center")

    ax.set_xlim(-0.7, width - 1 + 0.7)
    ax.set_ylim(-depth - 0.75, 0.75)
    ax.set_title(f"Retrosynthesis of {label}  ({int(res.cost)} steps)", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUTDIR, f"tree_{label.lower()}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  wrote", path)


def _figure_tree_text(label, smiles):
    """Fallback with no RDKit: a plain black-on-white text scheme."""
    res, children, reaction_of, pos = _build_tree(smiles)
    width = max(x for x, _ in pos.values()) + 1
    depth = -min(y for _, y in pos.values())

    fig, ax = plt.subplots(figsize=(3.2 * width, 1.6 * (depth + 1)), dpi=130)
    for parent, kids in children.items():
        if not kids:
            continue
        xp, yp = pos[parent]
        ax.annotate("", xy=(xp, yp - 0.62), xytext=(xp, yp - 0.30),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
        ax.text(xp + 0.08, yp - 0.46, reaction_of[parent], fontsize=8,
                style="italic", color="#555", ha="left", va="center")
        kid_pos = sorted(pos[c] for c in kids)
        for (x1, y1), (x2, _) in zip(kid_pos, kid_pos[1:]):
            ax.text((x1 + x2) / 2, y1, "+", fontsize=13, ha="center", va="center")
    for mol, (x, y) in pos.items():
        tag = name(mol).replace(" (target)", "")
        suffix = "\n(commercial)" if mol in CATALOG else ""
        ax.text(x, y, tag + suffix, ha="center", va="center", fontsize=9,
                family="serif",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    ax.set_xlim(-0.7, width - 1 + 0.7)
    ax.set_ylim(-depth - 0.6, 0.6)
    ax.set_title(f"Retrosynthesis of {label}  ({int(res.cost)} steps)")
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUTDIR, f"tree_{label.lower()}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  wrote", path)


def figure_tree(label, smiles):
    try:
        import rdkit  # noqa: F401
        _figure_tree_rdkit(label, smiles)
    except Exception:
        _figure_tree_text(label, smiles)


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
