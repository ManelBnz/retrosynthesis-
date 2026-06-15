"""
Main program: we look for the synthesis route of a few drugs using heuristic
search (Dijkstra, A*, IDA*), then compare the algorithms.

Run: python main.py
"""

from __future__ import annotations

import sys

# so accented characters show up correctly in the Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from reactions import CATALOG, name, validate_with_rdkit
from retro_problem import RetrosynthesisProblem, compute_levels
from search import astar, ida_star


def print_route(target, plan):
    """Print the route found: the target, then what it takes to make it."""
    by_product = {r.product: r for r in plan}
    print("  Synthesis tree:")

    def rec(smiles, depth, seen):
        margin = "    " + "    " * depth
        if smiles in CATALOG:
            print(f"{margin}> {name(smiles)}  (to buy)")
            return
        r = by_product.get(smiles)
        if r is None:
            print(f"{margin}> {name(smiles)}  (?)")
            return
        print(f"{margin}> {name(smiles)}   via {r.name}")
        if smiles in seen:
            return
        for p in r.precursors:
            rec(p, depth + 1, seen | {smiles})

    rec(target, 0, set())

    print("\n  Reactions to run (first to last):")
    for i, r in enumerate(reversed(plan), 1):
        precs = " + ".join(name(p) for p in r.precursors)
        print(f"    {i}. {precs}  gives  {name(r.product)}   ({r.name})")


def compare(target):
    print("=" * 70)
    print(f"Target: {name(target)}   ({target})")
    print("=" * 70)

    pb = RetrosynthesisProblem(target, mode="hsum")
    res = astar(pb, use_heuristic=True, name="A* (hsum)")
    if not res.found:
        print("No route found.\n")
        return

    bound = compute_levels().get(target, float("inf"))
    print(f"\nHeuristic lower bound: {bound:g} reactions")
    print(f"Optimal route found: {int(res.cost)} reactions\n")
    print_route(target, res.plan)

    print("\n  Algorithm comparison (same solution, different effort):")
    runs = [
        astar(RetrosynthesisProblem(target, "h0"), use_heuristic=False, name="Dijkstra"),
        astar(RetrosynthesisProblem(target, "h1"), True, "A* (h=size)"),
        astar(RetrosynthesisProblem(target, "hmax"), True, "A* (hmax)"),
        astar(RetrosynthesisProblem(target, "hsum"), True, "A* (hsum)"),
        ida_star(RetrosynthesisProblem(target, "hsum")),
    ]
    for r in runs:
        print("   ", r)

    costs = {int(r.cost) for r in runs if r.found}
    assert len(costs) == 1, f"the algorithms disagree: {costs}"
    print(f"\n  They all find the same optimal cost ({costs.pop()} reactions).")
    print("  A* and IDA* expand fewer nodes than Dijkstra thanks to the heuristic.\n")


def main():
    print("\nRetrosynthesis by heuristic search\n")

    print("Molecules available off the shelf:")
    for c in sorted(CATALOG):
        print(f"  {name(c):<22} {c}")
    print()

    validate_with_rdkit()
    print()

    for target in ["CC(=O)Nc1ccc(O)cc1",       # paracetamol
                   "CC(=O)Oc1ccccc1C(=O)O"]:   # aspirin
        compare(target)

    print("For the plots and trees as images, run: python figures.py")


if __name__ == "__main__":
    main()
