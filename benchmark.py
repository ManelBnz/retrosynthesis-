"""
A small benchmark to see how the algorithms scale.

The idea: we build a synthesis tree of depth d. Each molecule can be disconnected
two ways, but only one long route really reaches the catalog, all the other
branches are dead ends.

So Dijkstra (with no heuristic) explores the whole tree (2^d nodes), while A* and
IDA* spot the dead ends (infinite level) and go straight to the goal. That is
exactly the point of a good heuristic, made visible.

This is not real chemistry, just a complexity test.
"""

from __future__ import annotations

from reactions import Reaction
from retro_problem import RetrosynthesisProblem
from search import astar, ida_star


def build_tree(depth: int):
    """
    Return (reactions, catalog, target) for a binary tree of depth d.
    The target is the root "B", the only buyable molecule is the left-most leaf
    (the all-zeros path). Every other leaf is a dead end.
    """
    reactions = []
    for d in range(depth):
        # every node at this depth (paths of 0s and 1s of length d)
        for k in range(2 ** d):
            path = format(k, f"0{d}b") if d else ""
            product = "B" + path
            reactions.append(Reaction("step", product, ("B" + path + "0",)))
            reactions.append(Reaction("step", product, ("B" + path + "1",)))
    target = "B"
    good_leaf = "B" + "0" * depth
    catalog = frozenset({good_leaf})
    return reactions, catalog, target


def measure(depths):
    """For each depth, count the nodes expanded by each algorithm."""
    rows = []
    for d in depths:
        reactions, catalog, target = build_tree(d)
        dij = astar(RetrosynthesisProblem(target, "h0", reactions, catalog),
                    use_heuristic=False)
        ast = astar(RetrosynthesisProblem(target, "hsum", reactions, catalog),
                    use_heuristic=True, name="A*")
        ida = ida_star(RetrosynthesisProblem(target, "hsum", reactions, catalog))
        rows.append((d, dij.nodes_expanded, ast.nodes_expanded, ida.nodes_expanded))
    return rows


if __name__ == "__main__":
    print("depth | Dijkstra | A* | IDA*")
    for d, dij, ast, ida in measure(range(2, 15)):
        print(f"  {d:>2}  | {dij:>8} | {ast:>3} | {ida:>4}")
