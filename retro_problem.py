"""
The retrosynthesis problem seen as a backward state-space search.

A state is the set of molecules still left to make.
We start from the target and stop when everything is in the catalog (empty state).
Each move disconnects one molecule, replacing it by its precursors.

For the heuristic we first compute, for every molecule, the minimum number of
reactions needed to make it (its "level", like in Graphplan).
"""

from __future__ import annotations

from typing import Iterable

from reactions import CATALOG, REACTIONS, Reaction, reactions_for
from search import Problem

State = frozenset  # a state = frozenset of SMILES not in the catalog


def compute_levels(reactions=REACTIONS, catalog=CATALOG) -> dict[str, float]:
    """
    Level of a molecule = minimum number of reactions to reach it.
    Catalog = 0. Otherwise take the best reaction and relax the computation by
    assuming the precursors are independent. We loop until it stabilizes.
    """
    INF = float("inf")
    level = {c: 0.0 for c in catalog}
    for r in reactions:
        level.setdefault(r.product, INF)
        for p in r.precursors:
            level.setdefault(p, INF)

    changed = True
    while changed:
        changed = False
        for r in reactions:
            if any(level.get(p, INF) == INF for p in r.precursors):
                continue
            candidate = r.cost + max(level[p] for p in r.precursors)
            if candidate < level[r.product]:
                level[r.product] = candidate
                changed = True
    return level


class RetrosynthesisProblem(Problem):
    """
    heuristic modes:
      h0   : 0, so A* falls back to Dijkstra
      h1   : number of molecules left to make
      hmax : max of the levels, admissible
      hsum : sum of the levels, admissible here since the subgoals are independent
    """

    MODES = ("h0", "h1", "hmax", "hsum")

    def __init__(self, target: str, mode: str = "hsum",
                 reactions=REACTIONS, catalog=CATALOG):
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.target = target
        self.mode = mode
        self.reactions = reactions
        self.catalog = catalog
        self.levels = compute_levels(reactions, catalog)

    def initial_state(self) -> State:
        return State() if self.target in self.catalog else State({self.target})

    def is_goal(self, state: State) -> bool:
        return len(state) == 0

    def successors(self, state: State) -> Iterable[tuple[Reaction, State, float]]:
        if not state:
            return []
        # we only disconnect one molecule at a time (the smallest), which is enough
        # since they all have to be handled anyway and the state is a set
        m = min(state)
        rest = state - {m}
        out = []
        for r in reactions_for(m, self.reactions):
            new_mols = {p for p in r.precursors if p not in self.catalog}
            out.append((r, State(rest | new_mols), r.cost))
        return out

    def heuristic(self, state: State) -> float:
        INF = float("inf")
        if not state:
            return 0.0
        if self.mode == "h0":
            return 0.0
        if self.mode == "h1":
            return float(len(state))
        levels = [self.levels.get(m, INF) for m in state]
        if self.mode == "hmax":
            return max(levels)
        # hsum
        total = 0.0
        for lvl in levels:
            if lvl == INF:
                return INF  # an unreachable molecule, infinite cost
            total += lvl
        return total
