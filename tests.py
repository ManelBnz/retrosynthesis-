"""
A few checks, run with: python tests.py

We want to make sure of two things:
  1. all the algorithms find the same cost (so the optimal cost)
  2. the heuristic never overestimates the remaining cost (it is admissible)
"""

from __future__ import annotations

from retro_problem import RetrosynthesisProblem
from search import astar, ida_star

TARGETS = {
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
}
EXPECTED_OPTIMAL = {"Paracetamol": 3, "Aspirin": 2}


def test_optimality():
    for label, smiles in TARGETS.items():
        costs = {astar(RetrosynthesisProblem(smiles, "h0"), False).cost}
        for mode in ("h1", "hmax", "hsum"):
            costs.add(astar(RetrosynthesisProblem(smiles, mode), True).cost)
        costs.add(ida_star(RetrosynthesisProblem(smiles, "hsum")).cost)
        assert len(costs) == 1, f"{label}: different costs {costs}"
        cost = int(costs.pop())
        assert cost == EXPECTED_OPTIMAL[label], \
            f"{label}: cost {cost} instead of {EXPECTED_OPTIMAL[label]}"
        print(f"  ok optimality {label:<12} : {cost} reactions")


def test_admissibility():
    for label, smiles in TARGETS.items():
        optimal = astar(RetrosynthesisProblem(smiles, "hsum"), True).cost
        for mode in ("h1", "hmax", "hsum"):
            pb = RetrosynthesisProblem(smiles, mode)
            h0 = pb.heuristic(pb.initial_state())
            assert h0 <= optimal, f"{label}/{mode}: h={h0} > optimal={optimal}"
            print(f"  ok admissible {label:<12} {mode:<5} : "
                  f"h(start)={h0:g} <= optimal={optimal:g}")


if __name__ == "__main__":
    print("Optimality:")
    test_optimality()
    print("\nAdmissibility:")
    test_admissibility()
    print("\nAll good.")
