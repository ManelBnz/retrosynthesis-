"""
The chemistry knowledge base: the catalog of molecules we can buy, and the
reactions we know how to run (written backwards, this is retrosynthesis).

A molecule is just its SMILES string (ethanol = "CCO", benzene = "c1ccccc1"),
so a state is plain text, easy to handle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reaction:
    """A backward reaction: from the product we recover its precursors."""
    name: str
    product: str
    precursors: tuple[str, ...]
    cost: float = 1.0

    def __str__(self) -> str:
        left = " + ".join(self.precursors)
        return f"{self.product}  [{self.name}]  gives  {left}"


# human-readable names for display, otherwise we keep the raw SMILES
NAMES = {
    "CC(=O)Nc1ccc(O)cc1": "Paracetamol (target)",
    "CC(=O)Oc1ccccc1C(=O)O": "Aspirin (target)",
    "Nc1ccc(O)cc1": "4-aminophenol",
    "O=[N+]([O-])c1ccc(O)cc1": "4-nitrophenol",
    "O=C(O)c1ccccc1O": "salicylic acid",
    "Oc1ccccc1": "phenol",
    "CC(=O)OC(C)=O": "acetic anhydride",
    "CC(=O)O": "acetic acid",
    "O=[N+]([O-])c1ccccc1": "nitrobenzene",
    "c1ccccc1": "benzene",
    "DEADEND_A": "unreachable intermediate (trap)",
    "DEADEND_B": "unreachable intermediate (trap)",
}


def name(smiles: str) -> str:
    return NAMES.get(smiles, smiles)


# molecules available off the shelf, these are the states where we stop
CATALOG: frozenset[str] = frozenset({
    "Oc1ccccc1",            # phenol
    "CC(=O)OC(C)=O",        # acetic anhydride
    "CC(=O)O",              # acetic acid
    "O=[N+]([O-])c1ccccc1", # nitrobenzene
    "c1ccccc1",             # benzene
})


# the reactions we know, written backwards.
# I added a couple of trap disconnections (leading to a dead end or to a longer
# route) so the heuristic has something to avoid.
REACTIONS: list[Reaction] = [
    # paracetamol
    Reaction("N-acetylation (Ac2O)", "CC(=O)Nc1ccc(O)cc1",
             ("Nc1ccc(O)cc1", "CC(=O)OC(C)=O")),
    Reaction("N-acetylation (AcOH)", "CC(=O)Nc1ccc(O)cc1",
             ("Nc1ccc(O)cc1", "CC(=O)O")),
    Reaction("non-productive disconnection", "CC(=O)Nc1ccc(O)cc1",
             ("DEADEND_A",)),  # trap

    Reaction("nitro to amine reduction", "Nc1ccc(O)cc1",
             ("O=[N+]([O-])c1ccc(O)cc1",)),

    Reaction("phenol nitration", "O=[N+]([O-])c1ccc(O)cc1",
             ("Oc1ccccc1",)),
    Reaction("nitrobenzene hydroxylation", "O=[N+]([O-])c1ccc(O)cc1",
             ("O=[N+]([O-])c1ccccc1",)),  # another route, also 1 step

    # aspirin
    Reaction("O-acetylation (Ac2O)", "CC(=O)Oc1ccccc1C(=O)O",
             ("O=C(O)c1ccccc1O", "CC(=O)OC(C)=O")),
    Reaction("non-productive disconnection", "CC(=O)Oc1ccccc1C(=O)O",
             ("DEADEND_B", "CC(=O)O")),  # trap
    Reaction("Kolbe-Schmitt", "O=C(O)c1ccccc1O",
             ("Oc1ccccc1",)),
]


def reactions_for(smiles: str, reactions: list[Reaction] = REACTIONS) -> list[Reaction]:
    """All disconnections that apply to a given molecule."""
    return [r for r in reactions if r.product == smiles]


def validate_with_rdkit() -> None:
    """
    If RDKit is installed, check that every SMILES is a real molecule and print
    its molecular formula. Otherwise we skip it, nothing changes.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception:
        print("  (RDKit not installed, skipping the chemical validation)")
        return

    print("  RDKit validation of the molecules:")
    seen = set()
    everything = list(CATALOG) + [r.product for r in REACTIONS] \
        + [p for r in REACTIONS for p in r.precursors]
    for smiles in everything:
        if smiles in seen or smiles.startswith("DEADEND"):
            continue
        seen.add(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"    [invalid] {smiles}")
        else:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            print(f"    [ok] {name(smiles):<22} {smiles:<28} {formula}")
