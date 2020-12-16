""" Covert molecule atoms to a canonical order based on canonical SMILES from RDKit
"""

from rdkit import Chem


class CanonicalAtomOrderConverter:
    def __init__(self, mol):
        """ Convert the molecule atom order to a canonical order.

        Args:
            mol (rdkit Mol object): The RDKit Mol object with any atom order.
        """
        self._mol = mol

    @property
    def mol(self):
        return self._mol

    def convert(self):
        """ Convert atom order.

        Returns:
            RDKit Mol object: An RDKit Mol object with canonical atom order.
        """
        # Creat canonical order dict
        order = Chem.CanonicalRankAtoms(self.mol, includeChirality=True, breakTies=True)
        order = {o: i for i, o in enumerate(order)}
        # build new molecule based on the new atom order
        new_mol = Chem.rdchem.RWMol(Chem.Mol())
        # add Atoms
        for idx in range(len(order)):
            new_mol.AddAtom(self.mol.GetAtomWithIdx(order[idx]))
        # rebuild Bonds
        bonds = self.mol.GetBonds()
        for b in bonds:
            new_mol.AddBond(
                order[b.GetBeginAtomIdx()], order[b.GetEndAtomIdx()], b.GetBondType()
            )
        # Add conformer (atom 3D positions)
        old_conformer = self.mol.GetConformer(0)
        new_conformer = Chem.Conformer(new_mol.GetNumAtoms())
        for idx in range(len(order)):
            pos = old_conformer.GetAtomPosition(order[idx])
            new_conformer.SetAtomPosition(idx, pos)
        new_mol.AddConformer(new_conformer)
        return new_mol
