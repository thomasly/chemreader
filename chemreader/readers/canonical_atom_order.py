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
        old2new = Chem.CanonicalRankAtoms(
            self.mol, includeChirality=True, breakTies=True
        )
        new2old = {o: i for i, o in enumerate(old2new)}
        # build new molecule based on the new atom order
        new_mol = Chem.rdchem.RWMol(Chem.Mol())
        # add Atoms
        for idx in range(len(old2new)):
            new_mol.AddAtom(self.mol.GetAtomWithIdx(new2old[idx]))
        # rebuild Bonds
        bonds = self.mol.GetBonds()
        for b in bonds:
            new_mol.AddBond(
                old2new[b.GetBeginAtomIdx()],
                old2new[b.GetEndAtomIdx()],
                b.GetBondType(),
            )
        # Add conformer (atom 3D positions)
        try:
            old_conformer = self.mol.GetConformer(0)
        except ValueError:
            old_conformer = None
        if old_conformer is not None:
            new_conformer = Chem.Conformer(new_mol.GetNumAtoms())
            for idx in range(len(old2new)):
                pos = old_conformer.GetAtomPosition(new2old[idx])
                new_conformer.SetAtomPosition(idx, pos)
            new_mol.AddConformer(new_conformer)
        return new_mol
