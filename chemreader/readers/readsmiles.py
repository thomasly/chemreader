from rdkit import Chem

from .basereader import GraphFromRDKitMol


class Smiles(GraphFromRDKitMol):
    def __init__(self, smiles, sanitize=True):
        r"""
        smiles (str): smiles string
        """
        self._smiles_str = smiles
        self.sanitize = sanitize
        mol = Chem.MolFromSmiles(smiles)
        super().__init__(mol)

    @property
    def smiles_str(self):
        return self._smiles_str
