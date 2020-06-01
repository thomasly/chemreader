from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt
import numpy as np
from scipy import sparse as sp

from ..utils.tools import property_getter
from .basereader import _BaseReader


class Smiles(_BaseReader):
    def __init__(self, smiles, sanitize=True):
        r"""
        smiles (str): smiles string
        """
        self._smiles_str = smiles
        self.sanitize = sanitize

    @property
    def smiles_str(self):
        return self._smiles_str

    @property
    @property_getter
    def rdkit_mol(self):
        return self._rdkit_mol

    def _get_rdkit_mol(self):
        return Chem.MolFromSmiles(self.smiles_str, sanitize=self.sanitize)

    @property
    @property_getter
    def num_atoms(self):
        r""" Number of atoms in the molecule
        """
        return self._num_atoms

    def _get_num_atoms(self):
        return self.rdkit_mol.GetNumAtoms()

    @property
    @property_getter
    def num_bonds(self):
        return self._num_bonds

    def _get_num_bonds(self):
        return self.rdkit_mol.GetNumBonds()

    @property
    @property_getter
    def atom_names(self):
        return self._atom_names

    def _get_atom_names(self):
        atoms = self.rdkit_mol.GetAtoms()
        return [atom.GetSymbol() for atom in atoms]

    @property
    @property_getter
    def atom_types(self):
        return self._atom_types

    def _get_atom_types(self):
        atom_types = list()
        for atom in self.rdkit_mol.GetAtoms():
            symbol = atom.GetSymbol().upper()
            atom_types.append(symbol)
        return atom_types

    @property
    @property_getter
    def fingerprint(self):
        return self._fingerprint

    def _get_fingerprint(self):
        if self.rdkit_mol is None:
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(self.rdkit_mol, 2)
        return fingerprint

    @property
    @property_getter
    def bonds(self):
        return self._bonds

    def _get_bonds(self):
        bonds = list()
        for bond in self.rdkit_mol.GetBonds():
            b = dict()
            if bond.GetIsAromatic():
                type_ = "ar"
            else:
                type_ = str(int(bond.GetBondType()))
            b["connect"] = tuple([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            b["type"] = type_
            bonds.append(b)
        return bonds

    @property
    @property_getter
    def molecular_weight(self):
        return self._molecular_weight

    def _get_molecular_weight(self):
        return ExactMolWt(self.rdkit_mol)

    def get_adjacency_matrix(self, sparse=False, padding=None):
        r""" Get the adjacency matrix of the molecular graph.
        spase (bool): if True, return the matrix in sparse format
        =======================================================================
        return (numpy.array or scipy.sparse.csc_matrix)
        """
        if padding is None:
            matrix = np.zeros((self.num_atoms, self.num_atoms), dtype=np.int8)
        else:
            if padding < self.num_atoms:
                raise ValueError(
                    "Padding number should be larger than the atoms number."
                    "Got {} < {}".format(padding, self.num_atoms)
                )
            matrix = np.zeros((padding, padding), dtype=np.int8)
        for bond in self.bonds:
            edge = [c for c in bond["connect"]]
            matrix[edge, edge[::-1]] = 1
        if sparse:
            matrix = sp.csr_matrix(matrix)
        return matrix

    def to_graph(self, sparse=False, pad_atom=None, pad_bond=None):
        graph = dict()
        graph["adjacency"] = self.get_adjacency_matrix(sparse=sparse, padding=pad_atom)
        graph["atom_features"] = self.get_atom_features(numeric=True, padding=pad_atom)
        graph["bond_features"] = self.get_bond_features(numeric=True, padding=pad_bond)
        return graph

    def similar_to(self, other, threshold=0.5):
        sim = DataStructs.FingerprintSimilarity(self.fingerprint, other.fingerprint)
        if sim > threshold:
            return True
        return False
