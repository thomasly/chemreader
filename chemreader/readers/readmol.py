import gzip
import logging

from rdkit import Chem
import numpy as np
from scipy import sparse as sp

from .basereader import _BaseReader
from ..utils.tools import property_getter


class MolReader:
    """ Read .mol file and extract molecules in the file as blocks.
    """

    def __init__(self, path):
        """
        Args:
            path (str): path to the file
        """
        if path.endswith(".mol"):
            with open(path, "r") as fh:
                self.file_contents = fh.readlines()
        elif path.endswith(".gz"):
            with gzip.open(path, "r") as fh:
                try:
                    lines = fh.readlines()
                    self.file_contents = [line.decode() for line in lines]
                except OSError:
                    logging.error("{} is not readble by gzip".format(path))
                    self.file_contents = None
        else:
            self.file_contents = None

    @property
    def n_mols(self):
        """
        Returns:
            int: Number of molecules in the file
        """
        if self.file_contents is None:
            return 0
        try:
            return self._n_mols
        except AttributeError:
            self._n_mols = 0
            for line in self.file_contents:
                if "M  END" in line:
                    self._n_mols += 1
            return self._n_mols

    @property
    @property_getter
    def blocks(self):
        """
        Returns:
            list: list of block contents as strings.
        """
        return self._blocks

    def _get_blocks(self):
        r""" Read the blocks in .mol2 file based on 'M  END' label.
        return (list): list of block contents as strings.
        """
        if self.file_contents is None:
            return []
        blocks = list()
        block = ""
        for line in self.file_contents:
            if "M  END" in line:
                blocks.append(block + line)
                block = ""
            else:
                block += line
        return blocks


class MolBlock(_BaseReader):
    r""" MDL Molfile block object.

    Args:
        block (str): a mol format string of molecule block ends with
            "M  END"
    """

    def __init__(self, block):
        self.block_str = block

    @property
    def num_atoms(self):
        return self.rdkit_mol.GetNumAtoms()

    @property
    def bonds(self):
        return self.rdkit_mol.GetBonds()

    @property
    def atom_types(self):
        return [atom.GetSymbol() for atom in self.rdkit_mol.GetAtoms()]

    @property
    @property_getter
    def rdkit_mol(self):
        return self._rdkit_mol

    def _get_rdkit_mol(self):
        return Chem.MolFromMolBlock(self.block_str, sanitize=False)

    def get_adjacency_matrix(self, sparse=False, padding=None):
        r""" Get the adjacency matrix of the molecular graph.
        spase (bool): if True, return the matrix in sparse format
        =======================================================================
        return (numpy.array or scipy.sparse.csc_matrix)
        """
        num_atoms = self.rdkit_mol.GetNumAtoms()
        if padding is None:
            matrix = np.zeros((num_atoms, num_atoms), dtype=np.int8)
        else:
            if padding < num_atoms:
                raise ValueError(
                    "Padding number should be larger than the atoms number."
                    "Got {} < {}".format(padding, num_atoms)
                )
            matrix = np.zeros((padding, padding), dtype=np.int8)
        for bond in self.rdkit_mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            matrix[(begin, end)] = 1
            matrix[(end, begin)] = 1
        if sparse:
            matrix = sp.csr_matrix(matrix)
        return matrix

    def get_atom_features(self, numeric=False):
        r""" Get the atom features in the block. The feature contains
        coordinate and atom type for each atom.
        numeric (bool): if True, return the atom type as a number.
        =======================================================================
        return (list): list of tuples. The first three numbers in the tuples
            are coordinates and the last string or number is atom type.
        """
        features = list()
        conformer = self.rdkit_mol.GetConformer(0)
        for atom in self.rdkit_mol.GetAtoms():
            coors = conformer.GetAtomPosition(atom.GetIdx())
            atom_type = atom.GetSymbol()
            if numeric:
                atom_type = self.atom_to_num(atom_type)
            features.append([coors.x, coors.y, coors.z, atom_type])
        return features

    def to_smiles(self, isomeric=False):
        mol = Chem.RemoveHs(self.rdkit_mol)
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)

    def to_graph(self, sparse=False):
        graph = dict()
        graph["adjacency"] = self.get_adjacency_matrix(sparse=sparse)
        graph["atom_features"] = self.get_atom_features(numeric=True)
        return graph
