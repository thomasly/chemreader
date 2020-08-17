from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolFromPDBBlock
import numpy as np
from scipy.spatial.distance import pdist
from scipy import sparse as sp
from scipy.linalg import toeplitz

from .basereader import _BaseReader
from ..utils.tools import property_getter


class PDB(_BaseReader):
    """ Read PDB file and convert to graph.
    """

    resi_to_int = {
        "ALA": 0,
        "CYS": 1,
        "ASP": 2,
        "GLU": 3,
        "PHE": 4,
        "GLY": 5,
        "HIS": 6,
        "ILE": 7,
        "LYS": 8,
        "LEU": 9,
        "MET": 10,
        "ASN": 11,
        "PRO": 12,
        "GLN": 13,
        "ARG": 14,
        "SER": 15,
        "THR": 16,
        "VAL": 17,
        "TRP": 18,
        "TYR": 19,
        "NAA": 20,  # not a amino acid
    }

    def __init__(self, fpath, sanitize=True):
        r"""
        smiles (str): smiles string
        """
        self._fpath = fpath
        self._sanitize = sanitize

    @classmethod
    def from_pdb_block(cls, block, sanitize=True):
        inst = cls(fpath=None, sanitize=sanitize)
        inst._rdkit_mol = MolFromPDBBlock(block)
        return inst

    @property
    @property_getter
    def num_atoms(self):
        """ Number of atoms
        """
        return self._num_atoms

    def _get_num_atoms(self):
        return self.rdkit_mol.GetNumAtoms()

    @property
    @property_getter
    def bonds(self):
        """ Bonds
        """
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
    def rdkit_mol(self):
        """ RDKit Mol object
        """
        return self._rdkit_mol

    def _get_rdkit_mol(self):
        self._rdkit_mol = MolFromPDBFile(self._fpath, sanitize=self._sanitize)
        return self._rdkit_mol

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

    def get_adjacency_matrix(self, sparse=False, padding=None):
        r""" Get the adjacency matrix of the molecular graph.

        Params:
            spase (bool): if True, return the matrix in sparse format
        =======================================================================
        Return:
            numpy.array or scipy.sparse.csc_matrix
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
        graph["bond_features"] = self.get_bond_features(numeric=True)
        return graph

    def _is_atom(self, pdb_line):
        if pdb_line.startswith("ATOM"):
            return True
        if pdb_line.startswith("HETATM"):
            atom_type = pdb_line[17:20].strip().upper()
            if atom_type == "HOH":
                return False
            return True
        return False

    def get_atom_coordinates(self):
        """ Get atom coordinates from the raw pdb file.
        Return:
            list: list of atom coordinates with the same order as the pdb file.
        """
        atoms = list()
        with open(self._fpath, "r") as f:
            lines = f.readlines()
        for line in lines:
            if not self._is_atom(line):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atoms.append((x, y, z))
        return atoms


class PartialPDB(PDB):
    def __init__(self, fpath, atom_list=None, cutoff=None, sanitize=True):
        super().__init__(fpath, sanitize=sanitize)
        self._atom_list = atom_list
        self._cutoff = cutoff

    @classmethod
    def from_pdb_block(cls, block, atom_list=None, cutoff=None, sanitize=True):
        inst = cls(fpath=None, atom_list=atom_list, cutoff=cutoff, sanitize=sanitize)
        inst._rdkit_mol = MolFromPDBBlock(block)
        return inst

    @property
    def atom_list(self):
        return self._atom_list

    @atom_list.setter
    def atom_list(self, value: list):
        assert isinstance(value, list)
        self._atom_list = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        assert isinstance(value, (int, float))
        self._cutoff = value

    @property
    @property_getter
    def coordinates(self):
        return self._coordinates

    def _get_coordinates(self):
        coor = np.zeros((self.num_atoms, 3))
        conf = self.rdkit_mol.GetConformer()
        for i in range(self.rdkit_mol.GetNumAtoms()):
            coor[i] = conf.GetAtomPosition(i)
        return coor

    def _pairwise_dist(self):
        try:
            n_atoms = len(self.atom_list)
        except TypeError:
            print("atom_list cannot be None. Initialize it with self.atom_list = [...]")
            raise
        dist_array = pdist(self.coordinates[self.atom_list])
        mat = np.zeros((n_atoms - 1, n_atoms - 1))
        i_upper = np.triu_indices(n_atoms - 1)
        mat[i_upper] = dist_array
        mat = np.pad(mat, [[0, 1], [1, 0]])
        i_lower = np.tril_indices(len(mat))
        mat[i_lower] = mat.T[i_lower]
        return mat

    def get_adjacency_matrix(self, sparse=False):
        r""" Get the adjacency matrix of the molecular graph with distance.
        spase (bool): if True, return the matrix in sparse format
        =======================================================================
        return (numpy.array or scipy.sparse.csc_matrix)
        """
        matrix = (self._pairwise_dist() <= self.cutoff).astype(int)
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
        atom_coors = list()
        atom_types = list()
        atom_degrees = list()
        atom_aromatic = list()
        atom_masses = list()
        for atom_id in self.atom_list:
            atom_coors.append(self.coordinates[atom_id])
            atom_types.append(self.atom_types[atom_id])
            atom = self.rdkit_mol.GetAtomWithIdx(atom_id)
            atom_degrees.append(atom.GetDegree())
            atom_aromatic.append(int(atom.GetIsAromatic()))
            atom_masses.append(atom.GetMass())
        for coor, typ, mass, deg, aro in zip(
            atom_coors, atom_types, atom_masses, atom_degrees, atom_aromatic
        ):
            if numeric:
                typ = self.atom_to_num(typ)
            features.append((*coor, typ, mass, deg, aro))
        return features

    def to_graph(self, sparse=False):
        graph = dict()
        graph["adjacency"] = self.get_adjacency_matrix(sparse=sparse)
        graph["atom_features"] = self.get_atom_features(numeric=True)
        return graph


class PDBBB(PartialPDB):
    """ Convert protein backbone to graph from PDB file.
    """

    def __init__(self, fpath, sanitize=True):
        super().__init__(fpath=fpath, atom_list=None, cutoff=None, sanitize=sanitize)
        self._atom_list = self._get_backbone_atoms()

    def _get_backbone_atoms(self):
        atoms = list()
        for atom in self.rdkit_mol.GetAtoms():
            pdbinfo = atom.GetPDBResidueInfo()
            if pdbinfo.GetName().strip().upper() in ["N", "CA", "C"]:
                atoms.append(atom.GetIdx())
        return atoms

    def get_adjacency_matrix(self, sparse=False):
        r""" Get the adjacency matrix of the molecular graph with distance.
        spase (bool): if True, return the matrix in sparse format
        =======================================================================
        return (numpy.array or scipy.sparse.csc_matrix)
        """
        first_row = np.zeros((len(self.atom_list,)))
        first_row[[0, 1]] = 1
        first_col = np.zeros((len(self.atom_list,)))
        first_col[[0, 1]] = 1
        matrix = toeplitz(first_row, first_col)
        if sparse:
            matrix = sp.csr_matrix(matrix)
        return matrix

    def get_atom_features(self, numeric=False):
        r""" Get the atom features in the block. The feature contains
        coordinate and atom type for each atom.
        Params:
            numeric (bool): if True, return the atom type as a number.
        =======================================================================
        Return:
            list of tuples. The first three numbers in the tuples
            are coordinates and the last string or number is atom type.
        """
        features = list()
        conf = self.rdkit_mol.GetConformer()
        for atom_id in self.atom_list:
            feat = list()
            feat += conf.GetAtomPosition(atom_id)
            atom = self.rdkit_mol.GetAtomWithIdx(atom_id)
            pdbinfo = atom.GetPDBResidueInfo()
            try:
                feat.append(self.resi_to_int[pdbinfo.GetResidueName()])
            except KeyError:
                feat.append(self.resi_to_int["NAA"])
            features.append(feat)
        return features
