from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import numpy as np
from scipy import sparse as sp

from ..utils.tools import property_getter


class Smiles:

    _tripos_atom_types = "C.3,C.2,C.1,C.ar,C.cat,N.3,N.2,N.1,N.ar,N.am,N.pl3,"\
        "N.4,O.3,O.2,O.co2,O.spc,O.t3p,S.3,S.2,S.O,S.O2,P.3,F,Cl,Br,I,H,"\
        "H.spc,H.t3p,LP,Du,Du.C,Hal,Het,Hev,Li,Na,Mg,Al,Si,K,Ca,Cr.th,Cr.oh,"\
        "Mn,Fe,Co.oh,Cu,Zn,Se,Mo,Sn".split(",")
    _atom2int = {atom.upper(): idx for idx,
                 atom in enumerate(_tripos_atom_types)}

    _bond_types = "1,2,3,am,ar,du,un".split(",")
    _bond2int = {bond.upper(): idx for idx, bond in enumerate(_bond_types)}

    def __init__(self, smiles):
        r"""
        smiles (str): smiles string
        """
        self.rdkit_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        self.smiles_str = smiles

    @classmethod
    def atom_to_num(cls, atom_type):
        return cls._atom2int.get(atom_type.upper(), len(cls._atom2int))

    @classmethod
    def bond_to_num(cls, bond_type):
        return cls._bond2int.get(bond_type.upper(), len(cls._bond2int))

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
            if symbol in ["C", "N"]:
                if atom.GetIsAromatic():
                    atom_types.append(symbol+".AR")
                else:
                    degree = atom.GetDegree()
                    atom_types.append(symbol+"."+str(degree))
            elif symbol in ["O", "P", "S"]:
                degree = atom.GetDegree()
                atom_types.append(symbol+"."+str(degree))
            else:
                atom_types.append(symbol)
        return atom_types

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
            b["connect"] = tuple(
                [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
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
                    "Got {} < {}".format(padding, self.num_atoms))
            matrix = np.zeros((padding, padding), dtype=np.int8)
        for bond in self.bonds:
            edge = [c for c in bond["connect"]]
            matrix[edge, edge[::-1]] = 1
        if sparse:
            matrix = sp.csr_matrix(matrix)
        return matrix

    def get_atom_features(self, numeric=False, padding=None):
        r""" Get the atom features in the block. The feature contains
        coordinate and atom type for each atom.
        numeric (bool): if True, return the atom type as a number.
        =======================================================================
        return (list): list of tuples. Features are atom type, atom mass, atom
            degree, and atom aromatic
        """
        features = list()
        atom_degrees = list()
        atom_aromatic = list()
        atom_masses = list()
        for atom in self.rdkit_mol.GetAtoms():
            atom_degrees.append(atom.GetDegree())
            atom_aromatic.append(int(atom.GetIsAromatic()))
            atom_masses.append(atom.GetMass())
        for typ, mass, deg, aro in zip(self.atom_types,
                                       atom_masses,
                                       atom_degrees,
                                       atom_aromatic):
            if numeric:
                typ = self.atom_to_num(typ)
            features.append((typ, mass, deg, aro))
        if padding is not None:
            if padding < len(features):
                raise ValueError(
                    "Padding number should be larger than the feature number."
                    "Got {} < {}".format(padding, len(features)))
            pad = [(self.atom_to_num("ANY"), 0., 0, 0)] * \
                (padding - len(features))
            features.extend(pad)
        return features

    def get_bond_features(self, numeric=False, padding=None):
        r""" Get the bond features/types in the block.
        numeric (bool): if True, return the bond type as a number.
        =======================================================================
        return (list): list of bond types.
        """
        features = list()
        for bond in self.bonds:
            type_ = bond["type"]
            if numeric:
                type_ = self.bond_to_num(type_)
            features.append(type_)
        if padding is not None:
            if padding < len(features):
                raise ValueError(
                    "Padding number should be larger than the feature number."
                    "Got {} < {}".format(padding, len(features)))
            pad = [self.bond_to_num("nc")] * (padding - len(features))
            features.extend(pad)
        return features

    def to_graph(self, sparse=False, pad_atom=None, pad_bond=None):
        graph = dict()
        graph["adjacency"] = self.get_adjacency_matrix(sparse=sparse,
                                                       padding=pad_atom)
        graph["atom_features"] = self.get_atom_features(numeric=True,
                                                        padding=pad_atom)
        graph["bond_features"] = self.get_bond_features(numeric=True,
                                                        padding=pad_bond)
        return graph
