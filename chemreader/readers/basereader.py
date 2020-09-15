from abc import ABCMeta, abstractmethod, abstractproperty
from rdkit.Chem import AllChem
import numpy as np


class _BaseReader(metaclass=ABCMeta):

    # https://github.com/shionhonda/gae-dgl/blob/master/gae_dgl/prepare_data.py
    _avail_atom_types = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "Al",
        "I",
        "B",
        "K",
        "Se",
        "Zn",
        "H",
        "Cu",
        "Mn",
        "unknown",
    ]

    _atom2int = {atom.upper(): idx for idx, atom in enumerate(_avail_atom_types)}

    _bond_types = ["1", "2", "3", "am", "ar", "du", "un"]
    _bond2int = {bond.upper(): idx for idx, bond in enumerate(_bond_types)}

    @classmethod
    def atom_to_num(cls, atom_type):
        return cls._atom2int.get(atom_type.upper(), cls._atom2int["UNKNOWN"])

    @classmethod
    def bond_to_num(cls, bond_type):
        return cls._bond2int.get(bond_type.upper(), cls._bond2int["UN"])

    @staticmethod
    def rebuild_adj(adj, new_idx):
        """ Rebuld adjacency matrix with the new indices.
        Args:
            adj (numpy 2D array or matrix): The adjacency matrix to rebuild.
            new_idx (list of int): The list of new indices of the old nodes. For
                example, an old adjacency matrix with 3 nodes changes its first node
                index to 1 and second node index to 0. The new_idx should be [1, 0, 2].
        Returns:
            numpy 2D array or matrix: The rebuilt adjacency matrix.
        """
        new_idx = {old: new for new, old in enumerate(new_idx)}
        new_adj = np.zeros(adj.shape, dtype=np.int)
        for row in range(adj.shape[0]):
            for col in range(adj.shape[1]):
                if adj[row, col] == 0:
                    continue
                new_r = new_idx[row]
                new_c = new_idx[col]
                new_adj[new_r, new_c] = 1
        return new_adj

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception(
                "input {0} not in allowable set{1}:".format(x, allowable_set)
            )
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    @abstractproperty
    def num_atoms(self):
        """ Number of atoms
        """

    @abstractproperty
    def bonds(self):
        """ Bonds
        """

    @abstractproperty
    def rdkit_mol(self):
        """ RDKit Mol object
        """

    # def rdkit_conformer(self):
    #     """ RDKit Conformer object
    #     """

    @abstractproperty
    def atom_types(self):
        """ Atom types
        """

    @abstractmethod
    def get_adjacency_matrix(self):
        """ Get the adjacency matrix
        """

    @property
    def sorted_atoms(self):
        try:
            return self._sorted_atoms
        except AttributeError:
            self._sortAtoms(self.rdkit_mol.GetAtoms())
            return self._sorted_atoms

    def _sortAtoms(self, atoms):
        def key(atom):
            type_ = self._atom2int.get(atom.GetSymbol(), len(self._atom2int))
            degree = atom.GetDegree()
            idx = atom.GetIdx()
            return (type_, degree, idx)

        self._sorted_atoms = sorted(atoms, key=key)

    def get_atom_features(self, numeric=False, sort_atoms=False, padding=None):
        r""" Get the atom features in the block. The feature contains
        coordinate and atom type for each atom.
        numeric (bool): if True, return the atom type as a number.
        =======================================================================
        return (list): list of tuples. Features are atom type, atom mass, atom
            degree, and atom aromatic
        """
        features = list()
        atoms = self.rdkit_mol.GetAtoms()
        if sort_atoms:
            atoms = self.sorted_atoms
        for i, atom in enumerate(atoms):
            feature = list()
            # the features of an atom includes: atom type, degree, formal charge,
            # hybridization, aromatic, and chirality
            atom_type = self.atom_types[i]
            if numeric:
                atom_type = self.atom_to_num(atom_type)
            feature.append(atom_type)
            feature.append(atom.GetDegree())
            # feature.append(atom.GetImplicitValence())
            feature.append(atom.GetFormalCharge())
            # feature.append(atom.GetNumRadicalElectrons())
            feature.append(int(atom.GetHybridization()))
            feature.append(int(atom.GetIsAromatic()))
            feature.append(int(atom.GetChiralTag()))
            features.append(tuple(feature))
        if padding is not None:
            if padding < len(features):
                raise ValueError(
                    "Padding number should be larger than the feature number."
                    "Got {} < {}".format(padding, len(features))
                )
            pad = (
                [tuple([self.atom_to_num("unknown")] + [0] * (len(features[0]) - 1))]
            ) * (padding - len(features))
            features.extend(pad)
        return features

    def get_bond_features(self, numeric=False, sort_atoms=False):
        r""" Get the bond features/types in the block.
        numeric (bool): if True, return the bond type as a number.
        =======================================================================
        return (list): list of bond types.
        """
        features = dict()
        for bond in self.bonds:
            type_ = bond["type"]
            conn = str(bond["connect"][0]) + "-" + str(bond["connect"][1])
            conn2 = str(bond["connect"][1]) + "-" + str(bond["connect"][0])
            if numeric:
                type_ = self.bond_to_num(type_)
            features[conn] = type_
            features[conn2] = type_
        return features

    @abstractmethod
    def to_graph(self):
        """ Convert molecule to graph
        """
