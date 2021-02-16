import os
from copy import deepcopy
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy import sparse as sp
import networkx as nx

from ..utils.tools import property_getter


class MolFragmentsLabel:
    """ Label atoms in a molecule with the fragments they belong to. The fragment
    library is built from PubChem fingerprint section 3 to section 7. The labels are
    fingerprint like vectors for each atom of the molecule.

    Args:
        ref_file (str): path to the reference file (csv format) that contains the SMARTS
        strings to match molecular fragments.
    """

    ref_smarts = None

    def __init__(self, ref_file=None):
        if ref_file is None:
            cwd = os.path.dirname(__file__)
            self.ref_file = os.path.join(
                cwd, "..", "..", "resources", "pubchemFPKeys_to_SMARTSpattern.csv"
            )
        else:
            self.ref_file = ref_file
        if MolFragmentsLabel.ref_smarts is None:
            self._build_ref(self.ref_file)

    @classmethod
    def _build_ref(cls, ref_file):
        df = pd.read_csv(ref_file)
        cls.ref_smarts = [Chem.MolFromSmarts(smarts) for smarts in df["SMARTS"]]

    def create_labels_for(self, mol, sparse=True):
        """ Create fragment labels for a molecule:

        Args:
            mol (SMILES str or RDKit Mol object): the molecule to create labels for.
            sparse (bool): return the matrix in sparse format. Default: True.
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                raise ValueError(f"{mol} is not a valid SMILES string.")
        # add hydrogens to the molecule
        mol = Chem.AddHs(mol)
        # initiate the vectors
        labels = np.zeros((len(self.ref_smarts), mol.GetNumAtoms()), dtype=np.int)
        # search for the fragments in the molecule
        for i, pattern in enumerate(self.ref_smarts):
            mat_substructs = mol.GetSubstructMatches(pattern)
            # convert tuple of tuples to a set
            mat_atoms = set()
            for atoms in mat_substructs:
                mat_atoms = mat_atoms.union(set(atoms))
            mat_atoms = list(mat_atoms)
            labels[i, mat_atoms] = 1
        if sparse:
            labels = sp.coo_matrix(labels)
        return labels


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
        "Li",
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
        "As",
        "unknown",
    ]

    _atom2int = {atom.upper(): idx for idx, atom in enumerate(_avail_atom_types)}

    _bond_types = ["1", "2", "3", "ar", "un"]
    _bond2int = {bond.upper(): idx for idx, bond in enumerate(_bond_types)}
    # possible formal charges
    possible_fc = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    possible_bond_dirs = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ]

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

    def get_atom_features(
        self, numeric=False, sort_atoms=False, fragment_label=False, padding=None
    ):
        r""" Get the atom features in the block. The feature contains
        coordinate and atom type for each atom.
        Args:
            numeric (bool): if True, return the atom type as a number.
            sort_atoms (bool): Default is False. If True, sort the atoms by atom type.
            padding (None or int): Pad atom feature matrix to a fix length. The number
                must be larger than the number of atoms in the molecules.

        Returns:
            list: list of tuples. Features are atom type, atom mass, atom
                degree, and atom aromatic
        """
        features = list()
        if sort_atoms:
            atoms = self.sorted_atoms
        else:
            atoms = self.rdkit_mol.GetAtoms()
        if fragment_label:
            mfl = MolFragmentsLabel()
            frag_labels = mfl.create_labels_for(self.rdkit_mol, sparse=False)
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
            feature.append(self.possible_fc.index(atom.GetFormalCharge()))
            # feature.append(atom.GetNumRadicalElectrons())
            feature.append(int(atom.GetHybridization()))
            feature.append(int(atom.GetIsAromatic()))
            feature.append(int(atom.GetChiralTag()))
            if fragment_label:
                feature.extend(frag_labels[:, atom.GetIdx()].tolist())
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

    def sort_bonds(self, unsorted_bonds):
        """ Sort bonds based on sorted atoms.
        Args:
            unsorted_bonds (list): list of bonds in chemical compound.

        Returns:
            list: Sorted bonds.
        """
        new_idx = {old.GetIdx(): new for new, old in enumerate(self.sorted_atoms)}
        sorted_bonds = list()
        for bond in unsorted_bonds:
            start, end = bond["connect"]
            new_bond = deepcopy(bond)
            new_bond["connect"] = [0, 0]
            new_bond["connect"][0] = new_idx[start]
            new_bond["connect"][1] = new_idx[end]
            sorted_bonds.append(new_bond)
        return sorted_bonds

    def get_bond_features(self, numeric=False, sort_atoms=False):
        r""" Get the bond features/types in the block.
        numeric (bool): if True, return the bond type as a number.
        =======================================================================
        return (list): list of bond types.
        """
        features = dict()
        if sort_atoms:
            bonds = self.sort_bonds(self.bonds)
        else:
            bonds = self.bonds
        for bond in bonds:
            type_ = bond["type"]
            dir_ = bond["dir"]
            conn = str(bond["connect"][0]) + "-" + str(bond["connect"][1])
            conn2 = str(bond["connect"][1]) + "-" + str(bond["connect"][0])
            if numeric:
                type_ = self.bond_to_num(type_)
                dir_ = self.possible_bond_dirs.index(dir_)
            features[conn] = [type_, dir_]
            features[conn2] = [type_, dir_]
        return features

    @abstractmethod
    def to_graph(self):
        """ Convert molecule to graph
        """

    def graph_to_nx(self, graph):
        G = nx.Graph()
        G.graph["n_atomtypes"] = len(self._avail_atom_types)
        G.graph["n_formalcharges"] = 11
        G.graph["n_degrees"] = 11
        G.graph["n_chiralitytypes"] = 4
        G.graph["n_aromatictypes"] = 2
        G.graph["n_hybridizationtypes"] = 7

        for i, feat in enumerate(graph["atom_features"]):
            G.add_node(
                i,
                atomtype=feat[0],
                degree=feat[1],
                formalcharge=feat[2],
                hybridization=feat[3],
                aromatic=feat[4],
                chirality=feat[5],
            )
        adj = sp.coo_matrix(graph["adjacency"])
        for i, j in zip(adj.row, adj.col):
            bond_features = graph["bond_features"][f"{i}-{j}"]
            G.add_edge(i, j, bondtype=bond_features[0], bonddir=bond_features[1])

        return G

    def graph_to_pyg(self, graph):
        import torch
        from torch_geometric.data import Data

        x = torch.tensor(graph["atom_features"], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_attr = list()
        for i, j in zip(edge_idx.row, edge_idx.col):
            edge_attr.append(graph["bond_features"][f"{i}-{j}"])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        data = Data(x=x, edge_idx=edge_idx, edge_attr=edge_attr)
        return data


class GraphFromRDKitMol(_BaseReader):
    def __init__(self, mol):
        r"""
        Args:
            mol (rdkit Mol object)
        """
        self._rdkit_mol = mol

    @property
    def rdkit_mol(self):
        return self._rdkit_mol

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
            b["dir"] = bond.GetBondDir()
            bonds.append(b)
        return bonds

    @property
    @property_getter
    def molecular_weight(self):
        return self._molecular_weight

    def _get_molecular_weight(self):
        return ExactMolWt(self.rdkit_mol)

    def get_adjacency_matrix(self, sparse=False, sort_atoms=False, padding=None):
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
        if sort_atoms:
            matrix = self.rebuild_adj(matrix, [at.GetIdx() for at in self.sorted_atoms])
        if sparse:
            matrix = sp.csr_matrix(matrix)
        return matrix

    def to_graph(
        self,
        sparse=False,
        sort_atoms=False,
        fragment_label=False,
        pad_atom=None,
        pad_bond=None,
        networkx=False,
        pyg=False,
    ):
        assert not all([networkx, pyg]), "networkx and pyg can't both be True."
        graph = dict()
        # graph adjacency matrix must be sparse if pyg is True
        if pyg:
            sparse = True
        graph["adjacency"] = self.get_adjacency_matrix(
            sparse=sparse, sort_atoms=sort_atoms, padding=pad_atom,
        )
        graph["atom_features"] = self.get_atom_features(
            numeric=True,
            sort_atoms=sort_atoms,
            fragment_label=fragment_label,
            padding=pad_atom,
        )
        graph["bond_features"] = self.get_bond_features(
            numeric=True, sort_atoms=sort_atoms
        )
        if networkx:
            graph = self.graph_to_nx(graph)
        if pyg:
            graph = self.graph_to_pyg(graph)
        return graph

    def similar_to(self, other, threshold=0.5):
        sim = DataStructs.FingerprintSimilarity(self.fingerprint, other.fingerprint)
        if sim > threshold:
            return True
        return False
