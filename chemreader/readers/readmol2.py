import gzip
import logging

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import numpy as np
from scipy import sparse as sp
from tqdm import tqdm

from ..utils.tools import property_getter
from .basereader import GraphFromRDKitMol


class Mol2Reader:
    """ Read .mol2 file and extract molecules in the file as blocks.
    """

    def __init__(self, path):
        """
        Args:
            path (str): path to the file
        """
        if path.endswith(".mol2"):
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
                if "@<TRIPOS>MOLECULE" in line:
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
        r""" Read the blocks in .mol2 file based on @<TRIPOS>MOLECULE label.
        return (list): list of block contents as strings.
        """
        if self.file_contents is None:
            return []
        block_starts = [
            i for i, l in enumerate(self.file_contents) if "@<TRIPOS>MOLECULE" in l
        ]
        if len(block_starts) == 0:
            return []
        blocks = list()
        i = 0
        while i + 1 < len(block_starts):
            block = "".join(self.file_contents[block_starts[i] : block_starts[i + 1]])
            blocks.append(block)
            i += 1
        blocks.append("".join(self.file_contents[block_starts[-1] :]))
        return blocks


class Mol2Block(GraphFromRDKitMol):
    def __init__(self, block):
        r"""
        block (str): a mol2 format string of molecule block starting with
            @<TRIPOS>MOLECULE
        """
        self.block_str = block
        mol = Chem.MolFromMol2Block(self.block_str, sanitize=False)
        super().__init__(mol)

    def _parse(self, block):
        r""" Parse the block content and dump the records into a dict
        """
        contents = block.strip().split("\n")
        contents_dict = dict()
        current_key = "<TEMP>"
        contents_dict[current_key] = list()
        for line in contents:
            if line.startswith("@"):
                try:
                    # get record type
                    current_key = line.split("<TRIPOS>")[1]
                    contents_dict[current_key] = list()
                    continue
                except IndexError:  # <TRIPOS> without a name
                    current_key = "<TEMP>"
                    continue
            contents_dict[current_key].append(line.strip())
        # discard the contents without record types
        contents_dict.pop("<TEMP>")
        return contents_dict

    @property
    @property_getter
    def block(self):
        return self._block

    def _get_block(self):
        return self._parse(self.block_str)

    @property
    @property_getter
    def mol_name(self):
        r""" Name of the molecule
        """
        return self._mol_name

    def _get_mol_name(self):
        return self.block["MOLECULE"][0]

    @property
    @property_getter
    def num_subst(self):
        return self._num_subst

    def _get_num_subst(self):
        try:
            return int(self.block["MOLECULE"][1].split()[2])
        except IndexError:
            logging.warning(
                "num_subst information is not " "available for {}".format(self.mol_name)
            )

    @property
    @property_getter
    def num_feat(self):
        return self._num_feat

    def _get_num_feat(self):
        try:
            return int(self.block["MOLECULE"][1].split()[3])
        except IndexError:
            logging.warning(
                "num_feat information is not " "available for {}".format(self.mol_name)
            )

    @property
    @property_getter
    def num_sets(self):
        return self._num_sets

    def _get_num_sets(self):
        try:
            return int(self.block["MOLECULE"][1].split()[4])
        except IndexError:
            logging.warning(
                "num_sets information is not " "available for {}".format(self.mol_name)
            )

    @property
    @property_getter
    def mol_type(self):
        return self._mol_type

    def _get_mol_type(self):
        return self.block["MOLECULE"][2]

    @property
    @property_getter
    def charge_type(self):
        return self._charge_type

    def _get_charge_type(self):
        return self.block["MOLECULE"][3]

    @property
    @property_getter
    def coordinates(self):
        return self._coordinates

    def _get_coordinates(self):
        conformer = self.rdkit_mol.GetConformer()
        return conformer.GetPositions()

    @property
    @property_getter
    def atom_types(self):
        return self._atom_types

    def _get_atom_types(self):
        atom_types = list()
        for atom in self.block["ATOM"]:
            type_ = atom.split()[5].split(".")[0].upper()
            atom_types.append(type_)
        return atom_types

    @property
    @property_getter
    def atom_charges(self):
        return self._atom_charges

    def _get_atom_charges(self):
        charges = list()
        for atom in self.block["ATOM"]:
            try:
                charge = atom.split()[8]
                charges.append(float(charge))
            except IndexError:
                logging.warning(
                    "{} does not have charge " "information.".format(self.mol_name)
                )
        return charges

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

    def get_atom_features(self, numeric=False, padding=None):
        r""" Get the atom features in the block. The feature contains
        coordinate and atom type for each atom.
        numeric (bool): if True, return the atom type as a number.
        =======================================================================
        return (list): list of tuples. The first three numbers in the tuples
            are coordinates and the last string or number is atom type.
        """
        features = list()
        atom_degrees = list()
        atom_aromatic = list()
        atom_masses = list()
        for atom in self.rdkit_mol.GetAtoms():
            atom_degrees.append(atom.GetDegree())
            atom_aromatic.append(int(atom.GetIsAromatic()))
            atom_masses.append(atom.GetMass())
        for coor, typ, mass, deg, aro in zip(
            self.coordinates, self.atom_types, atom_masses, atom_degrees, atom_aromatic
        ):
            if numeric:
                typ = self.atom_to_num(typ)
            features.append((*coor, typ, mass, deg, aro))
        if padding is not None:
            if padding < len(features):
                raise ValueError(
                    "Padding number should be larger than the feature number."
                    "Got {} < {}".format(padding, len(features))
                )
            pad = [(0.0, 0.0, 0.0, self.atom_to_num("ANY"), 0.0, 0, 0)] * (
                padding - len(features)
            )
            features.extend(pad)
        return features

    def to_smiles(self, isomeric=False):
        mol = Chem.RemoveHs(self.rdkit_mol)
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)

    def to_graph(self, sparse=False):
        graph = dict()
        graph["adjacency"] = self.get_adjacency_matrix(sparse=sparse)
        graph["atom_features"] = self.get_atom_features(numeric=True)
        graph["bond_features"] = self.get_bond_features(numeric=True)
        return graph


class Mol2(Mol2Reader):
    def __init__(self, path):
        super().__init__(path)

    @property
    def mol2_blocks(self):
        try:
            return self._mol2_blocks
        except AttributeError:
            self._mol2_blocks = self._get_mol2_blocks()
            return self._mol2_blocks

    def _get_mol2_blocks(self):
        m2blocks = list()
        for block in self.blocks:
            m2blocks.append(Mol2Block(block))
        return m2blocks

    def to_smiles(self, isomeric=False, verbose=0):
        r""" Convert the molecules in the file to SMILES strings
        isomeric (bool): False for cannonical, True for isomeric SMILES.
            Default is False.
        verbose (bool): Set to True to show progress bar.
        return (list): list of SMILES strings. If the molecule is not valid,
            an empty string will be added to the corresponding position in the
            list.
        """
        smiles = list()
        it = tqdm(self.mol2_blocks) if verbose else self.mol2_blocks
        for block in it:
            smiles.append(block.to_smiles(isomeric=isomeric))
        return smiles

    def get_molecular_weights(self):
        r""" Calculate the molecular weights
        return (list): list of molecular weights with the same order in the
            input file
        """
        mw = list()
        for block in self.mol2_blocks:
            mw.append(block.molecular_weight)
        return mw

    def get_adjacency_matrices(self, sparse=False):
        r""" Get adjacency matrices of the molecules as graphs
        sparse (bool): if to use sparse format for the matrices
        =======================================================================
        return (list): list of adjacency matrices with the same order as in the
            input file. The representations of the matrices are numpy arrays or
            numpy sparse matrices if the sparse argument is True.
        """
        matrices = list()
        for block in self.mol2_blocks:
            matrices.append(block.get_adjacency_matrix(sparse=sparse))
        return matrices

    def get_atom_features(self, numeric=False):
        r""" Get atom features (coordinates, atom type)
        numeric (bool): if True, return the atom types as numbers. The
            atoms that are able to be converted to consistant numbers are:
            C, N, O, S, F, Si, P, Cl, Br, Mg, Na, Ca, Fe, Al, I, B, K, Se, Zn,
            H, Cu, Mn. All other atom types will be treated as unknown and
            given a numeric type as 22.
        =======================================================================
        return (list): list of atom features in the same order as the input
            file.
        """
        atom_features = list()
        for block in self.mol2_blocks:
            atom_features.append(block.get_atom_features(numeric=numeric))
        return atom_features

    def get_bond_features(self, numeric=False):
        r""" Get bond features/types
        numeric (bool): if True, returen the bond types as numbers. Convertable
            bond types are "1,2,3,am,ar,du,un,nc". All other bond types will be
            treated as ANY and given a numeric type as 8.
        =======================================================================
        return (list): list of bond features in the same order as the input
            file.
        """
        bond_features = list()
        for block in self.mol2_blocks:
            bond_features.append(block.get_bond_features(numeric=numeric))
        return bond_features

    def to_graphs(self, sparse=False):
        r""" Convert the molecules to graphs that represented by atom features,
        bond types, and adjacency matrices.
        sparse (bool): if to use sparse format for the adjacency matrix
        """
        graphs = list()
        for block in self.mol2_blocks:
            graphs.append(block.to_graph(sparse=sparse))
        return graphs
