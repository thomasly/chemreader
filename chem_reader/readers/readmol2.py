import gzip
import logging

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import numpy as np

from ..utils.tools import property_getter


class Mol2Reader:

    def __init__(self, path):
        if path.endswith(".mol2"):
            with open(path, "r") as fh:
                self.file_contents = fh.readlines()
        elif path.endswith(".gz"):
            with gzip.open(path, "r") as fh:
                try:
                    lines = fh.readlines()
                    self.file_contents = [l.decode() for l in lines]
                except OSError:
                    logging.error("{} is not readble by gzip".format(path))
                    self.file_contents = []

    @property
    def n_mols(self):
        try:
            return self._n_mols
        except AttributeError:
            self._n_mols = 0
            for l in self.file_contents:
                if "@<TRIPOS>MOLECULE" in l:
                    self._n_mols += 1
            return self._n_mols

    @property
    @property_getter
    def blocks(self):
        return self._blocks

    def _get_blocks(self):
        r""" Read the blocks in .mol2 file based on @<TRIPOS>MOLECULE label.
        return (list): list of block contents as strings.
        """
        block_starts = [i for i, l in enumerate(self.file_contents)
                        if "@<TRIPOS>MOLECULE" in l]
        blocks = list()
        i = 0
        while i+1 < len(block_starts):
            block = "".join(
                self.file_contents[block_starts[i]: block_starts[i+1]])
            blocks.append(block)
            i += 1
        blocks.append("".join(self.file_contents[block_starts[-1]:]))
        return blocks


class Mol2Block:

    def __init__(self, block):
        r"""
        block (str): a mol2 format string of molecule block starting with
            @<TRIPOS>MOLECULE
        """
        self.block = self._parse(block)

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
    def mol_name(self):
        r""" Name of the molecule
        """
        return self._name

    def _get_mol_name(self):
        return self.block["MOLECULE"][0]

    @property
    @property_getter
    def num_atoms(self):
        r""" Number of atoms in the molecule
        """
        return self._num_atoms

    def _get_num_atoms(self):
        return int(self.block["MOLECULE"][1].split()[0])

    @property
    @property_getter
    def num_bonds(self):
        return self._num_bonds

    def _get_num_bonds(self):
        try:
            return int(self.block["MOLECULE"][1].split()[1])
        except IndexError:  # num_bonds not specified in the file header
            # Get num_bonds from @<TRIPOS>BOND session
            if "BOND" in self.block:
                return len(self.block["BOND"])
            else:
                logging.warning("num_bonds information is not "
                                "available from {}".format(self.name))

    @property
    @property_getter
    def num_subst(self):
        return self._num_subst

    def _get_num_subst(self):
        try:
            return int(self.block["MOLECULE"][1].split()[2])
        except IndexError:
            logging.warning("num_subst information is not "
                            "available for {}".format(self.mol_name))

    @property
    @property_getter
    def num_feat(self):
        return self._num_feat

    def _get_num_feat(self):
        try:
            return int(self.block["MOLECULE"][1].split()[3])
        except IndexError:
            logging.warning("num_feat information is not "
                            "available for {}".format(self.mol_name))

    @property
    @property_getter
    def num_sets(self):
        return self._num_sets

    def _get_num_sets(self):
        try:
            return int(self.block["MOLECULE"][1].split()[4])
        except IndexError:
            logging.warning("num_sets information is not "
                            "available for {}".format(self.mol_name))

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
        coordinates = list()
        for atom in self.block["ATOM"]:
            tokens = atom.split()
            atom_name = tokens[1]
            coors = tuple(map(float, tokens[2:5]))
            coordinates.append(tuple([atom_name, coors]))
        return coordinates

    @property
    @property_getter
    def atom_types(self):
        return self._atom_types

    def _get_atom_types(self):
        atom_types = list()
        for atom in self.block["ATOM"]:
            type_ = atom.split()[5]
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
                logging.warning("{} does not have charge "
                                "information.".format(self.mol_name))
        return charges

    @property
    @property_getter
    def bonds(self):
        return self._bonds

    def _get_bonds(self):
        bonds = list()
        for bond in self.block["BOND"]:
            b = dict()
            tokens = bond.split()
            start = int(tokens[1])
            end = int(tokens[2])
            type_ = tokens[3]
            b["connect"] = tuple([start, end])
            b["type"] = type_
            bonds.append(b)
        return bonds


class Mol2(Mol2Reader):

    def __init__(self, path):
        super().__init__(path)

    def to_smiles(self, isomeric=False):
        r""" Convert the molecules in the file to SMILES strings
        isomeric (bool): False for cannonical, True for isomeric SMILES.
            Default is False.
        return (list): list of SMILES strings. If the molecule is not valid,
            an empty string will be added to the corresponding position in the
            list.
        """
        smiles = list()
        for block in self.blocks:
            mol = Chem.MolFromMol2Block(block)
            if mol is None:
                smiles.append("")
                continue
            smiles.append(Chem.MolToSmiles(mol, isomericSmiles=isomeric))
        return smiles

    def to_graphs(self, sparse=False):
        r""" Convert the molecules to graphs that represented by atom features,
        bond types, and adjacency matrices.
        sparse (bool): if to use sparse format for the adjacency matrix
        """
        # graphs = list()
        # for b in self.blocks:
        #     graph = dict()
        #     block = Mol2Block(b)
        #     block.
