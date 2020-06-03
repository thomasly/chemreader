from abc import ABCMeta, abstractmethod, abstractproperty


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

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception(
                "input {0} not in allowable set{1}:".format(x, allowable_set)
            )
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(x, allowable_set):
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

    def get_atom_features(self, numeric=False, padding=None):
        r""" Get the atom features in the block. The feature contains
        coordinate and atom type for each atom.
        numeric (bool): if True, return the atom type as a number.
        =======================================================================
        return (list): list of tuples. Features are atom type, atom mass, atom
            degree, and atom aromatic
        """
        features = list()
        for i, atom in enumerate(self.rdkit_mol.GetAtoms()):
            feature = list()
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

    def get_bond_features(self, numeric=False):
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
