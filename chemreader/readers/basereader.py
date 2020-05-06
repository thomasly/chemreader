from abc import ABCMeta, abstractmethod, abstractproperty


class _BaseReader(metaclass=ABCMeta):

    _tripos_atom_types = "C.3,C.2,C.1,C.ar,C.cat,N.3,N.2,N.1,N.ar,N.am,N.pl3,"\
        "N.4,O.3,O.2,O.co2,O.spc,O.t3p,S.3,S.2,S.O,S.O2,P.3,F,Cl,Br,I,H,"\
        "H.spc,H.t3p,LP,Du,Du.C,Hal,Het,Hev,Li,Na,Mg,Al,Si,K,Ca,Cr.th,Cr.oh,"\
        "Mn,Fe,Co.oh,Cu,Zn,Se,Mo,Sn".split(",")
    _atom2int = {
        atom.upper(): idx
        for idx, atom in enumerate(_tripos_atom_types)
    }

    _bond_types = "1,2,3,am,ar,du,un".split(",")
    _bond2int = {bond.upper(): idx for idx, bond in enumerate(_bond_types)}

    @classmethod
    def atom_to_num(cls, atom_type):
        return cls._atom2int.get(atom_type.upper(), len(cls._atom2int))

    @classmethod
    def bond_to_num(cls, bond_type):
        return cls._bond2int.get(bond_type.upper(), len(cls._bond2int))

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
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
            feature.append(atom.GetImplicitValence())
            feature.append(atom.GetFormalCharge())
            feature.append(atom.GetNumRadicalElectrons())
            feature.append(int(atom.GetHybridization()))
            feature.append(int(atom.GetIsAromatic()))
            features.append(tuple(feature))
        if padding is not None:
            if padding < len(features):
                raise ValueError(
                    "Padding number should be larger than the feature number."
                    "Got {} < {}".format(padding, len(features)))
            pad = ([
                tuple([self.atom_to_num("ANY")] + [0] * (len(features[0]) - 1))
            ]) * (padding - len(features))
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

    @abstractmethod
    def to_graph(self):
        """ Convert molecule to graph
        """
