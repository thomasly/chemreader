import os
import unittest
import logging

import numpy as np
from scipy import sparse as sp
from rdkit import Chem
from rdkit import RDLogger

from ..readers import Mol2, Mol2Block
from ..readers import Smiles
from ..readers import PDB, PartialPDB, PDBBB
from ..readers import CanonicalAtomOrderConverter
from ..readers.readmol import MolReader, MolBlock
from ..readers.basereader import MolFragmentsLabel

RDLogger.DisableLog("rdApp.*")


class TestReadingMol2File(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_gzip = os.path.join(dir_path, "testing_resources", "AAAARO.xaa.mol2.gz")
        self.mol = Mol2(test_gzip)
        self.blocks = self.mol.blocks
        self.first_block = self.blocks[0]
        test_f = os.path.join(dir_path, "testing_resources", "test_mol2_block")
        with open(test_f, "r") as f:
            self.block = Mol2Block(f.read())

    def test_read_blocks(self):
        self.assertTrue(len(self.blocks) > 0)
        self.assertTrue(isinstance(self.first_block, str))
        self.assertTrue(len(self.first_block) > 0)
        self.assertTrue("@<TRIPOS>" in self.first_block)

    def test_mol_name(self):
        self.assertEqual(self.block.mol_name, "ZINC000005319062")
        self.assertEqual(self.block._mol_name, "ZINC000005319062")

    def test_num_atoms(self):
        self.assertEqual(self.block.num_atoms, 28)
        self.assertEqual(self.block._num_atoms, 28)

    def test_num_bonds(self):
        self.assertEqual(self.block.num_bonds, 28)
        self.assertEqual(self.block._num_bonds, 28)

    def test_num_subst(self):
        self.assertEqual(self.block.num_subst, 1)
        self.assertEqual(self.block._num_subst, 1)

    def test_num_feat(self):
        self.assertEqual(self.block.num_feat, 0)
        self.assertEqual(self.block._num_feat, 0)

    def test_num_sets(self):
        self.assertEqual(self.block.num_sets, 0)
        self.assertEqual(self.block._num_sets, 0)

    def test_mol_type(self):
        self.assertEqual(self.block.mol_type, "SMALL")
        self.assertEqual(self.block._mol_type, "SMALL")

    def test_charge_type(self):
        self.assertEqual(self.block.charge_type, "USER_CHARGES")
        self.assertEqual(self.block._charge_type, "USER_CHARGES")

    def test_atom_names(self):
        self.assertEqual(len(self.block.atom_names), self.block.num_atoms)
        self.assertEqual(self.block.atom_names[0], "C")
        self.assertEqual(self.block.atom_names[-1], "H")
        self.assertTrue(hasattr(self.block, "_atom_names"))

    def test_coordinates(self):
        self.assertEqual(len(self.block.coordinates), self.block.num_atoms)
        self.assertTrue(
            np.array_equal(
                self.block.coordinates[0], np.array((-0.0178, 1.4648, 0.0101))
            )
        )
        self.assertTrue(
            np.array_equal(
                self.block.coordinates[-1], np.array((-1.3009, 0.3246, 7.4554))
            )
        )
        self.assertTrue(hasattr(self.block, "_coordinates"))

    def test_atom_types(self):
        self.assertEqual(len(self.block.atom_types), self.block.num_atoms)
        self.assertEqual(self.block.atom_types[0], "C")
        self.assertEqual(self.block.atom_types[-1], "H")
        self.assertEqual(self.block.atom_types[14], "N")
        self.assertTrue(hasattr(self.block, "_atom_types"))

    def test_atom_charges(self):
        self.assertEqual(len(self.block.atom_charges), self.block.num_atoms)
        self.assertEqual(self.block.atom_charges[0], -0.0600)
        self.assertEqual(self.block.atom_charges[-1], 0.4300)
        self.assertTrue(hasattr(self.block, "_atom_charges"))

    def test_bonds(self):
        self.assertEqual(len(self.block.bonds), self.block.num_bonds)
        self.assertTrue(hasattr(self.block, "_bonds"))
        bond = self.block.bonds[0]
        self.assertEqual(bond["connect"][0], 0)
        self.assertEqual(bond["connect"][1], 1)
        self.assertEqual(bond["type"], "1")
        bond = self.block.bonds[16]
        self.assertEqual(bond["connect"][0], 6)
        self.assertEqual(bond["connect"][1], 7)
        self.assertEqual(bond["type"], "1")


class TestBlockMissingInformation(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_f = os.path.join(
            dir_path, "testing_resources", "test_mol2_block_missing_info"
        )
        with open(test_f, "r") as f:
            self.block = Mol2Block(f.read())

    def test_num_atoms(self):
        self.assertEqual(self.block.num_atoms, 28)
        self.assertEqual(self.block._num_atoms, 28)

    def test_num_subst(self):
        with self.assertLogs() as cm:
            self.block.num_subst
        self.assertIn("num_subst information is not available", cm.output[0])

    def test_num_feat(self):
        with self.assertLogs() as cm:
            self.block.num_feat
        self.assertIn("num_feat information is not available", cm.output[0])

    def test_num_sets(self):
        with self.assertLogs() as cm:
            self.block.num_sets
        self.assertIn("num_sets information is not available", cm.output[0])

    def test_atom_charges(self):
        with self.assertLogs() as cm:
            self.block.atom_charges
        self.assertIn("does not have charge information", cm.output[0])


class TestMol2(TestReadingMol2File):
    def test_mol2_to_smiles(self):
        can_smiles = self.mol.to_smiles()
        iso_smiles = self.mol.to_smiles(isomeric=True)
        self.assertEqual(len(can_smiles), self.mol.n_mols)
        self.assertEqual(can_smiles[0], r"C[NH+](C)CCNC(=O)c1nonc1N")
        self.assertEqual(len(iso_smiles), self.mol.n_mols)
        self.assertEqual(iso_smiles[62], r"Cn1cncc1[C@H]([NH3+])C1(O)CNC1")
        self.assertNotEqual(can_smiles[1], iso_smiles[1])

    def test_molecular_weights(self):
        mol_weights = self.mol.get_molecular_weights()
        self.assertEqual(len(mol_weights), self.mol.n_mols)
        for mw in mol_weights:
            self.assertGreater(mw, 0)
        self.assertEqual(int(mol_weights[0]), 200)

    def test_mol2blocks(self):
        mol2_blocks = self.mol.mol2_blocks
        self.assertEqual(len(mol2_blocks), self.mol.n_mols)
        self.assertTrue(isinstance(mol2_blocks[0], Mol2Block))

    def test_get_adjacency_matrices(self):
        matrices = self.mol.get_adjacency_matrices()
        self.assertEqual(len(matrices), self.mol.n_mols)
        self.assertTrue(isinstance(matrices[0], np.ndarray))
        self.assertEqual(np.sum(matrices[0]), self.mol.mol2_blocks[0].num_bonds * 2)
        self.assertEqual(matrices[0].shape, (28, 28))
        sparse_matrices = self.mol.get_adjacency_matrices(sparse=True)
        self.assertEqual(len(sparse_matrices), self.mol.n_mols)
        self.assertTrue(sp.issparse(sparse_matrices[0]))
        self.assertTrue(np.array_equal(sparse_matrices[0].toarray(), matrices[0]))

    def test_atom2int_and_bond2int(self):
        self.assertEqual(Mol2Block.atom_to_num("C"), 0)
        self.assertEqual(Mol2Block.atom_to_num("Any"), 24)
        self.assertEqual(Mol2Block.atom_to_num("@#$%"), 24)
        self.assertEqual(Mol2Block.bond_to_num("1"), 0)
        self.assertEqual(Mol2Block.bond_to_num("ar"), 3)
        self.assertEqual(Mol2Block.bond_to_num("@#$%"), 4)

    def test_get_atom_features(self):
        atom_features = self.mol.get_atom_features(numeric=False)
        self.assertEqual(len(atom_features), self.mol.n_mols)
        self.assertEqual(len(atom_features[0]), self.block.num_atoms)
        self.assertEqual(len(atom_features[0][0]), 7)
        self.assertTrue(isinstance(atom_features[0][0][0], float))
        self.assertTrue(isinstance(atom_features[0][0][3], str))
        numeric_features = self.mol.get_atom_features(numeric=True)
        self.assertEqual(len(numeric_features), self.mol.n_mols)
        self.assertEqual(len(numeric_features[0]), self.block.num_atoms)
        self.assertEqual(len(numeric_features[0][0]), 7)
        self.assertTrue(isinstance(numeric_features[0][0][0], float))
        self.assertTrue(isinstance(numeric_features[0][0][-1], int))

    def test_get_bond_features(self):
        bond_features = self.mol.get_bond_features(numeric=False)
        self.assertEqual(len(bond_features), self.mol.n_mols)
        self.assertEqual(len(bond_features[0]), self.block.num_bonds * 2)
        self.assertIsInstance(bond_features[0]["1-2"], list)
        self.assertIsInstance(bond_features[0]["1-2"][0], str)
        self.assertEqual(bond_features[0]["1-2"][1], Chem.rdchem.BondDir.NONE)
        numeric_features = self.mol.get_bond_features(numeric=True)
        self.assertEqual(len(numeric_features), self.mol.n_mols)
        self.assertEqual(len(numeric_features[0]), self.block.num_bonds * 2)
        self.assertIsInstance(numeric_features[0]["1-2"], list)
        self.assertIsInstance(numeric_features[0]["1-2"][0], int)
        self.assertIsInstance(numeric_features[0]["1-2"][1], int)

    def test_to_graphs(self):
        graphs = self.mol.to_graphs(sparse=False)
        self.assertEqual(len(graphs), self.mol.n_mols)
        self.assertTrue(isinstance(graphs[0]["adjacency"], np.ndarray))
        self.assertEqual(
            np.sum(graphs[0]["adjacency"]), self.mol.mol2_blocks[0].num_bonds * 2
        )
        self.assertEqual(graphs[0]["adjacency"].shape, (28, 28))
        self.assertEqual(len(graphs[0]["atom_features"]), self.block.num_atoms)
        self.assertEqual(len(graphs[0]["atom_features"][0]), 7)
        self.assertIsInstance(graphs[0]["atom_features"][0][0], float)
        self.assertIsInstance(graphs[0]["atom_features"][0][-1], int)
        self.assertEqual(len(graphs[0]["bond_features"]), self.block.num_bonds * 2)
        self.assertIsInstance(graphs[0]["bond_features"]["1-2"], list)
        self.assertIsInstance(graphs[0]["bond_features"]["1-2"][0], int)
        self.assertIsInstance(graphs[0]["bond_features"]["1-2"][1], int)
        sparse_graphs = self.mol.to_graphs(sparse=True)
        self.assertEqual(len(sparse_graphs), self.mol.n_mols)
        self.assertTrue(sp.issparse(sparse_graphs[0]["adjacency"]))
        self.assertTrue(
            np.array_equal(
                sparse_graphs[0]["adjacency"].toarray(), graphs[0]["adjacency"]
            )
        )
        # graphs = self.mol.to_graphs(sparse=False, pad_atom=70, pad_bond=80)
        # self.assertEqual(graphs[0]["adjacency"].shape, (70, 70))
        # self.assertEqual(len(graphs[0]["atom_features"]), 70)
        # np.array(graphs[0]["atom_features"])
        # self.assertEqual(len(graphs[0]["bond_features"]), 80)
        # np.array(graphs[0]["bond_features"])
        # with self.assertRaises(ValueError):
        #     self.mol.to_graphs(sparse=False, pad_atom=27, pad_bond=80)
        # with self.assertRaises(ValueError):
        #     self.mol.to_graphs(sparse=False, pad_atom=70, pad_bond=27)


class TestReadingSmiles(unittest.TestCase):
    def setUp(self):
        self.smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        self.sm = Smiles(self.smiles)

    def test_building_mol(self):
        self.assertTrue(hasattr(self.sm, "smiles_str"))
        self.assertEqual(self.sm.num_atoms, 13)
        self.assertEqual(self.sm.num_bonds, 13)

    def test_atom_types(self):
        types = self.sm.atom_types
        self.assertEqual(len(types), 13)
        self.assertEqual(types[0], "C")
        self.assertEqual(types[3], "O")
        self.assertEqual(types[1], "C")

    def test_bonds(self):
        bonds = self.sm.bonds
        self.assertEqual(len(bonds), 13)
        self.assertEqual(bonds[0]["type"], "1")
        self.assertEqual(bonds[0]["connect"], (0, 1))
        self.assertEqual(bonds[4]["type"], "ar")
        self.assertEqual(bonds[4]["connect"], (4, 5))

    def test_atom_featurs(self):
        feats = self.sm.get_atom_features()
        self.assertEqual(len(feats), 13)
        self.assertEqual(feats[0], ("C", 1, 5, 4, 0, 0))
        self.assertEqual(feats[4], ("C", 3, 5, 3, 1, 0))
        feats = self.sm.get_atom_features(numeric=True)
        self.assertEqual(len(feats), 13)
        self.assertEqual(feats[0], (0, 1, 5, 4, 0, 0))
        feats = self.sm.get_atom_features(numeric=True, padding=20)
        self.assertEqual(len(feats), 20)
        self.assertEqual(feats[0], (0, 1, 5, 4, 0, 0))
        self.assertEqual(feats[-1], (24, 0, 0, 0, 0, 0))
        with self.assertRaises(ValueError):
            self.sm.get_atom_features(padding=12)

    # def _sorted_correctly(self, atoms):
    #     """ To confirm the atoms are sorted correctly: same atoms are grouped together
    #     and sorted by their x coordinates.
    #     """

    def test_sorted_atoms(self):
        unsorted_atoms = self.sm.rdkit_mol.GetAtoms()
        # >>> [C, C, O, O, C, C, C, C, C, C, C, O, O]
        sorted_atoms = self.sm.sorted_atoms
        correct_sorted = ["C"] * 9 + ["O"] * 4
        for at, coat in zip(sorted_atoms, correct_sorted):
            self.assertEqual(at.GetSymbol(), coat)
        unsorted_indices = [at.GetIdx() for at in unsorted_atoms]
        sorted_indices = [at.GetIdx() for at in sorted_atoms]
        logging.debug("unsorted:" + str(unsorted_indices))
        logging.debug("sorted:" + str(sorted_indices))
        logging.debug("\n")
        unsorted_adj = self.sm.get_adjacency_matrix(sort_atoms=False)
        sorted_adj = self.sm.get_adjacency_matrix(sort_atoms=True)
        logging.debug("unsorted_adj:\n" + str(unsorted_adj))
        logging.debug("\n")
        logging.debug("sorted_adj:\n" + str(sorted_adj))

    def test_bond_features(self):
        feats = self.sm.get_bond_features()
        self.assertEqual(len(feats), 26)
        self.assertEqual(feats["0-1"][0], "1")
        feats = self.sm.get_bond_features(numeric=True)
        self.assertEqual(len(feats), 26)
        self.assertEqual(feats["0-1"][0], 0)
        # feats = self.sm.get_bond_features(numeric=True, padding=15)
        # self.assertEqual(len(feats), 15)
        # self.assertEqual(feats[0], 0)
        # self.assertEqual(feats[-1], 6)
        # with self.assertRaises(ValueError):
        #     self.sm.get_bond_features(padding=12)

    def test_sorted_bond_features(self):
        feats = self.sm.get_bond_features(sort_atoms=True)
        self.assertEqual(len(feats), 26)
        self.assertEqual(feats["0-5"][0], "1")
        with self.assertRaises(KeyError):
            feats["0-1"]
        unsorted_feats = self.sm.get_bond_features(sort_atoms=False)
        self.assertEqual(len(unsorted_feats), 26)
        self.assertEqual(unsorted_feats["0-1"][0], "1")

    def test_graph(self):
        graph = self.sm.to_graph()
        self.assertEqual(len(graph), 3)
        self.assertEqual(graph["adjacency"].shape, (13, 13))
        self.assertIsInstance(graph["adjacency"], np.ndarray)
        self.assertEqual(len(graph["atom_features"]), 13)
        self.assertEqual(len(graph["bond_features"]), 26)
        # graph = self.sm.to_graph(sparse=True, pad_atom=20, pad_bond=15)
        # self.assertIsInstance(graph["adjacency"], sp.csr_matrix)
        # self.assertEqual(graph["adjacency"].shape, (20, 20))
        # self.assertEqual(len(graph["atom_features"]), 20)
        # self.assertEqual(len(graph["bond_features"]), 15)

    def test_fingerprints(self):
        fp = self.sm.fingerprint
        self.assertEqual(len(fp), 2048)
        other = Smiles("c1ccccc1")
        self.assertFalse(self.sm.similar_to(other))
        self.assertTrue(self.sm.similar_to(self.sm))
        fp = Smiles("C1").fingerprint
        self.assertIsNone(fp)

    def test_fragment_labels(self):
        atom_features = self.sm.get_atom_features(numeric=True, fragment_label=True)
        self.assertEqual(len(atom_features[0]), 624)
        graph = self.sm.to_graph(fragment_label=True)
        self.assertEqual(len(graph["atom_features"][0]), 624)
        # assert the order of the atom fragment labels are correct
        atom_features = self.sm.get_atom_features(
            numeric=True, fragment_label=True, sort_atoms=True
        )
        mfl = MolFragmentsLabel()
        frag_labels = mfl.create_labels_for(self.sm.rdkit_mol, sparse=False)
        for i, atom in enumerate(self.sm.sorted_atoms):
            idx = atom.GetIdx()
            self.assertEqual(atom_features[i][6:], tuple(frag_labels[:, idx].tolist()))
        # assert padding still work
        atom_features = self.sm.get_atom_features(
            numeric=True, fragment_label=True, padding=70
        )
        self.assertEqual(len(atom_features), 70)
        self.assertEqual(atom_features[-1], tuple([24] + [0] * 623))

    def test_networkx_graph(self):
        graph = self.sm.to_graph(networkx=True)
        self.assertEqual(graph.graph["n_atomtypes"], 25)
        self.assertEqual(graph.nodes[0]["atomtype"], 0)
        self.assertEqual(graph.nodes[0]["formalcharge"], 5)
        self.assertEqual(graph.nodes[0]["degree"], 1)
        self.assertEqual(graph.nodes[0]["hybridization"], 4)
        self.assertEqual(graph.nodes[0]["aromatic"], 0)
        self.assertEqual(graph.nodes[0]["chirality"], 0)
        self.assertEqual(graph.edges[0, 1]["bondtype"], 0)
        self.assertEqual(graph.edges[0, 1]["bonddir"], 0)

    def test_pyg_graph(self):
        try:
            import torch
            import torch_geometric as pyg

            torch_avail = True
        except ImportError:
            torch_avail = False

        if torch_avail:
            with self.assertRaises(AssertionError):
                self.sm.to_graph(networkx=True, pyg=True)
            graph = self.sm.to_graph(pyg=True)
            self.assertIsInstance(graph, pyg.data.Data)
            self.assertTrue(hasattr(graph, "x"))
            self.assertTrue(hasattr(graph, "edge_idx"))
            self.assertTrue(hasattr(graph, "edge_attr"))
            self.assertIsInstance(graph["edge_idx"], torch.Tensor)
            self.assertIsInstance(graph["edge_attr"], torch.Tensor)
            self.assertEqual(graph["edge_idx"].size(1), graph["edge_attr"].size(0))


class TestReadPDB(unittest.TestCase):
    def setUp(self):
        self.fpath = os.path.join(
            "chemreader", "tests", "testing_resources", "3CQW.pdb"
        )

    def test_mol_from_pdb_file(self):
        pdb = PDB(self.fpath)
        graph = pdb.to_graph()
        self.assertIn("adjacency", graph)
        self.assertIn("atom_features", graph)
        self.assertIn("bond_features", graph)

    def test_partial_pdb(self):
        al = [0, 1, 2, 3, 10]
        part_pdb = PartialPDB(self.fpath, atom_list=al)
        dist_mat = part_pdb._pairwise_dist()
        self.assertEqual(dist_mat.shape, (len(al), len(al)))
        part_pdb.cutoff = 1.5
        adj = part_pdb.get_adjacency_matrix()
        self.assertEqual(adj.shape, (len(al), len(al)))
        self.assertEqual(adj[0, 0], 1)
        self.assertEqual(adj[2, 1], 0)
        graph = part_pdb.to_graph()
        self.assertIn("adjacency", graph)
        self.assertIn("atom_features", graph)
        self.assertEqual(len(graph["atom_features"]), len(al))
        self.assertNotIn("bond_features", graph)
        conformer = part_pdb.rdkit_mol.GetConformer()
        self.assertEqual(
            graph["atom_features"][-1][:3], tuple(conformer.GetAtomPosition(10))
        )

    def test_coordinates_included_in_atom_features(self):
        pdb = PDB(self.fpath)
        feats = pdb.get_atom_features(include_coordinates=True)
        self.assertEqual(len(feats[0]), 9)
        graph = pdb.to_graph(include_coordinates=True)
        self.assertEqual(len(graph["atom_features"][0]), 9)

    def test_coordinates(self):
        pdb = PDB(self.fpath, sanitize=False)
        counter = 0
        for atom in pdb.rdkit_mol.GetAtoms():
            counter += 1
        atoms = pdb.get_atom_coordinates()
        self.assertIsInstance(atoms, list)
        # self.assertEqual(len(atoms), pdb.num_atoms)

    def test_backbone_pdb(self):
        pdb = PDBBB(self.fpath, sanitize=False)
        adj = pdb.get_adjacency_matrix()
        self.assertTrue(np.array_equal(adj.diagonal(), np.ones(len(pdb.atom_list),)))
        self.assertTrue(
            np.array_equal(adj.diagonal(offset=1), np.ones(len(pdb.atom_list) - 1,))
        )
        self.assertTrue(
            np.array_equal(adj.diagonal(offset=-1), np.ones(len(pdb.atom_list) - 1,))
        )
        self.assertTrue(
            np.array_equal(adj.diagonal(offset=2), np.zeros(len(pdb.atom_list) - 2,))
        )
        self.assertTrue(
            np.array_equal(adj.diagonal(offset=-2), np.zeros(len(pdb.atom_list) - 2,))
        )
        atom_features = pdb.get_atom_features()
        # assert only backbone atoms are included in atom_features
        self.assertEqual(len(atom_features), 319 * 3)
        self.assertEqual(atom_features[-1][:3], [-14.909, -4.100, 8.772])
        self.assertEqual(len(atom_features), adj.shape[0])
        self.assertEqual(len(atom_features[0]), 5)

    def test_fragment_labels(self):
        pdb = PDB(self.fpath, sanitize=False)
        atom_features = pdb.get_atom_features(numeric=True, fragment_label=True)
        self.assertEqual(len(atom_features[0]), 624)
        graph = pdb.to_graph(fragment_label=True)
        self.assertEqual(len(graph["atom_features"][0]), 624)
        # assert the order of the atom fragment labels are correct
        atom_features = pdb.get_atom_features(
            numeric=True, fragment_label=True, sort_atoms=True
        )
        mfl = MolFragmentsLabel()
        frag_labels = mfl.create_labels_for(pdb.rdkit_mol, sparse=False)
        for i, atom in enumerate(pdb.sorted_atoms):
            idx = atom.GetIdx()
            self.assertEqual(atom_features[i][6:], tuple(frag_labels[:, idx].tolist()))
        # assert padding still work
        atom_features = pdb.get_atom_features(
            numeric=True, fragment_label=True, padding=2618
        )
        self.assertEqual(len(atom_features), 2618)
        self.assertEqual(atom_features[-1], tuple([24] + [0] * 623))

    def test_networkx_graph(self):
        pdb = PDB(self.fpath, sanitize=False)
        graph = pdb.to_graph(networkx=True)
        self.assertEqual(graph.graph["n_atomtypes"], 25)
        self.assertEqual(graph.nodes[0]["atomtype"], 1)
        self.assertEqual(graph.nodes[0]["formalcharge"], 0)
        self.assertEqual(graph.nodes[0]["degree"], 1)
        self.assertEqual(graph.nodes[0]["hybridization"], 0)
        self.assertEqual(graph.nodes[0]["aromatic"], 0)
        self.assertEqual(graph.nodes[0]["chirality"], 0)
        self.assertEqual(graph.edges[0, 1]["bondtype"], 0)
        self.assertEqual(graph.edges[0, 1]["bonddir"], 0)

    def test_pyg_graph(self):
        try:
            import torch
            import torch_geometric as pyg

            torch_avail = True
        except ImportError:
            torch_avail = False

        if torch_avail:
            pdb = PDB(self.fpath, sanitize=False)
            with self.assertRaises(AssertionError):
                pdb.to_graph(networkx=True, pyg=True)
            graph = pdb.to_graph(include_coordinates=True, pyg=True)
            self.assertIsInstance(graph, pyg.data.Data)
            self.assertTrue(hasattr(graph, "x"))
            self.assertTrue(hasattr(graph, "edge_idx"))
            self.assertTrue(hasattr(graph, "edge_attr"))
            self.assertIsInstance(graph["edge_idx"], torch.Tensor)
            self.assertIsInstance(graph["edge_attr"], torch.Tensor)
            self.assertEqual(graph["edge_idx"].size(1), graph["edge_attr"].size(0))


class TestReadMol(unittest.TestCase):
    def setUp(self):
        self.fpath = os.path.join(
            "chemreader", "tests", "testing_resources", "test_mol_reader.mol"
        )

    def test_block_reading(self):
        reader = MolReader(self.fpath)
        self.assertEqual(reader.n_mols, 3)
        self.assertIsInstance(reader.blocks, list)
        self.assertEqual(len(reader.blocks), 3)

        block = MolBlock(reader.blocks[0])
        self.assertIsInstance(block.rdkit_mol, Chem.rdchem.Mol)
        self.assertEqual(block.rdkit_mol.GetNumAtoms(), 24)
        self.assertEqual(block.rdkit_mol.GetNumBonds(), 26)
        adj = block.get_adjacency_matrix(sparse=False)
        self.assertEqual(adj.shape, (24, 24))
        sparse_adj = block.get_adjacency_matrix(sparse=True)
        self.assertIsInstance(sparse_adj, sp.csr.csr_matrix)
        atom_features = block.get_atom_features(numeric=False)
        self.assertEqual(len(atom_features), 24)
        self.assertEqual(atom_features[21][3], "Cl")
        atom_features = block.get_atom_features(numeric=True)
        self.assertEqual(atom_features[21][3], 7)


class TestAtomOrderConverter(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.mol1 = Chem.MolFromMol2File(
            os.path.join(dir_path, "testing_resources", "mol_origin.mol2"),
            removeHs=False,
        )
        self.mol2 = Chem.MolFromMol2File(
            os.path.join(dir_path, "testing_resources", "mol_diff1.mol2"),
            removeHs=False,
        )
        self.mol3 = Chem.MolFromMol2File(
            os.path.join(dir_path, "testing_resources", "mol_diff2.mol2"),
            removeHs=False,
        )
        self.mol4 = Chem.MolFromMol2File(
            os.path.join(dir_path, "testing_resources", "mol2_origin.mol2"),
            removeHs=False,
        )
        self.mol5 = Chem.MolFromMol2File(
            os.path.join(dir_path, "testing_resources", "mol2_diff1.mol2"),
            removeHs=False,
        )
        self.mol6 = Chem.MolFromMol2File(
            os.path.join(dir_path, "testing_resources", "mol2_diff2.mol2"),
            removeHs=False,
        )

    def assert_mols_equal(self, mol1, mol2):
        conf1 = mol1.GetConformer(0)
        conf2 = mol2.GetConformer(0)
        for idx in range(mol1.GetNumAtoms()):
            a1, a2 = (mol1.GetAtomWithIdx(idx), mol2.GetAtomWithIdx(idx))
            self.assertEqual(a1.GetSymbol(), a2.GetSymbol())
            self.assertEqual(a1.GetIdx(), a2.GetIdx())
            pos1 = conf1.GetAtomPosition(a1.GetIdx())
            pos2 = conf2.GetAtomPosition(a2.GetIdx())
            for coor1, coor2 in zip(pos1, pos2):
                self.assertEqual(coor1, coor2)

    def test_output_atom_order_are_the_same(self):
        conv1 = CanonicalAtomOrderConverter(self.mol1)
        conv2 = CanonicalAtomOrderConverter(self.mol2)
        conv3 = CanonicalAtomOrderConverter(self.mol3)

        new_mol1 = conv1.convert()
        new_mol2 = conv2.convert()
        new_mol3 = conv3.convert()

        self.assert_mols_equal(new_mol1, new_mol2)
        self.assert_mols_equal(new_mol1, new_mol3)
