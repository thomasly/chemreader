import os
import unittest

import numpy as np
from scipy import sparse as sp

from ..readers import Mol2, Mol2Block
from ..readers import Smiles
from ..readers import PDB, PartialPDB, PDBBB


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
        self.assertEqual(bond["type"], "am")


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
        self.assertEqual(can_smiles[0], r"C[NH+](C)CCNC(=O)C1:N:O:N:C:1N")
        self.assertEqual(len(iso_smiles), self.mol.n_mols)
        self.assertEqual(iso_smiles[62], r"CN1:C:N:C:C:1[C@H]([NH3+])C1(O)CNC1")
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
        self.assertEqual(Mol2Block.atom_to_num("Any"), 22)
        self.assertEqual(Mol2Block.atom_to_num("@#$%"), 22)
        self.assertEqual(Mol2Block.bond_to_num("1"), 0)
        self.assertEqual(Mol2Block.bond_to_num("nc"), 6)
        self.assertEqual(Mol2Block.bond_to_num("@#$%"), 6)

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
        self.assertTrue(isinstance(bond_features[0]["1-2"], str))
        numeric_features = self.mol.get_bond_features(numeric=True)
        self.assertEqual(len(numeric_features), self.mol.n_mols)
        self.assertEqual(len(numeric_features[0]), self.block.num_bonds * 2)
        self.assertTrue(isinstance(numeric_features[0]["1-2"], int))

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
        self.assertTrue(isinstance(graphs[0]["atom_features"][0][0], float))
        self.assertTrue(isinstance(graphs[0]["atom_features"][0][-1], int))
        self.assertEqual(len(graphs[0]["bond_features"]), self.block.num_bonds * 2)
        self.assertTrue(isinstance(graphs[0]["bond_features"]["1-2"], int))
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
        # Aspirin
        self.smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
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
        self.assertEqual(feats[0], ("C", 1, 0, 4, 0, 0))
        self.assertEqual(feats[4], ("C", 3, 0, 3, 1, 0))
        feats = self.sm.get_atom_features(numeric=True)
        self.assertEqual(len(feats), 13)
        self.assertEqual(feats[0], (0, 1, 0, 4, 0, 0))
        feats = self.sm.get_atom_features(numeric=True, padding=20)
        self.assertEqual(len(feats), 20)
        self.assertEqual(feats[0], (0, 1, 0, 4, 0, 0))
        self.assertEqual(feats[-1], (22, 0, 0, 0, 0, 0))
        with self.assertRaises(ValueError):
            self.sm.get_atom_features(padding=12)

    def test_bond_features(self):
        feats = self.sm.get_bond_features()
        self.assertEqual(len(feats), 26)
        self.assertEqual(feats["0-1"], "1")
        feats = self.sm.get_bond_features(numeric=True)
        self.assertEqual(len(feats), 26)
        self.assertEqual(feats["0-1"], 0)
        # feats = self.sm.get_bond_features(numeric=True, padding=15)
        # self.assertEqual(len(feats), 15)
        # self.assertEqual(feats[0], 0)
        # self.assertEqual(feats[-1], 6)
        # with self.assertRaises(ValueError):
        #     self.sm.get_bond_features(padding=12)

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
        other = Smiles("C1ccccC1")
        self.assertFalse(self.sm.similar_to(other))
        self.assertTrue(self.sm.similar_to(self.sm))
        fp = Smiles("C1").fingerprint
        self.assertIsNone(fp)


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

    def test_coordinates(self):
        pdb = PDB(self.fpath, sanitize=False)
        counter = 0
        for atom in pdb.rdkit_mol.GetAtoms():
            print(atom.GetSymbol(), end=" ")
            counter += 1
        print()
        print(counter)
        atoms = pdb.get_atom_coordinates()
        self.assertIsInstance(atoms, list)
        self.assertEqual(len(atoms), pdb.num_atoms)

    def test_backbone_pdb(self):
        pdb = PDBBB(self.fpath, sanitize=False)
        adj = pdb.get_adjacency_matrix()
        self.assertTrue(np.array_equal(adj.diagonal(), np.ones(pdb.num_atoms,)))
        self.assertTrue(
            np.array_equal(adj.diagonal(offset=1), np.ones(pdb.num_atoms - 1,))
        )
        self.assertTrue(
            np.array_equal(adj.diagonal(offset=-1), np.ones(pdb.num_atoms - 1,))
        )
        self.assertTrue(
            np.array_equal(adj.diagonal(offset=2), np.zeros(pdb.num_atoms - 2,))
        )
        self.assertTrue(
            np.array_equal(adj.diagonal(offset=-2), np.zeros(pdb.num_atoms - 2,))
        )
        # assert only backbone atoms are included in atom_features
        self.assertEqual(len(pdb.get_atom_features()), 319 * 3)
        self.assertEqual(pdb.get_atom_features()[-1][:3], [-14.909, -4.100, 8.772])
