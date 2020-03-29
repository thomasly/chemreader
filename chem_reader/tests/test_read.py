import os
import unittest

from ..readers import Mol2, Mol2Block


class TestReadingMol2File(unittest.TestCase):

    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_gzip = os.path.join(
            dir_path, "testing_resources", "AAAARO.xaa.mol2.gz")
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

    def test_coordinates(self):
        self.assertEqual(len(self.block.coordinates), self.block.num_atoms)
        self.assertEqual(
            self.block.coordinates[0][0], "C1")
        self.assertEqual(
            self.block.coordinates[0][1], (-0.0178, 1.4648, 0.0101))
        self.assertEqual(
            self.block.coordinates[-1][0], "H28")
        self.assertEqual(
            self.block.coordinates[-1][1], (-1.3009, 0.3246, 7.4554))
        self.assertTrue(hasattr(self.block, "_coordinates"))

    def test_atom_types(self):
        self.assertEqual(len(self.block.atom_types), self.block.num_atoms)
        self.assertEqual(self.block.atom_types[0], "C.3")
        self.assertEqual(self.block.atom_types[-1], "H")
        self.assertEqual(self.block.atom_types[14], "N.pl3")
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
        self.assertEqual(bond["connect"][0], 1)
        self.assertEqual(bond["connect"][1], 2)
        self.assertEqual(bond["type"], "1")
        bond = self.block.bonds[16]
        self.assertEqual(bond["connect"][0], 7)
        self.assertEqual(bond["connect"][1], 8)
        self.assertEqual(bond["type"], "am")


class TestBlockMissingInformation(unittest.TestCase):

    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_f = os.path.join(dir_path,
                              "testing_resources",
                              "test_mol2_block_missing_info")
        with open(test_f, "r") as f:
            self.block = Mol2Block(f.read())

    def test_num_atoms(self):
        self.assertEqual(self.block.num_atoms, 28)
        self.assertEqual(self.block._num_atoms, 28)

    # number of bonds is got from BOND record instead of MOLECULE record
    def test_num_bonds(self):
        self.assertEqual(self.block.num_bonds, 28)
        self.assertEqual(self.block._num_bonds, 28)

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
        self.assertEqual(iso_smiles[1],
                         r"C[NH2+]C[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)CO")
        self.assertNotEqual(can_smiles[1], iso_smiles[1])


if __name__ == "__main__":
    unittest.main()
