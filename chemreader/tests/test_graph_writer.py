import os
import shutil
from unittest import TestCase

from ..writers import GraphWriter
from ..readers import Mol2
from ..readers import Smiles
from ..readers import PDB, PartialPDB


class TestGraphWriting(TestCase):
    def setUp(self):
        self.outpath = os.path.join("tmp", "test_graph_writer")

    def tearDown(self):
        # remove created files
        shutil.rmtree(self.outpath)
        # pass

    def test_generating_graphs_from_smiles(self):
        smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Asprin
            "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@@H]1[C@H]([C@H]"
            "([C@](O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)"
            "OC4=CC=CC=C4",  # Remdesivir
        ]
        mols = [Smiles(s) for s in smiles]
        labels = [1, 0]

        writer = GraphWriter(mols)
        writer.write(self.outpath, prefix="test", graph_labels=labels)
        self.assertTrue(os.path.isfile(os.path.join(self.outpath, "test_A.txt")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.outpath, "test_graph_indicator.txt"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.outpath, "test_node_attributes.txt"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.outpath, "test_edge_attributes.txt"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.outpath, "test_graph_labels.txt"))
        )
        # assert adjacency matrices are correct
        with open(os.path.join(self.outpath, "test_A.txt"), "r") as f:
            lines = f.readlines()
        line1 = lines[0].split(",")
        adj1 = mols[0].get_adjacency_matrix(sparse=True).tocoo()
        self.assertEqual(int(line1[0]), adj1.row[0] + 1)
        self.assertEqual(int(line1[1]), adj1.col[0] + 1)
        n_bonds = mols[0].num_bonds
        n_atoms = mols[0].num_atoms
        line2 = lines[n_bonds * 2].split(",")  # Bonds are symmetric, so *2
        adj2 = mols[1].get_adjacency_matrix(sparse=True).tocoo()
        self.assertEqual(int(line2[0]), adj2.row[0] + n_atoms + 1)
        self.assertEqual(int(line2[1]), adj2.col[0] + n_atoms + 1)
        # assert graph indicator is correct
        with open(os.path.join(self.outpath, "test_graph_indicator.txt"), "r") as f:
            lines = f.readlines()
            self.assertEqual(lines.count("1\n"), mols[0].num_atoms)
            self.assertEqual(lines.count("2\n"), mols[1].num_atoms)
        # assert node labels are correct
        with open(os.path.join(self.outpath, "test_node_attributes.txt"), "r") as f:
            lines = f.readlines()
        true_feats = list()
        for mol in mols:
            true_feats += mol.get_atom_features(numeric=True)
        for line, tf in zip(lines, true_feats):
            for f1, f2 in zip(line.split(","), tf):
                self.assertEqual(float(f1), float(f2))
        # assert edge attributes are correct
        with open(os.path.join(self.outpath, "test_edge_attributes.txt"), "r") as f:
            attrs = f.readlines()
        with open(os.path.join(self.outpath, "test_A.txt"), "r") as f:
            adj = f.readlines()
        true_attrs = list()
        num_edges = list()
        num_nodes = list()
        for mol in mols:
            bond_featurs = mol.get_bond_features(numeric=True)
            true_attrs.append(bond_featurs)
            num_edges.append(len(bond_featurs))
            num_nodes.append(mol.num_atoms)
        t_nodes = 0
        t_edges = 0
        for tattr, ne, nn in zip(true_attrs, num_edges, num_nodes):
            counter = 0
            while counter < ne:
                at, ad = attrs[t_edges + counter], adj[t_edges + counter]
                at = [int(a) for a in at.split(",")]
                key = "-".join(map(lambda x: str(int(x) - 1 - t_nodes), ad.split(",")))
                ea = tattr[key]
                self.assertEqual(ea, at)
                counter += 1
            t_nodes += nn
            t_edges += ne
        # assert graph labels are correct
        with open(os.path.join(self.outpath, "test_graph_labels.txt"), "r") as f:
            lines = f.readlines()
        for l1, l2 in zip(labels, lines):
            self.assertEqual(int(l1), int(l2))

    def test_generating_graphs_from_mol2blocks(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_gzip = os.path.join(dir_path, "testing_resources", "AAAARO.xaa.mol2.gz")
        mol = Mol2(test_gzip)
        mols = mol.mol2_blocks
        writer = GraphWriter(mols)
        writer.write(self.outpath, prefix="test", graph_labels=None)
        self.assertTrue(os.path.isfile(os.path.join(self.outpath, "test_A.txt")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.outpath, "test_graph_indicator.txt"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.outpath, "test_node_attributes.txt"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.outpath, "test_edge_attributes.txt"))
        )
        # assert adjacency matrices are correct
        with open(os.path.join(self.outpath, "test_A.txt"), "r") as f:
            lines = f.readlines()
        line1 = lines[0].split(",")
        adj1 = mols[0].get_adjacency_matrix(sparse=True).tocoo()
        self.assertEqual(int(line1[0]), adj1.row[0] + 1)
        self.assertEqual(int(line1[1]), adj1.col[0] + 1)
        n_bonds = mols[0].num_bonds
        n_atoms = mols[0].num_atoms
        line2 = lines[n_bonds * 2].split(",")  # Bonds are symmetric, so *2
        adj2 = mols[1].get_adjacency_matrix(sparse=True).tocoo()
        self.assertEqual(int(line2[0]), adj2.row[0] + n_atoms + 1)
        self.assertEqual(int(line2[1]), adj2.col[0] + n_atoms + 1)
        # assert graph indicator is correct
        with open(os.path.join(self.outpath, "test_graph_indicator.txt"), "r") as f:
            lines = f.readlines()
            self.assertEqual(lines.count("1\n"), mols[0].num_atoms)
            self.assertEqual(lines.count("2\n"), mols[1].num_atoms)
        # assert node labels are correct
        with open(os.path.join(self.outpath, "test_node_attributes.txt"), "r") as f:
            lines = f.readlines()
        true_feats = list()
        for mol in mols:
            true_feats += mol.get_atom_features(numeric=True)
        for line, tf in zip(lines, true_feats):
            for f1, f2 in zip(line.split(","), tf):
                self.assertEqual(float(f1), float(f2))
        # assert edge attributes are correct
        with open(os.path.join(self.outpath, "test_edge_attributes.txt"), "r") as f:
            attrs = f.readlines()
        with open(os.path.join(self.outpath, "test_A.txt"), "r") as f:
            adj = f.readlines()
        true_attrs = list()
        num_edges = list()
        num_nodes = list()
        for mol in mols:
            bond_featurs = mol.get_bond_features(numeric=True)
            true_attrs.append(bond_featurs)
            num_edges.append(len(bond_featurs))
            num_nodes.append(mol.num_atoms)
        t_nodes = 0
        t_edges = 0
        for tattr, ne, nn in zip(true_attrs, num_edges, num_nodes):
            counter = 0
            while counter < ne:
                at, ad = attrs[t_edges + counter], adj[t_edges + counter]
                at = [int(a) for a in at.split(",")]
                key = "-".join(map(lambda x: str(int(x) - 1 - t_nodes), ad.split(",")))
                ea = tattr[key]
                self.assertEqual(ea, at)
                counter += 1
            t_nodes += nn
            t_edges += ne

    def test_prefix_bug_fix(self):
        smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Asprin
            "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@@H]1[C@H]([C@H]"
            "([C@](O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)"
            "OC4=CC=CC=C4",  # Remdesivir
        ]
        mols = [Smiles(s) for s in smiles]
        labels = [1, 0]

        writer = GraphWriter(mols)
        writer.write(self.outpath, prefix=None, graph_labels=labels)
        self.assertTrue(os.path.isfile(os.path.join(self.outpath, "A.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.outpath, "graph_labels.txt")))

    def test_generating_graphs_from_pdb(self):
        fpath = os.path.join("chemreader", "tests", "testing_resources", "3CQW.pdb")
        pdb = PDB(fpath)
        part_pdb = PartialPDB(fpath, atom_list=list(range(10)), cutoff=1.5)
        writer = GraphWriter([pdb])
        writer.write(self.outpath, prefix="pdb", edge_features=False)
        writer = GraphWriter([part_pdb])
        writer.write(self.outpath, prefix="part_pdb", edge_features=False)
