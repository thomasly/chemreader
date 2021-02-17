import os


class _BaseWriter:
    def __init__(self):
        """
        """


class GraphWriter(_BaseWriter):
    def __init__(self, mols: list, **kwargs):
        super().__init__(**kwargs)
        self.mols = mols

    def write(self, out_path, prefix=None, edge_features=True, graph_labels=None):
        """ Write graphs to path.

        Args:
            out_path (str): Path to write out the graphs.
            prefix (str): Optional. Prefix of the output file names. Default is
                None.

        Returns:
            prefix_A.txt: adjacency matrices
            prefix_graph_indicator.txt: graph indicator
            prefix_node_label.txt: node features
            prefix_edge_attributes.txt: edge features
            prefix_graph_labels.txt: graph labels if graph_labels is not None
        """
        os.makedirs(out_path, exist_ok=True)
        prefix = "" if prefix is None else prefix + "_"
        a = open(os.path.join(out_path, prefix + "A.txt"), "w")
        idc = open(os.path.join(out_path, prefix + "graph_indicator.txt"), "w")
        n_label = open(os.path.join(out_path, prefix + "node_attributes.txt"), "w")
        if edge_features:
            edge_attr = open(
                os.path.join(out_path, prefix + "edge_attributes.txt"), "w"
            )
        # initialize variables for graph indicator and nodes indices
        graph_id = 1
        node_starting_index = 0
        for mol in self.mols:
            graph = mol.to_graph(sparse=True)
            adj = graph["adjacency"].tocoo()
            atom_feat = graph["atom_features"]
            if edge_features:
                bond_feat = graph["bond_features"]
            # write adjacency matrix and update node starting index
            for origin, target in zip(adj.row, adj.col):
                if edge_features:
                    # write bond features
                    bond = str(origin) + "-" + str(target)
                    writable_bond_feat = str(bond_feat[bond])[1:-1] + "\n"
                    edge_attr.write(writable_bond_feat)
                origin = origin + 1 + node_starting_index
                target = target + 1 + node_starting_index
                a.write(str(origin) + "," + str(target) + "\n")
            node_starting_index += mol.num_atoms
            # write graph indicator
            idc.write((str(graph_id) + "\n") * mol.num_atoms)
            graph_id += 1
            # write node features
            writable_atom_feat = ""
            for feat in atom_feat:
                writable_atom_feat += ",".join(map(str, feat)) + "\n"
            n_label.write(writable_atom_feat)

        # close files
        a.close()
        idc.close()
        n_label.close()
        if edge_features:
            edge_attr.close()
        # write graph labels
        if graph_labels is not None:
            g_label = open(os.path.join(out_path, prefix + "graph_labels.txt"), "w")
            g_label.write("\n".join(map(str, graph_labels)))
            g_label.write("\n")
            g_label.close()
