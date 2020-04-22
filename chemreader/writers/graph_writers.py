import os


class _BaseWriter:

    def __init__(self):
        """
        """


class GraphWriter(_BaseWriter):

    def __init__(self, mols: list, **kwargs):
        super().__init__(**kwargs)
        self.mols = mols

    def write(self, out_path, prefix=None, graph_labels=None):
        """ Write graphs to path.

        Args:
            out_path (str): Path to write out the graphs.
            prefix (str): Optional. Prefix of the output file names. Default is
                None.
        """
        os.makedirs(out_path, exist_ok=True)
        sep = "" if prefix is None else "_"
        a = open(os.path.join(out_path, prefix+sep+"A.txt"), "w")
        idc = open(
            os.path.join(out_path, prefix+sep+"graph_indicator.txt"), "w")
        n_label = open(
            os.path.join(out_path, prefix+sep+"node_labels.txt"), "w")
        edge_attr = open(
            os.path.join(out_path, prefix+sep+"edge_attributes.txt"), "w")
        # initialize variables for graph indicator and nodes indices
        graph_id = 1
        node_starting_index = 0
        for mol in self.mols:
            graph = mol.to_graph(sparse=True)
            adj = graph["adjacency"].tocoo()
            atom_feat = graph["atom_features"]
            bond_feat = graph["bond_features"]
            # write adjacency matrix and update node starting index
            for origin, target in zip(adj.row, adj.col):
                origin = origin + 1 + node_starting_index
                target = target + 1 + node_starting_index
                a.write(str(origin)+","+str(target)+"\n")
            node_starting_index += mol.num_atoms
            # write graph indicator
            idc.write((str(graph_id)+"\n")*mol.num_atoms)
            graph_id += 1
            # write node features
            writable_atom_feat = ""
            for feat in atom_feat:
                writable_atom_feat += ",".join(map(str, feat)) + "\n"
            n_label.write(writable_atom_feat)
            # write bond features
            writable_bond_feat = "\n".join(map(str, bond_feat)) + "\n"
            edge_attr.write(writable_bond_feat)

        # close files
        a.close()
        idc.close()
        n_label.close()
        edge_attr.close()
        # write graph labels
        if graph_labels is not None:
            g_label = open(
                os.path.join(out_path, prefix+sep+"graph_labels.txt"), "w")
            g_label.write("\n".join(map(str, graph_labels)))
            g_label.close()
