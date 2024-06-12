import os
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from anytree import Node, find_by_attr, LevelOrderIter
from anytree.importer import DictImporter
from anytree.exporter import DictExporter

from label_metric.paths import OrchideaSOL_DIR
from label_metric.utils.tree_utils import print_tree


class OrchideaSOL(Dataset):
    
    def __init__(self, dataset_dir: str = OrchideaSOL_DIR):
        self.dataset_dir = os.path.join(dataset_dir, 'OrchideaSOL2020')
        self.data, self.tree = self.load_data()
        self.node_to_index = self.prepare_node_to_index_mapping()

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        return self.data[idx]

    def load_data(self) -> Tuple[List[Dict], Node]:
        dataset = []
        for root, children, files in os.walk(self.dataset_dir):
            rel_dirs = os.path.relpath(root, self.dataset_dir).split(os.sep)
            parent = rel_dirs[-1]
            if parent == '.':
                parent = 'OrchideaSOL'
                tree = Node(parent)
            for child in children:
                node = Node(child, parent=find_by_attr(tree, parent))
            for file in files:
                if file.endswith('.wav'):
                    fn_sep = file.split('-')
                    # leaf name is not unique, search from parent node
                    parent_node = find_by_attr(tree, rel_dirs[1])
                    node = find_by_attr(parent_node, rel_dirs[2])
                    data = {
                        'path':         os.path.join(root, file),
                        'inst_fam':     rel_dirs[0],
                        'inst+mute':    rel_dirs[1],
                        'tech':         rel_dirs[2],
                        'pitch':        fn_sep[2],
                        'dynamics':     fn_sep[3],
                        'node':         node
                    }
                    dataset.append(data)
        return dataset, tree

    def prepare_node_to_index_mapping(self) -> Dict:
        node_to_index = {}
        for leaf in self.tree.leaves:
            node_to_index[leaf] = []
        for idx, data in enumerate(self.data):
            leaf = data['node']
            node_to_index[leaf].append(idx)
        for leaf in self.tree.leaves:
            for ancestor in leaf.ancestors:
                if ancestor not in node_to_index:
                    node_to_index[ancestor] = []
                node_to_index[ancestor] += node_to_index[leaf]
        return node_to_index
        
    def print_num_per_node(self):
        # copy the original tree, add data num to node name, and print
        importer = DictImporter()
        exporter = DictExporter()
        tree_dict = exporter.export(self.tree)
        ctree = importer.import_(tree_dict)
        nodes = list(LevelOrderIter(self.tree))
        cnodes = list(LevelOrderIter(ctree))
        for cnode in cnodes:
            node = nodes[cnodes.index(cnode)]
            cnode.name = f'{cnode.name} ({len(self.node_to_index[node])})'
        print_tree(ctree)
        

if __name__ == '__main__':
    dataset = OrchideaSOL()
    print_tree(dataset.tree)
    dataset.print_num_per_node()
