import os
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from anytree import Node, find_by_attr, LevelOrderIter

from label_metric.paths import OrchideaSOL_DIR
from label_metric.utils.tree_utils import tree_to_string, iter_parent_nodes


class OrchideaSOL(Dataset):
    
    def __init__(self, dataset_dir: str = OrchideaSOL_DIR):
        self.dataset_dir = os.path.join(dataset_dir, 'OrchideaSOL2020')
        self.data, self.tree = self.load_data()
        self.node_to_index = self.prepare_node_to_index_mapping()
        self.add_num_per_node()

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        return self.data[idx]

    def __str__(self) -> str:
        return tree_to_string(self.tree)

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
                    # leaf name is not unique so search by parent node
                    parent_node = find_by_attr(tree, rel_dirs[1])
                    node = find_by_attr(parent_node, rel_dirs[2])
                    try:
                        inst, mute = rel_dirs[1].split('+')
                    except ValueError:
                        inst, mute = rel_dirs[1], 'open'
                    data = {
                        'path':         os.path.join(root, file),
                        'inst_fam':     rel_dirs[0],
                        'inst':         inst,
                        'mute':         mute,
                        'p_tech':       rel_dirs[2],
                        'pitch':        fn_sep[2],
                        'dynamics':     fn_sep[3],
                        'node':         node
                    }
                    dataset.append(data)
        # unfold 'inst+mute' to 'mute under inst'
        inst_fam_nodes = iter_parent_nodes(tree)[1]
        for inst_fam_node in inst_fam_nodes:
            inst_name_to_inst_node = {}
            inst_name_to_mute_node = {}
            for inst_fam_child in inst_fam_node.children:
                try:
                    inst, mute = inst_fam_child.name.split('+')
                except ValueError:
                    inst, mute = inst_fam_child.name, 'open'
                inst_fam_child.name = mute
                if inst not in inst_name_to_inst_node:
                    inst_name_to_inst_node[inst] = Node(inst)
                if inst not in inst_name_to_mute_node:
                    inst_name_to_mute_node[inst] = []
                inst_name_to_mute_node[inst].append(inst_fam_child)
            inst_fam_node.children = list(inst_name_to_inst_node.values())
            for inst, mute_nodes in inst_name_to_mute_node.items():
                inst_node = inst_name_to_inst_node[inst]
                inst_node.children = mute_nodes
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
        
    def add_num_per_node(self) -> None:
        for node in LevelOrderIter(self.tree):
            node.name = f'{node.name} {len(self.node_to_index[node])}'


if __name__ == '__main__':
    dataset = OrchideaSOL()
    print(dataset)
