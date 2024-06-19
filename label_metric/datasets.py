import os
from typing import Dict, List, Tuple, Any
import logging
import random

import torch
from torch.utils.data import Dataset
from anytree import Node, find_by_attr, LevelOrderIter
import torchaudio

from label_metric.utils.tree_utils import tree_to_string, iter_parent_nodes
from label_metric.utils.audio_utils import standardize_duration

class OrchideaSOL(Dataset):
    
    def __init__(
        self, 
        dataset_dir: str,
        split: str,
        min_num_per_leaf: int,
        duration: float,
        train_ratio: float,
        valid_ratio: float,
        logger: logging.Logger
    ) -> None:
        
        self.dataset_dir = os.path.join(dataset_dir, 'OrchideaSOL2020')
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.min_num_per_leaf = min_num_per_leaf
        self.duration = duration
        assert train_ratio + valid_ratio <= 1
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.logger = logger
        
        self.data, self.tree = self.load_data()
        self.node_to_index = self.prepare_node_to_index_mapping()
        self.update_node_name_with_num()

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: Any) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return tree_to_string(self.tree)

    def prepare_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.data[idx]['path']
        label = self.data[idx]['label']
        audio, sr = torchaudio.load(audio_path)
        assert audio.shape[0] == 1 # one channel
        assert sr == 44100
        audio = standardize_duration(audio, sr=sr, dur=self.duration)
        audio = torch.squeeze(audio, 0)
        return audio, torch.tensor(label)
    
    def load_tree(self) -> Tuple[Node, Dict[Node, str]]:

        leaf_node_to_dir = {}
        for root, children, _ in os.walk(self.dataset_dir):
            rel_bases = os.path.relpath(root, self.dataset_dir).split(os.sep)
            parent = rel_bases[-1]
            if parent == '.':
                parent = self.__class__.__name__
                tree = Node(parent)
            for child in children:
                cur_dir = os.path.join(root, child)
                if not os.listdir(cur_dir)[0].endswith('.wav'): # if not leaf
                    node = Node(child, parent=find_by_attr(tree, parent))
                elif len(os.listdir(cur_dir)) > self.min_num_per_leaf: # if leaf and has enough data
                    node = Node(child, parent=find_by_attr(tree, parent))
                    leaf_node_to_dir[node] = cur_dir
        
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
        
        return tree, leaf_node_to_dir

    def load_data(self) -> Tuple[List[Dict[str, Any]], Node]:

        dataset = []
        tree, leaf_node_to_dir = self.load_tree()

        for leaf in tree.leaves: # a leaf is a class
            audio_dir = leaf_node_to_dir[leaf]
            audio_files = os.listdir(audio_dir)
            
            # shuffle and split
            random.shuffle(audio_files)
            train_size = int(self.train_ratio * len(audio_files))
            valid_size = int(self.valid_ratio * len(audio_files))
            if self.split == 'train':
                audio_files = audio_files[:train_size]
            elif self.split == 'valid':
                audio_files = audio_files[train_size:train_size + valid_size]
            elif self.split == 'test':
                audio_files = audio_files[train_size + valid_size:]
            
            for f in audio_files:
                assert f.endswith('.wav')
                fn_sep = f.split('-')
                data = {
                    'path':         os.path.join(audio_dir, f),
                    'inst_fam':     leaf.parent.parent.parent.name,
                    'inst':         leaf.parent.parent.name,
                    'mute':         leaf.parent.name,
                    'p_tech':       leaf.name,
                    'pitch':        fn_sep[2],
                    'dynamics':     fn_sep[3],
                    'node':         leaf,
                    'label':        tree.leaves.index(leaf),
                }
                dataset.append(data)

        return dataset, tree

    def prepare_node_to_index_mapping(self) -> Dict[Node, List]:
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
        
    def update_node_name_with_num(self) -> None:
        for node in LevelOrderIter(self.tree):
            node.name = f'{node.name} {len(self.node_to_index[node])}'
        self.logger.info(f'Load {self.split} set data\n{self.__str__()}')


class BasicOrchideaSOL(OrchideaSOL):

    def __init__(
        self, 
        dataset_dir: str,
        split: str,
        min_num_per_leaf: int,
        duration: float,
        train_ratio: float,
        valid_ratio: float,
        logger: logging.Logger
    ) -> None:

        super().__init__(
            dataset_dir,
            split,
            min_num_per_leaf,
            duration,
            train_ratio,
            valid_ratio,
            logger
        )
    
    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.prepare_item(idx)


class TripletOrchideaSOL(OrchideaSOL):

    def __init__(
        self, 
        dataset_dir: str,
        split: str,
        min_num_per_leaf: int,
        duration: float,
        train_ratio: float,
        valid_ratio: float,
        logger: logging.Logger
    ) -> None:

        super().__init__(
            dataset_dir,
            split,
            min_num_per_leaf,
            duration,
            train_ratio,
            valid_ratio,
            logger
        )
    
    def __getitem__(
        self, 
        idxs: Tuple[int, int, int]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:

        a_idx, p_idx, n_idx = idxs
        return {
            'anc': self.prepare_item(a_idx),
            'pos': self.prepare_item(p_idx),
            'neg': self.prepare_item(n_idx),
        }


if __name__ == '__main__':

    # example code

    import lightning as L
    L.seed_everything(2024)
    from label_metric.utils.log_utils import setup_logger
    logger = logging.getLogger(__name__)
    setup_logger(logger)

    train_set = TripletOrchideaSOL(
        dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
        split = 'train',
        min_num_per_leaf = 10,
        duration = 1.0,
        train_ratio = 0.8,
        valid_ratio = 0.1,
        logger = logger
    )

    valid_set = BasicOrchideaSOL(
        dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
        split = 'valid',
        min_num_per_leaf = 10,
        duration = 1.0,
        train_ratio = 0.8,
        valid_ratio = 0.1,
        logger = logger        
    )

    test_set = BasicOrchideaSOL(
        dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
        split = 'test',
        min_num_per_leaf = 10,
        duration = 1.0,
        train_ratio = 0.8,
        valid_ratio = 0.1,
        logger = logger        
    )
