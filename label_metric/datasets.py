import os
from typing import Dict, List, Tuple, Any, Iterable, Optional
import logging
import random
from itertools import accumulate

import torch
from torch.utils.data import Dataset
from anytree import Node, find_by_attr, LevelOrderIter, LevelOrderGroupIter
import torchaudio
import lightning as L

from label_metric.utils.tree_utils import tree_to_string, iter_parent_nodes, \
    NodeAffinity, prune_tree, repair_tree
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
        logger: logging.Logger,
        fold_id: int,
        fold_num: int,
        mask_value: int,
        dataset_sr: int,
        dataset_channel_num: int,
        random_seed: Optional[int]
    ) -> None:
        self.dataset_dir = os.path.join(dataset_dir, 'OrchideaSOL2020')
        assert split in ['train', 'valid', 'test', 'predict']
        self.split = split
        # minimum n_samples required for a leaf to be visible
        self.min_num_per_leaf = min_num_per_leaf
        self.duration = duration
        assert train_ratio + valid_ratio <= 1
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.logger = logger
        assert fold_id < fold_num
        self.fold_id = fold_id
        self.fold_num = fold_num
        # target index to ignore
        assert mask_value < 0
        self.mask_value = mask_value
        # default values just for check
        self.dataset_sr = dataset_sr
        self.dataset_channel_num = dataset_channel_num
        # set seed here for consistent split
        if random_seed is not None:
            L.seed_everything(random_seed)
        # prepare data
        self.data, self.tree, self.visible_leaves, self.level_order_visible_nodes, \
            self.level_order_flat_visible_nodes = self.load_data()
        self.node_to_index = self.prepare_node_to_index_mapping()
        self.node_affinity = NodeAffinity(self.tree)
        self.aff_mtx = self.prepare_leaf_affinity()
        # log info
        self.logger.info(f'{self.split} set data loaded\n{self.__str__()}')

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: Any) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return tree_to_string(self.tree)

    def prepare_item(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        audio_path = self.data[idx]['path']
        audio, sr = torchaudio.load(audio_path)
        assert audio.shape[0] == self.dataset_channel_num
        assert sr == self.dataset_sr
        audio = standardize_duration(audio, sr=sr, dur=self.duration)
        audio = audio.squeeze(0)
        label = self.data[idx]['label']
        return audio, label
    
    def load_tree(self) -> Tuple[Node, Dict[Node, str]]:
        # load tree according to file hierarchy
        leaf_node_to_dir = {}
        for root, children, _ in os.walk(self.dataset_dir):
            rel_bases = os.path.relpath(root, self.dataset_dir).split(os.sep)
            parent = rel_bases[-1]
            if parent == '.':
                parent = self.__class__.__name__
                tree = Node(parent)
            for child in children:
                node = Node(child, parent=find_by_attr(tree, parent))
                cur_dir = os.path.join(root, child)
                if os.listdir(cur_dir)[0].endswith('.wav'):
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

    def load_data(self) -> Tuple[List[Dict[str, Any]], Node, List[Node]]:
        dataset = []
        tree, leaf_node_to_dir = self.load_tree()
        # split visible/hidden (i.e. train+valid+test/predict)
        tree_leaves = list(tree.leaves)
        deficient_leaves = self.get_deficient_leaves(leaf_node_to_dir)
        sufficient_leaves = [leaf for leaf in tree_leaves if leaf not in deficient_leaves]
        cv_unseen_leaves = self.get_cv_unseen_leaves(sufficient_leaves)
        hidden_leaves = deficient_leaves + cv_unseen_leaves
        # prune
        pruned_edges = prune_tree(root=tree, leaves_to_prune=hidden_leaves)
        # target spaces
        visible_leaves = [leaf for leaf in tree_leaves if leaf not in hidden_leaves]
        level_order_visible_nodes = list(LevelOrderGroupIter(tree))
        level_order_flat_visible_nodes = list(LevelOrderIter(tree))
        # repair
        repair_tree(pruned_edges)
        # load data per leaf
        leaves = hidden_leaves if self.split == 'predict' else visible_leaves
        for leaf in leaves:
            # prepare labels
            try:
                leaf_label = torch.tensor(visible_leaves.index(leaf))
            except ValueError:
                assert leaf in hidden_leaves
                leaf_label = torch.tensor(self.mask_value)
            paths = leaf.ancestors + (leaf,)
            per_level_labels = []
            for i in range(len(level_order_visible_nodes)):
                if len(level_order_visible_nodes[i]) == 1:
                    continue # skip when only has one node, root level is an example
                try:
                    per_level_labels.append(level_order_visible_nodes[i].index(paths[i]))
                except (ValueError, IndexError):
                    # ValueError: paths[i] not in level_order_visible_nodes[i] (it's unseen)
                    # IndexError: index out of range as paths length shorter than tree height
                    per_level_labels.append(self.mask_value)
            per_level_labels = torch.tensor(per_level_labels)
            binary_labels = torch.tensor([1.0 if node in paths else 0.0 \
                for node in level_order_flat_visible_nodes[1:]])
            aff_idx = torch.tensor(tree.leaves.index(leaf))
            # split train/valid/test, (the shuffle) won't effect predict
            audio_dir = leaf_node_to_dir[leaf]
            audio_files = os.listdir(audio_dir)
            random.shuffle(audio_files)
            train_size = int(self.train_ratio * len(audio_files))
            valid_size = int(self.valid_ratio * len(audio_files))
            if self.split == 'train':
                audio_files = audio_files[:train_size]
            elif self.split == 'valid':
                audio_files = audio_files[train_size:train_size + valid_size]
            elif self.split == 'test':
                audio_files = audio_files[train_size + valid_size:]
            # load files
            for f in audio_files:
                assert f.endswith('.wav'), f'{f}'
                fn_sep = f.split('-')
                data = {
                    'path':                 os.path.join(audio_dir, f),
                    'inst_fam':             paths[1].name,
                    'inst':                 paths[2].name,
                    'mute':                 paths[3].name,
                    'p_tech':               paths[4].name,
                    'pitch':                fn_sep[2],
                    'dynamics':             fn_sep[3],
                    'node':                 leaf,
                    'label': {
                        'leaf':             leaf_label,
                        'per_level':        per_level_labels,
                        'binary':           binary_labels,
                        'aff_idx':          aff_idx
                    }
                }
                dataset.append(data)
        # update leaf names
        self.update_node_name_with_str(visible_leaves, '[visible]')
        self.update_node_name_with_str(deficient_leaves, '[deficient]')
        self.update_node_name_with_str(cv_unseen_leaves, '[cv-unseen]')
        return dataset, tree, visible_leaves, level_order_visible_nodes, \
            level_order_flat_visible_nodes

    def prepare_node_to_index_mapping(self) -> Dict[Node, List[int]]:
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
                node_to_index[ancestor].extend(node_to_index[leaf])
        self.update_node_name_with_num(node_to_index)
        return node_to_index

    def get_deficient_leaves(self, leaf_node_to_dir: Dict[Node, str]) -> List[Node]:
        return [node for node, audio_dir in leaf_node_to_dir.items() \
            if len(os.listdir(audio_dir)) < self.min_num_per_leaf]

    def get_cv_unseen_leaves(self, leaves: List[Node]) -> List[Node]:
        random.shuffle(leaves)
        base_size, reminder = divmod(len(leaves), self.fold_num)
        fold_sizes = [base_size + 1] * reminder + [base_size] * (self.fold_num - reminder)
        indexing = [0] + list(accumulate(fold_sizes))
        unseen_leaves = leaves[indexing[self.fold_id]:indexing[self.fold_id + 1]]
        return unseen_leaves

    def prepare_leaf_affinity(self) -> Dict[str, torch.Tensor]:
        leaf_num = len(self.tree.leaves)
        aff_mtx = {
            'sum': torch.zeros(leaf_num, leaf_num),
            'max': torch.zeros(leaf_num, leaf_num)
        }
        for i in range(leaf_num):
            node_i = self.tree.leaves[i]
            for j in range(leaf_num):
                node_j = self.tree.leaves[j]
                aff_mtx['sum'][i,j] = self.node_affinity(node_i, node_j, 'sum')
                aff_mtx['max'][i,j] = self.node_affinity(node_i, node_j, 'max')
        return aff_mtx

    def update_node_name_with_num(self, node_to_index: Dict[Node, List[int]]) -> None:
        for node in LevelOrderIter(self.tree):
            node.name = f'{node.name} {len(node_to_index[node])}'

    def update_node_name_with_str(self, nodes: Iterable[Node], addstring: str) -> None:
        for node in nodes:
            node.name = f'{node.name} {addstring}'


class BasicOrchideaSOL(OrchideaSOL):
    def __init__(
        self, 
        dataset_dir: str,
        split: str,
        min_num_per_leaf: int,
        duration: float,
        train_ratio: float,
        valid_ratio: float,
        logger: logging.Logger,
        fold_id: int,
        fold_num: int,
        mask_value: int,
        dataset_sr: int,
        dataset_channel_num: int,
        random_seed: Optional[int]
    ) -> None:
        super().__init__(
            dataset_dir,
            split,
            min_num_per_leaf,
            duration,
            train_ratio,
            valid_ratio,
            logger,
            fold_id,
            fold_num,
            mask_value,
            dataset_sr,
            dataset_channel_num,
            random_seed
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
        logger: logging.Logger,
        fold_id: int,
        fold_num: int,
        mask_value: int,
        dataset_sr: int,
        dataset_channel_num: int,
        random_seed: Optional[int]
    ) -> None:
        super().__init__(
            dataset_dir,
            split,
            min_num_per_leaf,
            duration,
            train_ratio,
            valid_ratio,
            logger,
            fold_id,
            fold_num,
            mask_value,
            dataset_sr,
            dataset_channel_num,
            random_seed
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

    from label_metric.paths import DATA_DIR_EECS, DATA_DIR_APOCRITA
    from label_metric.utils.log_utils import setup_logger
    logger = logging.getLogger(__name__)
    setup_logger(logger)

    train_paths, valid_paths, test_paths, predict_paths = [], [], [], []
    
    for i in range(5):

        logger.info(f'datasets for cv fold {i}:')

        train_set = TripletOrchideaSOL(
            dataset_dir = DATA_DIR_EECS,
            split = 'train',
            min_num_per_leaf = 10,
            duration = 1.0,
            train_ratio = 0.8,
            valid_ratio = 0.1,
            logger = logger,
            dataset_sr = 44100,
            dataset_channel_num = 1,
            fold_num = 5,
            fold_id = i,
            mask_value = -1,
            random_seed = 2024
        )

        valid_set = BasicOrchideaSOL(
            dataset_dir = DATA_DIR_EECS,
            split = 'valid',
            min_num_per_leaf = 10,
            duration = 1.0,
            train_ratio = 0.8,
            valid_ratio = 0.1,
            logger = logger,
            dataset_sr = 44100,
            dataset_channel_num = 1,
            fold_num = 5,
            fold_id = i,
            mask_value = -1,
            random_seed = 2024
        )

        test_set = BasicOrchideaSOL(
            dataset_dir = DATA_DIR_EECS,
            split = 'test',
            min_num_per_leaf = 10,
            duration = 1.0,
            train_ratio = 0.8,
            valid_ratio = 0.1,
            logger = logger,
            dataset_sr = 44100,
            dataset_channel_num = 1,
            fold_num = 5,
            fold_id = i,
            mask_value = -1,
            random_seed = 2024
        )

        predict_set = BasicOrchideaSOL(
            dataset_dir = DATA_DIR_EECS,
            split = 'predict',
            min_num_per_leaf = 10,
            duration = 1.0,
            train_ratio = 0.8,
            valid_ratio = 0.1,
            logger = logger,
            dataset_sr = 44100,
            dataset_channel_num = 1,
            fold_num = 5,
            fold_id = i,
            mask_value = -1,
            random_seed = 2024
        )

        train_paths.append(set([data['path'] for data in train_set.data]))
        valid_paths.append(set([data['path'] for data in valid_set.data]))
        test_paths.append(set([data['path'] for data in test_set.data]))
        predict_paths.append(set([data['path'] for data in predict_set.data]))

        assert not bool(train_paths[i] & valid_paths[i])
        assert not bool(train_paths[i] & test_paths[i])
        assert not bool(train_paths[i] & predict_paths[i])

    assert train_paths[0] | valid_paths[0] | test_paths[0] | predict_paths[0] == \
        predict_paths[0] | predict_paths[1] | predict_paths[2] | predict_paths[3] | predict_paths[4]

    from torch.utils.data import DataLoader

    valid_loader = DataLoader(
        valid_set,
        batch_size = 32,
        shuffle = True,
        drop_last = False
    )

    batch = next(iter(valid_loader))
    print(f'audio shape: {batch[0].shape}\nlabel shape:')
    for k,v in batch[1].items():
        print(k, v.shape)
