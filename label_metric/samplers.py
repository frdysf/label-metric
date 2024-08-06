from itertools import combinations, permutations
import random
from typing import List, Tuple, Iterator, Dict, Union
import logging
import textwrap

import torch
from anytree import Node, LevelOrderGroupIter, LevelOrderIter
from torch.utils.data import Dataset, Sampler

from label_metric.utils.tree_utils import (
    iter_parent_nodes, node_distance, tree_to_string)


class WeightManager():

    def __init__(self, logger: logging.Logger, active: bool) -> None:
        self.logger = logger
        self.active = active

    def update_weight(self, counter: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> None:
        # w = 1 / leaf_counts
        # self.weight = w / w.sum()
        # total_cnt = torch.ones_like(all_nodes_except_root_counts) * leaf_counts.sum()
        # self.pos_weight = (total_cnt - all_nodes_except_root_counts) / all_nodes_except_root_counts
        self.weight = {}
        w = 1 / counter['leaf']
        self.weight['leaf'] = w / w.sum()
        total_cnt = torch.ones_like(counter['binary']) * counter['leaf'].sum()
        self.weight['binary'] = (total_cnt - counter['binary']) / counter['binary']
        self.weight['per_level'] = []
        for cnt in counter['per_level']:
            w = 1 / cnt
            self.weight['per_level'].append(w / w.sum())

    def get_weight(self) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        assert hasattr(self, 'weight'), 'update weight first'
        if not self.active:
            return {'leaf': None, 'binary': None, 'per_level': [None] * len(self.weight['per_level'])}
        return self.weight


class SampleTripletsFromTree(Sampler):

    def __init__(
        self, 
        dataset: Dataset, 
        more_level: int,
        logger: logging.Logger,
        weight_manager: WeightManager
    ) -> None:

        self.dataset = dataset
        self.more_level = more_level
        self.logger = logger
        self.weight_manager = weight_manager
        self.triplets = self.sample()
        self.logger.info(
            f'Each training epoch has {len(self.triplets)} triplets'
        )

    def __len__(self) -> int:
        return len(self.triplets)

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        self.triplets = self.sample()
        random.shuffle(self.triplets)
        return iter(self.triplets)

    def sample(self) -> List[Tuple[int, int, int]]:

        self.counter = {}
        leaf_num = self.dataset.get_leaf_num()
        self.counter['leaf'] = torch.zeros(leaf_num)
        binary_num = self.dataset.get_node_num() - 1
        self.counter['binary'] = torch.zeros(binary_num)
        self.counter['per_level'] = []
        level_sizes = self.dataset.get_level_sizes()
        for size in level_sizes:
            self.counter['per_level'].append(torch.zeros(size))

        triplets = []
        for nodes in iter_parent_nodes(self.dataset.tree):
            for parent_node in nodes:
                self.logger.debug(f'visiting {parent_node}')
                self.logger.debug(tree_to_string(parent_node))
                triplets += self.sample_subtree(parent_node)
        
        self.weight_manager.update_weight(self.counter)
        
        return triplets

    def sample_subtree(self, root: Node) -> List[Tuple[int, int, int]]:
        """
        sample  pos and neg from root.children
                x_a and x_p from pos
                x_n from neg
        """
        triplets = []
        if len(root.children) == 1:
            return triplets
        more_level = min(root.height - 1, self.more_level) # under pos
        for pos, neg in permutations(root.children, 2):
            self.logger.debug(
                f"pos: {pos.name.split(' ')[0]}, "
                f"neg: {neg.name.split(' ')[0]}"
            )
            shallow_leaf_nodes = [node for node in pos.leaves \
                if node_distance(pos, node) < more_level]
            more_level_nodes = list(list(LevelOrderGroupIter(pos))[more_level])
            nodes = shallow_leaf_nodes + more_level_nodes
            self.logger.debug(f"under pos: {[node.name.split(' ')[0] for node in nodes]}")
            if len(nodes) == 1: # nodes = [pos]
                idx_a, idx_p = random.sample(self.dataset.node_to_index[nodes[0]], 2)
                idx_n = random.choice(self.dataset.node_to_index[neg])
                triplets.append((idx_a, idx_p, idx_n))
                self._count(idx_a)
                self._count(idx_p)
                self._count(idx_n)
            else:
                for node_a, node_p in combinations(nodes, 2):
                    self.logger.debug(
                        f"x_a: from {node_a.name.split(' ')[0]}, "
                        f"x_p: from {node_p.name.split(' ')[0]}"
                    )
                    idx_a = random.choice(self.dataset.node_to_index[node_a])
                    idx_p = random.choice(self.dataset.node_to_index[node_p])
                    idx_n = random.choice(self.dataset.node_to_index[neg])
                    triplets.append((idx_a, idx_p, idx_n))
                    self._count(idx_a)
                    self._count(idx_p)
                    self._count(idx_n)
        return triplets

    def _count(self, idx) -> None:
        self.counter['leaf'][self.dataset.data[idx]['label']['leaf']] += 1
        for level, cnt in enumerate(self.counter['per_level']):
            cnt[self.dataset.data[idx]['label']['per_level'][level]] += 1
        self.counter['binary'] += self.dataset.data[idx]['label']['binary']


if __name__ == '__main__':

    # example code
    
    import lightning as L
    from torch.utils.data import DataLoader

    from label_metric.datasets import TripletOrchideaSOL, BasicOrchideaSOL
    from label_metric.utils.log_utils import setup_logger

    logger = logging.getLogger(__name__)
    setup_logger(logger)

    L.seed_everything(2024)

    train_set = TripletOrchideaSOL(
        dataset_dir = '/data/scratch/acw751/_OrchideaSOL2020_release',
        split = 'train',
        min_num_per_leaf = 10,
        duration = 1.0,
        train_ratio = 0.8,
        valid_ratio = 0.1,
        logger = logger,
        dataset_sr = 44100,
        dataset_channel_num = 1
    )

    weight_manager = WeightManager(logger, active = True)

    sampler = SampleTripletsFromTree(
        dataset = train_set, 
        more_level = 1,
        logger = logger,
        weight_manager = weight_manager
    )

    train_loader = DataLoader(
        train_set,
        batch_size = 32,
        sampler = sampler,
        num_workers = 0,
        drop_last = True
    )
    
    batch = next(iter(train_loader))

    print("\nweight of the first epoch:")
    weight = weight_manager.get_weight()
    print(f"leaf weight shape: {weight['leaf'].shape}")
    print(f"binary weight shape: {weight['binary'].shape}")
    print(f"per_level weight shape: {[w.shape for w in weight['per_level']]}")

    print("\nlabels of the first anchor:")
    _, label = batch['anc']
    print(f"leaf label: {label['leaf'][0]}")
    print(f"binary label: {label['binary'][0]}")
    print(f"per_level label: {[y for y in label['per_level'][0]]}")

    """ Plotting the first batch

    from label_metric.models import Audio2LogMelSpec

    melspec = Audio2LogMelSpec(
        sr = 44100,
        n_fft = 2048,
        hop_length = 512
    )

    x_a, y_a = batch['anc']
    x_a = melspec(x_a)

    x_p, y_p = batch['pos']
    x_p = melspec(x_p)

    x_n, y_n = batch['neg']
    x_n = melspec(x_n)
    
    import matplotlib.pyplot as plt

    # Define the number of rows and columns
    nrows = len(x_a)
    ncols = 3 # a, p, n

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))

    # Plot each Mel spectrogram in the appropriate subplot
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if j == 0:
                mel_spectrogram, label = x_a[i], y_a[i]
            elif j == 1:
                mel_spectrogram, label = x_p[i], y_p[i]
            elif j == 2:
                mel_spectrogram, label = x_n[i], y_n[i]
            mel_spectrogram_np = mel_spectrogram.numpy()
            title = str(train_set.label_to_node(int(label)))
            title = '\n'.join(textwrap.wrap(title, 30))
            ax.imshow(mel_spectrogram_np, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            ax.label_outer()  # Only show outer labels to avoid clutter

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('batch_mel_spectrograms.png', dpi=300)

    """
