from itertools import combinations, permutations
import random
from typing import List, Tuple, Iterator
import logging

from anytree import Node, LevelOrderGroupIter
from torch.utils.data import Dataset, Sampler

from label_metric.utils.tree_utils import iter_parent_nodes, node_distance, tree_to_string

class SampleTripletsFromTree(Sampler):

    def __init__(
        self, 
        data: Dataset, 
        more_level: int,
        logger: logging.Logger
    ):

        self.data = data
        self.more_level = more_level
        self.logger = logger
        self.triplets = self.sample()
        self.logger.info(
            'Each training epoch has '
            f'{len(self.triplets)} triplets'
        )

    def __len__(self) -> int:
        return len(self.triplets)

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        self.triplets = self.sample()
        random.shuffle(self.triplets)
        return iter(self.triplets)

    def sample(self) -> List[Tuple[int, int, int]]:
        triplets = []
        for nodes in iter_parent_nodes(self.data.tree):
            for parent_node in nodes:
                self.logger.debug(f'visiting {parent_node}')
                self.logger.debug(tree_to_string(parent_node))
                triplets += self.sample_subtree(parent_node)
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
        # more level under pos
        more_level = min(root.height - 1, self.more_level)
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
                idx_a, idx_p = random.sample(self.data.node_to_index[nodes[0]], 2)
                idx_n = random.choice(self.data.node_to_index[neg])
                triplets.append((idx_a, idx_p, idx_n))
            else:
                for node_a, node_p in combinations(nodes, 2):
                    self.logger.debug(
                        f"x_a: from {node_a.name.split(' ')[0]}, "
                        f"x_p: from {node_p.name.split(' ')[0]}"
                    )
                    idx_a = random.choice(self.data.node_to_index[node_a])
                    idx_p = random.choice(self.data.node_to_index[node_p])
                    idx_n = random.choice(self.data.node_to_index[neg])
                    triplets.append((idx_a, idx_p, idx_n))
        return triplets


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
        logger = logger
    )

    sampler = SampleTripletsFromTree(
        data = train_set, 
        more_level = 0,
        logger = logger
    )

    print(f'the first triplet indices: {next(iter(sampler))}')

    train_loader = DataLoader(
        train_set,
        batch_size = 32,
        sampler = sampler,
        num_workers = 0,
        drop_last = True
    )
    
    batch = next(iter(train_loader))

    print("anchors in the first minibatch: "
          f"audio shape {batch['anc'][0].shape}, "
          f"label shape {batch['anc'][1].shape}")
    print("labels of the first triplet in the minibatch: "
          f"anchor label: {batch['anc'][1][0]}, "
          f"positive label: {batch['pos'][1][0]}, "
          f"negative label: {batch['neg'][1][0]}")
