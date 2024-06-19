from itertools import combinations, permutations
import random
from typing import List, Tuple
import logging

from anytree import Node, LevelOrderGroupIter
from torch.utils.data import Dataset, Sampler

from label_metric.utils.tree_utils import iter_parent_nodes, node_distance, tree_to_string
from label_metric.utils.log_utils import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

class TreeSampler(Sampler):

    def __init__(self, data: Dataset, more_level: int) -> None:
        self.data = data
        self.more_level = more_level
        self.triplets = self.sample()
        logger.info(
            f'{len(self.triplets)} triplets '
            'are sampled for each training epoch'
        )

    def __len__(self) -> int:
        return len(self.triplets)

    def __iter__(self):
        random.shuffle(self.triplets)
        return iter(self.triplets)

    def sample(self) -> List[Tuple]:
        triplets = []
        for nodes in iter_parent_nodes(self.data.tree):
            for parent_node in nodes:
                logger.debug(f'visiting {parent_node}')
                logger.debug(tree_to_string(parent_node))
                triplets += self.sample_subtree(parent_node)
        return triplets
                
    def sample_subtree(self, root: Node) -> List[Tuple]:
        """
        sample  pos and neg from root.children
                x_a and x_p from pos
                x_n from neg
        """
        triplets = []
        if len(root.children) == 1:
            return triplets
        # more level starts from pos
        more_level = min(root.height - 1, self.more_level)
        for pos, neg in permutations(root.children, 2):
            logger.debug(
                f"pos: {pos.name.split(' ')[0]}, "
                f"neg: {neg.name.split(' ')[0]}"
            )
            shallow_leaf_nodes = [node for node in pos.leaves \
                                  if node_distance(pos, node) < more_level]
            more_level_nodes = list(list(LevelOrderGroupIter(pos))[more_level])
            nodes = shallow_leaf_nodes + more_level_nodes
            logger.debug(f"within pos: {[node.name.split(' ')[0] for node in nodes]}")
            if len(nodes) == 1: # nodes = [pos]
                idx_a, idx_p = random.sample(self.data.node_to_index[nodes[0]], 2)
                idx_n = random.choice(self.data.node_to_index[neg])
                triplets.append((idx_a, idx_p, idx_n))
            else:
                for node_a, node_p in combinations(nodes, 2):
                    logger.debug(
                        f"x_a: {node_a.name.split(' ')[0]}, "
                        f"x_p: {node_p.name.split(' ')[0]}"
                    )
                    idx_a = random.choice(self.data.node_to_index[node_a])
                    idx_p = random.choice(self.data.node_to_index[node_p])
                    idx_n = random.choice(self.data.node_to_index[neg])
                    triplets.append((idx_a, idx_p, idx_n))
        return triplets


if __name__ == '__main__':
    
    import lightning as L
    from torch.utils.data import DataLoader
    from label_metric.datasets import TripletOrchideaSOL, BasicOrchideaSOL

    L.seed_everything(2024)

    train_set = TripletOrchideaSOL('train')
    sampler = TreeSampler(train_set, more_level = 1)
    train_loader = DataLoader(train_set,
                              batch_size = 32,
                              sampler = sampler,
                              num_workers = 2,
                              drop_last = False)
    x, y = next(iter(train_loader))['anc']
    print(x.shape, y.shape)
    
    # valid_set = BasicOrchideaSOL('valid')
    # valid_loader = DataLoader(valid_set,
    #                           batch_size = 2,
    #                           num_workers = 0,
    #                           shuffle = True,
    #                           drop_last = False)
    # x, y = next(iter(valid_loader))
    # print(x, y)
