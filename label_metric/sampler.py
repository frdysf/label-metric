from itertools import combinations, permutations

from torch.utils.data import Sampler

from label_metric.utils.tree_utils import iter_parent_nodes

class TreeSampler(Sampler):

    def __init__(self, data):
        self.data = data

    def __len__(self) -> int:
        pass

    def __iter__(self):
        
        for level, nodes in enumerate(iter_parent_nodes(self.data.tree)):
            for parent_node in nodes:
                for pos, neg in permutations(parent_node.children, 2):
                    yield(pos, neg)

if __name__ == '__main__':
    from label_metric.dataset import OrchideaSOL
    dataset = OrchideaSOL()
    sampler = TreeSampler(dataset)
    for i, x in enumerate(sampler):
        print(i, x)