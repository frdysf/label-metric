from typing import Set, Tuple, List, Dict, Optional

from anytree import Node, RenderTree, LevelOrderIter, LevelOrderGroupIter
from anytree.walker import Walker
from anytree.importer import DictImporter
from anytree.exporter import DictExporter


def tree_to_string(root: Node) -> str:
    """
    Return the tree structure as a string.
    """
    lines = []
    for pre, _, node in RenderTree(root):
        lines.append(pre + node.name)
    return "\n".join(lines)


def node_distance(node1: Node, node2: Node, dist_type: str = 'sum') -> int:
    """
    Calculate the distance between two nodes in a tree.
    
    Parameters
    ----------
    node1 : Node
        The first node.
    node2 : Node
        The second node.
    dist_type : str
        The type of distance to calculate. Either 'sum' or 'max'.
        'sum' calculates the sum of the distances to the lowest common ancestor.
        'max' calculates the maximum distance to the lowest common ancestor.
    
    Returns
    -------
    distance : int
        The distance between the two nodes.
    """
    assert dist_type in ['sum', 'max']
    w = Walker()
    upwards, _, downwards = w.walk(node1, node2)
    if dist_type == 'sum':
        return len(upwards) + len(downwards)
    elif dist_type == 'max':
        return max(len(upwards), len(downwards))


class NodeAffinity():
    def __init__(self, any_node: Node):
        self.any_node = any_node
        self.tree_height = any_node.root.height
        self.tree_diameter = self._get_diameter(any_node)

    def __call__(self, node1: Node, node2: Node, dist_type: str) -> float:
        dist = self._get_dist(node1, node2, dist_type)
        if dist_type == 'max':
            return 1 - dist / self.tree_height
        elif dist_type == 'sum':
            return 1 - dist / self.tree_diameter

    def _get_dist(self, node1: Node, node2: Node, dist_type: str) -> int:
        assert node1.root is self.any_node.root
        return node_distance(node1, node2, dist_type)

    def _search(self, node: Node) -> List[Node]:
        c_list = list(node.children)
        p_list = [node.parent] if node.parent else []
        return c_list + p_list

    def _dfs(self, node: Node, dist: int, visited: Set[Node]) -> Tuple[Node, int]:
        farthest_node = node
        max_dist = dist
        visited.add(node)
        search_list = self._search(node)
        for cur_node in search_list:
            if cur_node not in visited:
                cur_farthest_node, cur_max_dist = self._dfs(cur_node, dist+1, visited)
                if cur_max_dist > max_dist:
                    max_dist = cur_max_dist
                    farthest_node = cur_farthest_node
        return farthest_node, max_dist

    def _get_diameter(self, node: Node) -> int:
        visited = set()
        farthest_node, _ = self._dfs(node, 0, visited)
        visited = set()
        _, diameter = self._dfs(farthest_node, 0, visited)
        return diameter


def iter_parent_nodes(root: Node, maxlevel: Optional[int] = None) -> List[Tuple[Node]]:
    """
    Iterate over all parent nodes in the tree, level by level, up to a certain level.
    
    Parameters
    ----------
    root : Node
        The root node of the tree.
    maxlevel : int, None
        The maximum level to iterate over. If None, iterate over all levels.
    
    Returns
    -------
    List[Tuple[Node]]
        A list of tuples, where the i-th tuple contains the nodes at level i.
    """
    return [nodes for nodes in LevelOrderGroupIter(root, filter_=lambda n: not n.is_leaf, 
                                                   maxlevel=maxlevel) if len(nodes)]
    

def copy_tree(tree: Node) -> Node:
    importer = DictImporter(nodecls=Node)
    exporter = DictExporter()
    tree_dict = exporter.export(tree)
    new_tree = importer.import_(tree_dict)
    return new_tree


def prune_tree(root: Node, leaves_to_prune: List[Node]) -> Dict[Node, Node]:
    pruned_edges = {}
    all_leaves = set(root.leaves)
    for leaf in leaves_to_prune:
        assert leaf.root is root
        pruned_edges[leaf] = leaf.parent
        leaf.parent = None # detach from parent
    for node in LevelOrderIter(root):
        if node in all_leaves:
            continue
        # detach if all leaves under this node have been pruned
        if not bool(set(node.leaves) & all_leaves):
            pruned_edges[node] = node.parent
            node.parent = None
    return pruned_edges


def repair_tree(pruned_edges: Dict[Node, Node]):
    for child, parent in pruned_edges.items():
        child.parent = parent


if __name__ == '__main__':

    # example code

    root = Node("root")
    a = Node("a", parent=root)
    b = Node("b", parent=root)
    c = Node("c", parent=root)
    d = Node("d", parent=a)
    e = Node("e", parent=a)
    d1 = Node("d1", parent=d)
    d2 = Node("d2", parent=d)
    d3 = Node("d3", parent=d)
    e1 = Node("e1", parent=e)
    e11 = Node("e11", parent=e1)
    e12 = Node("e12", parent=e1)
    f = Node("f", parent=b)
    g = Node("g", parent=b)
    h = Node("h", parent=c)
    i = Node("i", parent=c)
    i1 = Node("i1", parent=i)
    i11 = Node("i11", parent=i1)

    # 'detach and re-attach' 
    #  - may change the element order in root.leaves
    #  - won't change the hash value of nodes
    print(root.leaves)
    temp_hash_value = hash(e11)
    e11.parent = None
    assert hash(e11) == temp_hash_value
    e11.parent = e1
    assert hash(e11) == temp_hash_value
    print(root.leaves)

    print(tree_to_string(root))

    node_affinity = NodeAffinity(root)
    print('tree height: ', node_affinity.tree_height)
    print('tree diameter: ', node_affinity.tree_diameter)
    
    # e11 vs all other nodes
    for node in LevelOrderIter(root):
        print(f'between {e11} and {node}')
        print('positive max rel:', node_affinity(e11, node, 'max'))
        print('positive sum rel:', node_affinity(e11, node, 'sum'))

    print('Iterate over all parent nodes:')
    
    levels = iter_parent_nodes(root)
    for i, level in enumerate(levels):
        print(f'this is the {i}th level')
        for node in level:
            print('the parent node: ', node)
            print('children: ', node.children)
    