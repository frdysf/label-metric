from anytree import Node, RenderTree, LevelOrderGroupIter
from anytree.walker import Walker
from typing import Optional, Tuple, List
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


def node_distance(node1: Node, 
                  node2: Node, 
                  type: str = 'sum') -> int:
    """
    Calculate the distance between two nodes in a tree.
    
    Parameters
    ----------
    node1 : Node
        The first node.
    node2 : Node
        The second node.
    type : str
        The type of distance to calculate. Either 'sum' or 'max'.
        'sum' calculates the sum of the distances to the lowest common ancestor.
        'max' calculates the maximum distance to the lowest common ancestor.
    
    Returns
    -------
    distance : int
        The distance between the two nodes.
    """
    w = Walker()
    upwards, _, downwards = w.walk(node1, node2)
    if type == 'sum':
        return len(upwards) + len(downwards)
    elif type == 'max':
        return max(len(upwards), len(downwards))
    else:
        raise ValueError("type must be either 'sum' or 'max'.")


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
    importer = DictImporter()
    exporter = DictExporter()
    tree_dict = exporter.export(tree)
    new_tree = importer.import_(tree_dict)
    return new_tree


if __name__ == '__main__':

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

    print(tree_to_string(root))
    
    print('sum dist between e11 and i:', node_distance(e11, i, type='sum'))
    print('max dist between e11 and i:', node_distance(e11, i, type='max'))

    print('Iterate over all parent nodes:')
    
    levels = iter_parent_nodes(root)
    for i, level in enumerate(levels):
        print(f'this is the {i}th level')
        for node in level:
            print('the parent node: ', node)
            print('children: ', node.children)
    
    