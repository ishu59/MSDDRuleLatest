from itertools import combinations
from typing import List
# from MSDDNode import MSDDToken, MSDDNodeElement
from Trie.SuffixTree import SuffixTree

class NewMSDDNode:
    def __init__(self, element):
        self.element = element
        self.children = []


class NewMSDDTrie:
    def __init__(self, ordering_list: List[List]):
        self.ordering_list = ordering_list
        self.ordering_list = [['A', 'B', 'C', 'D'], ['1', '2'], ['X', 'Y']]
        self.ordering_list = ['A', 'B', 'C', 'D']
        self.root = NewMSDDNode('root')

    def create_tree(self):
        cur = self.root
        for a_index, elem in self.ordering_list:
            newNode = NewMSDDNode(elem)

tree = SuffixTree()
ordering_list = ['A', 'B', 'C', 'D']
for i in range(len(ordering_list)):
    for j in range(i + 1, len(ordering_list) + 1):
        if i - j == 0:
            continue
        for item in list(combinations(ordering_list[i:], j)):
            if len(item) > 0:
                print(item)
                tree.addSuffix(item)

tree.visualize()

