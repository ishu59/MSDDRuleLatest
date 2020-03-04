from itertools import product, permutations
from pprint import pprint
from ConvertStream import ConvertStream
from typing import Dict, Tuple, List, Union, Set
from MSDDNode import MSDDToken, MSDDNodeElement


class NewMSDDNode:
    def __init__(self, element):
        self.element = element
        self.children = []


class NewMSDDTrie:
    def __init__(self, ordering_list: List[List]):
        self.ordering_list = ordering_list
        self.ordering_list = [['A','B','C','D'],['1','2'],['X','Y']]
        self.ordering_list = ['A','B','C','D']
        self.root = NewMSDDNode('root')

    def create_tree(self):
        cur = self.root
        for a_index, elem in self.ordering_list:
            newNode = NewMSDDNode(elem)

