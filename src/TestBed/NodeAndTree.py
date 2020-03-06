import sys
sys.path.append("../")
sys.path.append("../../")
from itertools import product, combinations
from pprint import pprint
from typing import Tuple, List, Dict, Set
from collections import defaultdict
from utils.pptree import print_tree
from collections import deque

class NewMSDDNode:
    def __init__(self, token, depth: int, count: int, parent=None, cost: float = 0, is_leaf: bool = False):
        self.precursor_child = []
        self.successor_child = []
        self.token = token
        self.cost: float = cost
        self.depth: int = depth
        self.count: int = count
        self.parent: NewMSDDNode = parent
        self.final_token = self.compute_final_token()

    def compute_final_token(self):
        tok = []
        cur = self
        while cur.parent is not None:
            tok.append(cur.token)
            cur = cur.parent
        tok.reverse()
        return tok

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def increase_count(self):
        self.count += 1

    def __repr__(self):
        return self.__str__()
        # return str({'Token': self.final_token})

    def __str__(self):
        return ('(T:' + str(self.final_token)+",C:"+str(self.count)+ ")")
        # return ('(T:' + str(self.token) + ",C:" + str(self.count) + ",D:" + str(self.depth) + ")")


class NewMSDDTrieStructure:

    def add_token_to_node(self, node_tokens, current_node: NewNode = None):
        if current_node is None:
            current_node = self.root
        for token in node_tokens:
            depth = current_node.depth
            if token not in current_node.children:
                newNode = NewNode(token, depth=(depth + 1), count=1, parent=current_node)
                current_node.children[token] = newNode
            else:
                current_node.children[token].increase_count()
            current_node = current_node.children[token]

    def print_depth_first(self, starting_token: NewNode = None):
        if starting_token is None:
            return
        cur: NewNode = starting_token
        if cur.is_leaf():
            path_list = []
            while (cur is not None):
                path_list.append(cur.token)
                cur = cur.parent
            path_list.reverse()
            pprint(path_list)
        else:
            for child in cur.children.values():
                self.print_depth_first(child)

    def reset_count(self, node:NewNode = None):
        if node is None:
            node = self.root
        node.count = 0
        for n in node.children.values():
            self.reset_count(n)


    def get_all_children(self, node:NewNode = None):
        all_childs = []
        if node is None:
            node = self.root
        if node is not None:
            all_childs.append(node)
            for child in node.children.values():
                all_childs.extend(self.get_all_children(child))
            return all_childs

    def get_all_leafs(self, node:NewNode = None):
        all_childs = []
        if node is None:
            node = self.root
        if node is not None:
            if node.is_leaf():
                all_childs.append(node)
            for child in node.children.values():
                all_childs.extend(self.get_all_leafs(child))
            return all_childs

    def get_nodes_count(self):
        s = 0
        for child in self.root.children.values():
            s+=child.count
        return s

    def search(self, search_token, node=None):
        cur = node
        if not cur:
            cur = self.root
        for i, character in enumerate(search_token):
            if character == "*":
                if i == len(search_token) - 1:
                    # for child in cur.children.values():
                    #     if child.is_leaf():
                    #         return True
                    # return False
                    return cur
                for child in cur.children.values():
                    n = self.search(search_token[i+1:], child)
                    if self.search(search_token[i+1:], child):
                        return n
                return None
            if character not in cur.children:
                return False
            cur = cur.children[character]
        return cur

    def __init__(self, root_token = None):
        if root_token is None:
            root_token = "*"
        self.root = NewNode(root_token, depth=0, count=1)


def test_new_trie():
    trie = NewTrieStructure()
    ordering_list = ['A', 'B', 'C', 'D']
    # ordering_list = [x for x in "abcdexyz".capitalize()]
    for i in range(len(ordering_list)):
        for j in range(i + 1, len(ordering_list) + 1):
            if i - j == 0:
                continue
            for item in list(combinations(ordering_list[i:], j)):
                if len(item) > 0:
                    # print(item)
                    trie.add_token_to_node(item)
    search_token = ["A","B"]
    print_tree(trie.root, childattr='children')
    # search_token = ["*", "C","*"]
    search_token = ['A', 'A']
    print(trie.search(search_token))
    # pprint(trie.get_all_children(None))
    # for child in trie.get_all_children():


    # print_tree(trie.root, childattr='children')

if __name__ == '__main__':
    test_new_trie()