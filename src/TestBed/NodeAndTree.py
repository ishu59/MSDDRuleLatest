import sys

sys.path.append("../")
sys.path.append("../../")
from itertools import product, combinations
from pprint import pprint
from typing import Tuple, List, Dict, Set
from collections import defaultdict
from utils.pptree import print_tree
from collections import deque
from Trie.MSDDNode import MSDDToken


class NewMSDDNode:
    def __init__(self, token, depth: int, count: int, parent=None, cost: float = 0, is_leaf: bool = False):
        self.children = {}
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
        return '(T:' + str(self.final_token) + ",C:" + str(self.count) + ")"
        # return ('(T:' + str(self.token) + ",C:" + str(self.count) + ",D:" + str(self.depth) + ")")


class NewMSDDTrieStructure:
    def __init__(self, root_token=None):
        if root_token is None:
            root_token = "*"
        self.root = NewMSDDNode(root_token, depth=0, count=1)

    def _add_token_to_node(self, node_tokens, current_node: NewMSDDNode = None):
        if current_node is None:
            current_node = self.root
        for token in node_tokens:
            depth = current_node.depth
            if token not in current_node.children:
                newNode = NewMSDDNode(token, depth=(depth + 1), count=1, parent=current_node)
                current_node.children[token] = newNode
            else:
                current_node.children[token].increase_count()
            current_node = current_node.children[token]
        current_node.precursor_child.append()

    def add_both_token(self, p_tok, s_tok, current_node: NewMSDDNode = None):
        if current_node is None:
            current_node = self.root
        p_tok = sorted(p_tok) if len(p_tok) > 0 else []
        s_tok = sorted(s_tok) if len(s_tok) > 0 else []
        for token in p_tok:
            depth = current_node.depth
            if token not in current_node.children:
                newNode = NewMSDDNode(token, depth=(depth + 1), count=1, parent=current_node)
                current_node.children[token] = newNode
            else:
                current_node.children[token].increase_count()
            current_node = current_node.children[token]
        current_node.precursor_child.append(p_tok)
        current_node.successor_child.append(s_tok)

    def get_all_children(self, node=None):
        all_childs = []
        if node is None:
            node = self.root
        if node is not None:
            for ps in list(zip(node.precursor_child, node.successor_child)):
                all_childs.append(ps)
            for child in node.children.values():
                all_childs.extend(self.get_all_children(child))
            return all_childs

    def get_all_children_nodes(self, node=None):
        all_childs = []
        if node is None:
            node = self.root
        if node is not None:
            # for ps in list(zip(node.precursor_child, node.successor_child)):
            all_childs.append(node)
            for child in node.children.values():
                all_childs.extend(self.get_all_children_nodes(child))
            return all_childs

    def get_all_children_with_min_score(self, min_score, node=None):
        all_childs = []
        if node is None:
            node = self.root
        if node is not None:
            for ps in list(zip(node.precursor_child, node.successor_child)):
                if node == self.root:
                    continue
                if node.cost > min_score:
                    all_childs.append(ps)
            for child in node.children.values():
                all_childs.extend(self.get_all_children(child))
            return all_childs

    def print_depth_first(self, starting_token: NewMSDDNode = None):
        if starting_token is None:
            return
        cur: NewMSDDNode = starting_token
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

    def reset_count(self, node: NewMSDDNode = None):
        if node is None:
            node = self.root
        node.count = 0
        for n in node.precursor_child.values():
            self.reset_count(n)
        for n in node.successor_child.values():
            self.reset_count(n)

    # def get_all_children(self, node: NewMSDDNode = None):
    #     all_childs = []
    #     if node is None:
    #         node = self.root
    #     if node is not None:
    #         all_childs.append(node)
    #         for child in node.children.values():
    #             all_childs.extend(self.get_all_children(child))
    #         return all_childs

    # def get_all_leafs(self, node: NewMSDDNode = None):
    #     all_childs = []
    #     if node is None:
    #         node = self.root
    #     if node is not None:
    #         if node.is_leaf():
    #             all_childs.append(node)
    #         for child in node.children.values():
    #             all_childs.extend(self.get_all_leafs(child))
    #         return all_childs

    # def get_nodes_count(self):
    #     s = 0
    #     for child in self.root.children.values():
    #         s += child.count
    #     return s

    # def search(self, search_token, node=None):
    #     cur = node
    #     if not cur:
    #         cur = self.root
    #     for i, character in enumerate(search_token):
    #         if character == "*":
    #             if i == len(search_token) - 1:
    #                 # for child in cur.children.values():
    #                 #     if child.is_leaf():
    #                 #         return True
    #                 # return False
    #                 return cur
    #             for child in cur.children.values():
    #                 n = self.search(search_token[i + 1:], child)
    #                 if self.search(search_token[i + 1:], child):
    #                     return n
    #             return None
    #         if character not in cur.children:
    #             return False
    #         cur = cur.children[character]
    #     return cur
    #


def test_new_trie():
    print('hello')


if __name__ == '__main__':
    test_new_trie()
