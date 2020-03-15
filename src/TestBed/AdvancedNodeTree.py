import sys

sys.path.append("../")
sys.path.append("../../")
from itertools import product, combinations
from pprint import pprint
from typing import Tuple, List, Dict, Set, Optional, Union, Sequence
from collections import namedtuple
from utils.pptree import print_tree
from collections import deque
from utils.CombinationTree import compute_comb_adv
from Trie.MSDDNode import MSDDToken


class TokenPair:
    def __init__(self, precursor: Sequence[Optional[MSDDToken]], successor: Sequence[Optional[MSDDToken]]):
        self.successor = successor
        self.precursor = precursor

    def __str__(self):
        return '(P:' + str(self.precursor) + ",S:" + str(self.successor) + ")"

    def __repr__(self):
        return self.__str__()


# Complete_Token = namedtuple('Complete_Token', 'prec, succ')


class AdvMSDDNode:
    def __init__(self, token, depth: int, count: int, parent=None, cost: float = 0, is_prec: bool = True):
        self.children:Dict[Tuple[MSDDToken],AdvMSDDNode] = {}
        self.token = token if token else ''
        self.cost: float = cost
        self.depth: int = depth
        self.count: int = count
        self.dead = False
        self.parent: AdvMSDDNode = parent
        self.final_token: TokenPair = None
        self.is_prec: bool = is_prec

    def set_final_token(self, p, s):
        self.final_token = TokenPair(precursor=p, successor=s)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def increase_count(self):
        self.count += 1

    def __repr__(self):
        return self.__str__()
        # return str({'Token': self.final_token})

    def __str__(self):
        return '(T:' + str(self.final_token) + ",C:" + str(self.count) + ")"


class AdvMSDDTrieStructure:
    def __init__(self, root_token=None):
        if root_token is None:
            root_token = [('',), ('',)]

        self.root = AdvMSDDNode(root_token, depth=0, count=1)
        self.root.set_final_token(root_token[0], root_token[1])

    def add_token(self, p_tok, s_tok, current_node: AdvMSDDNode = None):
        if current_node is None:
            current_node = self.root
        p_tok = sorted(p_tok) if len(p_tok) > 0 else []
        s_tok = sorted(s_tok) if len(s_tok) > 0 else []
        for token in p_tok:
            depth = current_node.depth
            if token in current_node.children and current_node.children[token].is_prec:
                current_node.children[token].increase_count()
            else:
                newNode = AdvMSDDNode(token, depth=(depth + 1), count=1, parent=current_node)
                current_node.children[token] = newNode
            current_node = current_node.children[token]
        for token in s_tok:
            depth = current_node.depth
            if token in current_node.children and not current_node.children[token].is_prec:
                current_node.children[token].increase_count()
            else:
                newNode = AdvMSDDNode(token, depth=(depth + 1), count=1, parent=current_node, is_prec=False)
                current_node.children[token] = newNode

            current_node = current_node.children[token]
        current_node.set_final_token(p=p_tok, s=s_tok)

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

    def get_all_children_nodes_tok(self, search_tok):
        prec, succ = search_tok
        node = self.search(prec)
        all_childs = []
        if node is None:
            node = self.root
        if node is not None:
            all_childs.append(node)
            for child in node.children.values():
                all_childs.extend(self.get_all_children_nodes(child))
        return all_childs

    def search(self, search_token):
        if len(search_token) < 1:
            return None
        cur = self.root
        for i, character in enumerate(search_token):
            if i == len(search_token):
                return cur
            if character not in cur.children:
                return None
            cur = cur.children[character]

        return cur

    def get_all_children_with_min_score(self, min_score, node=None):
        all_childs = []
        if node is None:
            node = self.root
        # else:
        #     node = self.search()
        if node is not None:
            for ps in list(zip(node.precursor_child, node.successor_child)):
                if node == self.root:
                    continue
                if node.cost > min_score:
                    all_childs.append(ps)
            for child in node.children.values():
                all_childs.extend(self.get_all_children(child))
            return all_childs

    def print_depth_first(self, starting_token: AdvMSDDNode = None):
        if starting_token is None:
            return
        cur: AdvMSDDNode = starting_token
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

    def reset_count(self, node: AdvMSDDNode = None):
        if node is None:
            node = self.root
        node.count = 0
        for n in node.precursor_child.values():
            self.reset_count(n)
        for n in node.successor_child.values():
            self.reset_count(n)


def test_new_trie():
    print('')
    t = AdvMSDDTrieStructure()

    prec = compute_comb_adv(['A', 'B'])
    succ = compute_comb_adv(['C', 'D'])
    for item in product(prec, succ):
        t.add_token(item[0], item[1])
    print_tree(t.root, childattr='children')


if __name__ == '__main__':
    test_new_trie()
