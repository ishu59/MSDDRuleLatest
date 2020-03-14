



import os, sys
from itertools import combinations, product
from pprint import pprint
import math
from TestBed.NodeAndTree import NewMSDDTrieStructure

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 'rules' module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from typing import List, Dict, Union, Tuple
from collections import defaultdict, Set
import numpy as np
from utils.CombinationTree import compute_comb
from Trie.MSDDNode import MSDDToken

def validate_token(p, s):
    for pc in p:
        for sc in s:
            if sc < pc:
                return False
    return True


def create_msdd_token_1d(stream_list, start_time):
    msdd_list = []
    for i, s in enumerate(stream_list):
        msdd_list.append(MSDDToken(value=s, stream_index=0, time_offset=start_time + i))
    return msdd_list


def create_msdd_token(stream_list, start_time):
    msdd_list = []
    if all(isinstance(e, (list, tuple)) for e in stream_list):
        for i, s in enumerate(stream_list):
            for j, t in s:
                msdd_list.append(MSDDToken(value=t, stream_index=j, time_offset=start_time + i))
        return msdd_list
    else:
        for i, s in enumerate(stream_list):
            msdd_list.append(MSDDToken(value=s, stream_index=0, time_offset=start_time + i))
        return msdd_list


def systematic_expand(cur_node, wp, ws, delta, tp, ts):
    # prec, succ = cur_node
    children = []
    prec_list = []
    succ_list = []
    for i, tok in enumerate(tp):
        prec_list.append(MSDDToken(value=tok, time_offset=i, stream_index=0))
    for i, tok in enumerate(ts):
        succ_list.append(MSDDToken(value=tok, time_offset=i + wp + delta, stream_index=0))
    prec = compute_comb(prec_list)
    succ = compute_comb(succ_list)

    for comb in product(prec, succ):
        p = comb[0]
        s = comb[-1]
        p_list = []
        s_list = []
        if validate_token(p, s):
            for pc in p:
                p_list.append(pc)
            for sc in s:
                s_list.append(sc)
            p_list = sorted(p_list)
            s_list = sorted(s_list)
            final_comb = [p_list, s_list]
            tree.add_both_token(p_list, s_list)
            if final_comb not in children:
                children.append(final_comb)
    other = tree.get_all_children()
    return other


def remove_prunable(children, stream, delta, wp, ws, min_score):
    child_nodes = tree.get_all_children_nodes()
    for node in child_nodes:
        if node == tree.root:
            continue
        if len(node.children) > 0:
            pre_suc = list(zip(node.precursor_child, node.successor_child))
            node_score = []
            for ps in pre_suc:
                node_score.append(f_max(ps, stream, delta, wp, ws))
            if len(node_score) > 0:
                node.cost = min(node_score)
    children = tree.get_all_children_with_min_score(min_score)
    return children


def compute_g_score(n1_x_y, n2_x_not_y, n3_not_x_y, n4_not_x_not_y):
    total = n1_x_y + n2_x_not_y + n3_not_x_y + n4_not_x_not_y
    n1_x_y = 1 if n1_x_y == 0 else n1_x_y
    n2_x_not_y = 1 if n2_x_not_y == 0 else n2_x_not_y
    n3_not_x_y = 1 if n3_not_x_y == 0 else n3_not_x_y
    n4_not_x_not_y = 1 if n4_not_x_not_y == 0 else n4_not_x_not_y
    if total == 0:
        return 0
    n1_hat = (n1_x_y + n2_x_not_y) * (n1_x_y + n3_not_x_y) / total
    n2_hat = (n1_x_y + n2_x_not_y) * (n2_x_not_y + n4_not_x_not_y) / total
    n3_hat = (n3_not_x_y + n4_not_x_not_y) * (n1_x_y + n3_not_x_y) / total
    n4_hat = (n3_not_x_y + n4_not_x_not_y) * (n2_x_not_y + n4_not_x_not_y) / total
    g_score = 2 * (n1_x_y * np.log(n1_x_y / n1_hat) +
                   (n2_x_not_y * np.log(n2_x_not_y / n2_hat)) +
                   (n3_not_x_y * np.log(n3_not_x_y / n3_hat))
                   + (n4_not_x_not_y * np.log(n4_not_x_not_y / n4_hat)))
    return g_score


def f_max(node, stream, delta, wp, ws):
    search_p, search_s = node
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    state = [0, 0]
    for i in range(len(stream)):
        if i + wp + delta + ws > len(stream):
            break
        prec, succ = stream[i:i + wp], stream[i + wp + delta:i + wp + delta + ws]
        prec = create_msdd_token_1d(prec, 0)
        succ = create_msdd_token_1d(succ, i + wp + delta)
        if set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
            n1 += 1
        elif set(search_p).issubset(set(prec)) and not set(search_s).issubset(set(succ)):
            n2 += 1
        elif not set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
            n3 += 1
        else:
            n4 += 1
    return max(compute_g_score(n1, n2, 0, n3 + n4), compute_g_score(0, n1 + n2, n3, n4))


def f(node, stream, delta, wp, ws):
    search_p, search_s = node
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    state = [0, 0]
    for i in range(len(stream)):
        if i + wp + delta + ws > len(stream):
            break
        prec, succ = stream[i:i + wp], stream[i + wp + delta:i + wp + delta + ws]
        prec = create_msdd_token_1d(prec, 0)
        succ = create_msdd_token_1d(succ, i + wp + delta)
        if set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
            n1 += 1
        elif set(search_p).issubset(set(prec)) and not set(search_s).issubset(set(succ)):
            n2 += 1
        elif not set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
            n3 += 1
        else:
            n4 += 1
    return compute_g_score(n1, n2, n3, n4)


stream = "ABCABDABAABDBDABCDAAADCABABABABABABABABABABABABABABABABABABABABABAABABABA"

best = []
k = 2
wp = 2
ws = 2
delta = 0
min_score = -1 * math.inf
open_list = [[[], []]]
visited = []

tree = NewMSDDTrieStructure()

for i in range(len(stream)):
    if i + wp + delta + ws > len(stream):
        break
    while len(open_list) > 0:

        cur_node = open_list.pop(0)
        visited.append(cur_node)
        children = systematic_expand(cur_node, wp, ws, delta, stream[i:i + wp],
                                     stream[i + wp + delta:i + wp + delta + ws])
        # print(children)
        children = remove_prunable(children, stream, delta, wp, ws, min_score)
        for child in children:
            if child not in visited:
                open_list.append(child)

        for child in children:
            n = [{}, {}]
            score = f(child, stream, delta, wp, ws)

            if {"child": child, 'score': score} in best:
                continue
            if len(best) < k:
                best.append({"child": child, 'score': score})
            else:
                best = sorted(best, key=lambda k: k['score'], reverse=True)
                if score > best[-1]['score']:
                    best.pop(-1)
                    best.append({"child": child, 'score': score})
                    min_score = min(min_score, best[-1]['score'])

pprint(best)
print(len(best))

for i in range(len(stream)):
    if i + wp + delta + ws >= len(stream):
        break
    p = stream[i:i + wp]
    s = stream[i + wp + delta:i + wp + delta + ws]


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
