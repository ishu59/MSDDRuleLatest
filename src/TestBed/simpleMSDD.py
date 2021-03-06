import os, sys
from itertools import combinations, product
from pprint import pprint
import math
from NodeAndTree import NewMSDDTrieStructure
from utils.TokenGen import create_msdd_token
from utils.pptree import print_tree

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 'rules' module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from typing import List, Dict, Union, Tuple
from collections import defaultdict, Set
import numpy as np
from utils.CombinationTree import compute_comb
from Trie.MSDDNode import MSDDToken
from SIROrderingEnum import SIRS


class SimpleMSDD:

    def __init__(self, stream, wp, ws, delta, fn=None, k=1):
        self.stream = stream
        # stream = "ABCABDABAABDBDABCDAAADCABABABABABABABABABABABABABABABABABABABABABAABABABA"
        self.best = []
        self.k = k
        self.wp = wp
        self.ws = ws
        self.delta = delta
        self.min_score = -1 * math.inf
        self.open_list = [[[], []]]
        self.visited = []
        self.tree = NewMSDDTrieStructure()

    def compute_g_score(self, n1_x_y, n2_x_not_y, n3_not_x_y, n4_not_x_not_y):
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

    def f_max(self, node, stream, delta, wp, ws):
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
            prec = create_msdd_token(prec, 0)
            succ = create_msdd_token(succ, i + wp + delta)
            if set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
                n1 += 1
            elif set(search_p).issubset(set(prec)) and not set(search_s).issubset(set(succ)):
                n2 += 1
            elif not set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
                n3 += 1
            else:
                n4 += 1
        return max(self.compute_g_score(n1, n2, 0, n3 + n4), self.compute_g_score(0, n1 + n2, n3, n4))

    def f(self, node, stream, delta, wp, ws):
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
            prec = create_msdd_token(prec, 0)
            succ = create_msdd_token(succ, i + wp + delta)
            if set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
                n1 += 1
            elif set(search_p).issubset(set(prec)) and not set(search_s).issubset(set(succ)):
                n2 += 1
            elif not set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
                n3 += 1
            else:
                n4 += 1
        return self.compute_g_score(n1, n2, n3, n4)

    def validate_token(self, p, s):
        for pc in p:
            for sc in s:
                if sc < pc:
                    return False
        return True

    def systematic_expand(self, cur_node, wp, ws, delta, tp, ts):
        children = []
        prec_list = create_msdd_token(tp, start_time=0)
        succ_list = create_msdd_token(ts, start_time=wp + delta)
        prec = compute_comb(prec_list)
        succ = compute_comb(succ_list)
        for comb in product(prec, succ):
            p = comb[0]
            s = comb[-1]
            p_list = []
            s_list = []
            if self.validate_token(p, s):
                # if True:
                for pc in p:
                    p_list.append(pc)
                for sc in s:
                    s_list.append(sc)
                p_list = sorted(p_list)
                s_list = sorted(s_list)
                final_comb = [p_list, s_list]
                self.tree.add_both_token(p_list, s_list)
                if final_comb not in children:
                    children.append(final_comb)
        other = self.tree.get_all_children_nodes()
        # other = self.tree.get_all_children()
        # other = self.tree.get_all_children(cur_node)
        # other = self.tree.get_all_children_nodes_tok(cur_node)
        return other

    def create_tree(self):
        for i in range(len(self.stream)):
            if i + self.wp + self.delta + self.ws > len(self.stream):
                break
            tp = self.stream[i:i + self.wp]
            ts = self.stream[i + self.wp + self.delta:i + self.wp + self.delta + self.ws]
            prec_list = create_msdd_token(tp, start_time=0)
            succ_list = create_msdd_token(ts, start_time=self.wp + self.delta)
            prec = compute_comb(prec_list)
            succ = compute_comb(succ_list)
            for comb in product(prec, succ):
                p = comb[0]
                s = comb[-1]
                p_list = []
                s_list = []
                # if self.validate_token(p, s):
                if True:
                    for pc in p:
                        p_list.append(pc)
                    for sc in s:
                        s_list.append(sc)
                    p_list = sorted(p_list)
                    s_list = sorted(s_list)
                    final_comb = [p_list, s_list]
                    self.tree.add_both_token(p_list, s_list)
        print_tree(self.tree.root)

    def systematic_expand_pre_build_tree(self, cur_node, wp, ws, delta, tp, ts):
        # other = self.tree.get_all_children_nodes_tok(cur_node)
        other = self.tree.get_all_children_nodes()
        return list(other)

    def remove_prunable(self, children, stream, delta, wp, ws, min_score):
        child_nodes = []
        # for c in children:
        #     child_nodes.append(self.tree.get_all_children_nodes_tok(c))
        child_nodes = [child for child in children if child != self.tree.root]

        # child_nodes = self.tree.get_all_children_nodes()
        for node in child_nodes:
            if node == self.tree.root:
                continue
            if len(node.children) > 0:
                pre_suc = list(zip(node.precursor_child, node.successor_child))
                node_score = []
                for ps in pre_suc:
                    node_score.append(self.f_max(ps, stream, delta, wp, ws))
                if len(node_score) > 0:
                    node.cost = min(node_score)

        # children = self.tree.get_all_children_with_min_score(min_score)
        children2 = [zip(node.precursor_child, node.successor_child) for node in child_nodes if node.cost > min_score]
        children2 = [item for sublist in children2 for item in sublist]
        children = children2
        return children

    def run2(self):
        for i in range(len(self.stream)):
            if i + self.wp + self.delta + self.ws > len(self.stream):
                break
            while len(self.open_list) > 0:

                cur_node = self.open_list.pop(0)
                self.visited.append(cur_node)
                children = self.systematic_expand(cur_node, self.wp, self.ws, self.delta,
                                                  self.stream[i:i + self.wp],
                                                  self.stream[
                                                  i + self.wp + self.delta:i + self.wp + self.delta + self.ws])
                # print(children)
                # children = self.remove_prunable(children, self.stream, self.delta, self.wp, self.ws, self.min_score)
                for child in children:
                    if child not in self.visited:
                        self.open_list.append(child)

                for child in children:
                    n = [{}, {}]
                    score = self.f(child, self.stream, self.delta, self.wp, self.ws)

                    if {"child": child, 'score': score} in self.best:
                        continue
                    if len(self.best) < self.k:
                        self.best.append({"child": child, 'score': score})
                    else:
                        self.best = sorted(self.best, key=lambda k: k['score'], reverse=True)
                        if score > self.best[-1]['score']:
                            self.best.pop(-1)
                            self.best.append({"child": child, 'score': score})
                            self.best = sorted(self.best, key=lambda k: k['score'], reverse=True)
                            self.min_score = min(self.min_score, self.best[-1]['score'])

        pprint(self.best)
        print(len(self.best))

    def run(self):
        for i in range(len(self.stream)):
            if i + self.wp + self.delta + self.ws > len(self.stream):
                break
            while len(self.open_list) > 0:

                cur_node = self.open_list.pop(0)

                self.visited.append(cur_node)
                children = self.systematic_expand(cur_node, self.wp, self.ws, self.delta, self.stream[i:i + self.wp],
                                                  self.stream[
                                                  i + self.wp + self.delta:i + self.wp + self.delta + self.ws])
                # print(children)
                children = self.remove_prunable(children, self.stream, self.delta, self.wp, self.ws, self.min_score)
                for child in children:
                    if child not in self.visited:
                        self.open_list.append(child)

                for child in children:
                    n = [{}, {}]
                    score = self.f(child, self.stream, self.delta, self.wp, self.ws)

                    if {"child": child, 'score': score} in self.best:
                        continue
                    if len(self.best) < self.k:
                        self.best.append({"child": child, 'score': score})
                    else:
                        self.best = sorted(self.best, key=lambda k: k['score'], reverse=True)
                        if score > self.best[-1]['score']:
                            self.best.pop(-1)
                            self.best.append({"child": child, 'score': score})
                            self.best = sorted(self.best, key=lambda k: k['score'], reverse=True)
                            self.min_score = min(self.min_score, self.best[-1]['score'])

        # pprint(self.best)
        # print(len(self.best))
        self.best = sorted(self.best, key=lambda k: k['score'], reverse=True)
        return self.best

    # def run_ensemble(self):
    #     for i in range(len(self.stream)):
    #
    #         if i + self.wp + self.delta + self.ws > len(self.stream):
    #             break
    #         while len(self.open_list) > 0:
    #
    #             cur_node = self.open_list.pop(0)
    #             self.visited.append(cur_node)
    #             children = self.systematic_expand(cur_node, self.wp, self.ws, self.delta, self.stream[i:i + self.wp],
    #                                               self.stream[
    #                                               i + self.wp + self.delta:i + self.wp + self.delta + self.ws])
    #             # print(children)
    #             children = self.remove_prunable(children, self.stream, self.delta, self.wp, self.ws, self.min_score)
    #             for child in children:
    #                 if child not in self.visited:
    #                     self.open_list.append(child)
    #
    #             for child in children:
    #                 if child == self.tree.root:
    #                     continue
    #                 n = [{}, {}]
    #                 score = self.f(child, self.stream, self.delta, self.wp, self.ws)
    #
    #                 if {"child": child, 'score': score} in self.best:
    #                     continue
    #                 if len(self.best) < self.k:
    #                     self.best.append({"child": child, 'score': score})
    #                 else:
    #                     self.best = sorted(self.best, key=lambda k: k['score'], reverse=True)
    #                     if score > self.best[-1]['score']:
    #                         self.best.pop(-1)
    #                         self.best.append({"child": child, 'score': score})
    #                         self.min_score = min(self.min_score, self.best[-1]['score'])
    #
    #     pprint(self.best)
    #     print(len(self.best))
    #     return self.best


def test1():
    np.random.seed(0)
    s0 = np.random.choice(['A', 'B', 'C', 'D'], size=30)
    s1 = np.random.choice([1, 2], size=30)
    stream = list(zip(s0, s1))
    k = 2
    wp = 2
    ws = 2
    delta = 0

    c = SimpleMSDD(stream, wp, ws, delta, fn=None, k=k)
    rules = c.run()
    pprint(rules)


def test_Simple_msdd():
    np.random.seed(0)
    s0 = np.random.choice(['A', 'B', 'C', 'D'], size=30)
    s1 = np.random.choice([1, 2], size=30)
    stream = list(zip(s0, s1))
    # sir = [c for c in 'SIRSIRSIRSIRSIRS']
    #
    # stream = []
    #
    # for item in sir:def test_Simple_msdd():
    np.random.seed(0)
    s0 = np.random.choice(['A', 'B', 'C', 'D'], size=30)
    s1 = np.random.choice([1, 2], size=30)
    stream = list(zip(s0, s1))
    k = 2
    wp = 2
    ws = 2
    delta = 0

    c = SimpleMSDD(stream, wp, ws, delta, fn=None, k=k)
    rules = c.run()
    pprint(rules)


def test_code():
    sir = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'I', 'I', 'I', 'R', 'R', 'R', 'R',
           'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'S', 'I', 'R', 'R',
           'R', 'S', 'S', 'S', 'S', 'S', 'S']
    stream = sir
    encoded_stream = []
    for item in stream:
        encoded_stream.append(SIRS[item])
    print(encoded_stream)


if __name__ == '__main__':
    test_Simple_msdd()
    print()
