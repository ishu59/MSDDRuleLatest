from itertools import product
from pprint import pprint

from ConvertStream import ConvertStream
from MSDDNode import MSDDToken, MSDDNodeElement
from typing import Dict, Tuple, List, Union, Set

# from SupportTrie import get_prec_succ, TrieStructure
from utils.pptree import print_tree
WILDCARD = '*'




class MSDDTrie:
    def __init__(self, p_wid, s_wid, delay):
        self.delay = delay
        self.s_wid = s_wid
        self.p_wid = p_wid
        self.root = MSDDNodeElement("root")
        self.member_dict = {}

    def find_best_parent(self, node):
        pass

    def create_msdd_token(self, token_chain, start_time):
        token_set = list()
        for time_stream_index, single_time_stream in enumerate(token_chain):
            for elem_index, elem in enumerate(single_time_stream):
                if elem != '*':
                    token_set.append(
                        MSDDToken(value=elem, stream_index=elem_index, time_offset=(start_time + time_stream_index)))
        token_set.sort()
        token_set = tuple(token_set)
        return token_set

    def add_token_chain(self, prec_tok, succ_tok=None, parent_node: MSDDNodeElement = None):
        if prec_tok is None:
            return
        if not parent_node:
            parent_node = self.root
        precursor = self.create_msdd_token(prec_tok, 0)

        if len(precursor) <1:
            return
        if precursor in parent_node.children:
            parent_node.children[precursor].increase_count()
        else:
            cur = MSDDNodeElement(precursor, depth=parent_node.depth + 1)
            parent_node.children[precursor] = cur
        parent_node = parent_node.children[precursor]
        self.add_token_chain(succ_tok, parent_node=parent_node)




def test_msdd_trie():
    print("Start")
    p =2
    s = 2
    d= 0
    trie = MSDDTrie(p,s,d)
    # test_data = ((('D', 'X'), ('A', 'Y'), ('B', 'Z')), (('D', 'X'), ('A', 'Y'), ('B', 'Z')))
    # test_data = ((('D', 'X'), ('A', 'Y'), ('B', 'Z')), (('D', 'X'), ('A', 'Y'), ('*', 'Z')),
    #              (('D', 'X'), ('A', 'Y'), ('B', '*')), (('D', 'X'), ('A', 'Y'), ('*', '*')),
    #              (('D', 'X'), ('*', 'Y'), ('B', 'Z')))

    data = [['D', 'X'], ['A', 'Y'], ['B', 'Z'], ['B', 'X'], ['A', 'Z'], ['B', 'Y']]
    myst = [[s] for s in 'ABCDABCD']
    s = ConvertStream(myst, p,s,d)
    data = s.get_res()
    for test_data in data:
        for t in test_data:
            trie.add_token_chain(t[:p], t[p:])
        # break

    print_tree(trie.root, childattr='children')
    print("End")


if __name__ == '__main__':
    print()
    test_msdd_trie()
