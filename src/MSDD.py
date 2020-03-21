from pprint import pprint
from typing import List, Dict, Tuple
import heapq
import numpy as np
from itertools import product

from SIROrderingEnum import SIRS
from Trie.AnotherTrie import NewTrieStructure
from utils.pptree import print_tree
from utils.CombinationTree import compute_comb
WILDCARD = '*'

def getTrie(ordered_tree_node_str):
    tree = NewTrieStructure("*")
    tree_str_list = compute_comb([x for x in ordered_tree_node_str])
    for item in tree_str_list:
        tree.add_token_to_node(item)
    return tree

def expand_data(data_list: List):
    pass

def systematic_expand():
    pass

def removePrunable():
    pass

def get_row_combination(data:List[str])->List[List[str]]:
    if len(data) == 1:
        return [[data[0]],[ WILDCARD]]
    curr_list = get_row_combination(data[1:])
    result_list = []
    for index in range(len(curr_list)):
        a =  [[data[0]]+curr_list[index]]
        b = [[WILDCARD] + curr_list[index]]
        result_list.extend(a)
        result_list.extend(b)
    # result_tuple = tuple(tuple(x) for x in result_list)
    return result_list

def convert_tuple(data: List[List[List[str]]])->Tuple[Tuple[Tuple[str]]]:
    return tuple(tuple(tuple(x) for x in inner_list) for inner_list in data)

def get_all_combination(data:List[List[str]])->List[List[List[str]]]:
    keeper = []
    for row in data:
        res = get_row_combination(row)
        keeper.append(res)
    if len(data) > 1:
        output = []
        for item in product(*keeper):
            output.append((item))
        return convert_tuple(output)
    else:
        return convert_tuple(keeper)

def msdd_algorithm_simple(sequence_data, precursor_width, successor_width, lagtime_between,
                          num_rules=None,dependency_evaluator_fn=None):
    best = []
    openHeap = []
    precursors_list =  get_all_combination(sequence_data[:precursor_width])
    successor_List = get_all_combination(sequence_data[precursor_width+lagtime_between,precursor_width+lagtime_between+successor_width])
    tree = getTrie("ABCD")
    print_tree(tree.root)

def compute_g_score(n1_x_y, n2_x_not_y, n3_not_x_y, n4_not_x_not_y):
    total = n1_x_y + n2_x_not_y + n3_not_x_y + n4_not_x_not_y
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

def test_msdd():
    best = []
    k = 2
    wp = 2
    ws =2
    delta = 0
    stream_str = "BACD"
    stream_str = sorted(stream_str)
    tree = getTrie(stream_str)
    tree.reset_count()
    tree= NewTrieStructure()
    print_tree(tree.root)
    stream = "ABCDABCDABABABBABACCCCDDACADBBBB"


    sir = [c for c in 'SIRSIRSIRSIRSIRS']
    stream = []
    for item in sir:
        stream.append(SIRS[item])

    for i in range(len(stream)):
        if i+wp+delta+ws >= len(stream):
            break
        p = stream[i:i+wp]
        s = stream[i+wp+delta:i+wp+delta+ws]
        tree.add_token_to_node(sorted(p+s))
    print_tree(tree.root)
    print(tree.search(['A','*','C']))
    # print(tree.get_all_leafs())
    # print(tree.get_all_children())

    for child in tree.get_all_children():
        if child == tree.root or child.parent == tree.root:
            continue
        n1 = child.count
        n2 = child.parent.count - n1
        n3 = tree.search(['*']*(len(child.final_token)-1)+[child.token])
        n4 = (len(child.final_token)-1)

print("hello")


if __name__ == '__main__':
    print("Running Program")
    test_msdd()