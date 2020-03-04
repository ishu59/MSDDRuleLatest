from pprint import pprint
from typing import List, Dict, Tuple
import heapq
import numpy as np
from itertools import product
WILDCARD = '*'

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





def test_msdd():
    data = [['D','X','2'],['B','Y','1'],
            ['B','Y','3'],['A','Z','2'],
            ['D','Y','2'],['C','X','1'],
            ['D','Z','2'],['A','Z','3'],
            ['B','X','2'],['C','Y','1']]
    myData = np.array(data)
    print(myData)


if __name__ == '__main__':
    print("Running Program")