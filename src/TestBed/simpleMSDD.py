import os, sys
from itertools import combinations, product

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 'rules' module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from typing import List, Dict, Union, Tuple
from collections import defaultdict, Set
import numpy as np
from utils.CombinationTree import compute_comb
from Trie.MSDDNode import MSDDToken

stream = "ABCABDABAABDBDABCDAAADC"
best = []
k = 2
wp = 2
ws = 2
delta = 0

open_list = [[[], []]]


def validate_token(p, s):
    for pc in p:
        for sc in s:
            if sc < pc:
                return False
    return True

def create_msdd_token_1d(stream_list, start_time):
    msdd_list = []
    for i, s in enumerate(stream_list):
        msdd_list.append(MSDDToken( value=s, stream_index = 0, time_offset = start_time+i))
    return msdd_list

# print(create_msdd_token_1d(["A","B","C"], start_time=1))

def systematic_expand(cur_node, wp, ws, delta, tp, ts):
    # prec, succ = cur_node
    children = []
    prec_list = []
    succ_list = []
    for i, tok in enumerate(tp):
        prec_list.append(MSDDToken(value=tok, time_offset=i, stream_index=0))
    for i, tok in enumerate(ts):
        succ_list.append(MSDDToken(value=tok, time_offset=i+wp+delta, stream_index=0))
    prec = compute_comb(prec_list)
    succ = compute_comb(succ_list)

    for comb in product(prec, succ):
        p = comb[0]
        s = comb[-1]
        p_list = []
        s_list = []
        if (validate_token(p, s)):
            for pc in p:
                p_list.append(pc)
            for sc in s:
                s_list.append(sc)
            p_list = sorted(p_list)
            s_list = sorted(s_list)
            final_comb = [p_list, s_list]
            if final_comb not in children:
                children.append(final_comb)
    return children


def remove_prunable(children):
    return children

def compute_g_score(n1_x_y, n2_x_not_y, n3_not_x_y, n4_not_x_not_y):
    total = n1_x_y + n2_x_not_y + n3_not_x_y + n4_not_x_not_y
    n1_x_y = 1 if n1_x_y == 0 else n1_x_y
    n2_x_not_y = 1 if n2_x_not_y == 0 else n2_x_not_y
    n3_not_x_y = 1 if n3_not_x_y == 0 else n3_not_x_y
    n4_not_x_not_y = 1 if  n4_not_x_not_y == 0 else  n4_not_x_not_y
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

def f(node, stream, delta, wp, ws):
    search_p, search_s = node
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    state = [0,0]
    for i in range(len(stream)):
        if i + wp + delta + ws > len(stream):
            break
        prec, succ = stream[i:i + wp], stream[i + wp + delta:i + wp + delta + ws]
        prec = create_msdd_token_1d(prec,0)
        succ = create_msdd_token_1d(succ,i+wp+delta)
        if set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
            n1 += 1
        elif set(search_p).issubset(set(prec)) and not set(search_s).issubset(set(succ)):
            n2 +=1
        elif  not set(search_p).issubset(set(prec)) and set(search_s).issubset(set(succ)):
            n3 +=1
        else:
            n4+=1
    return compute_g_score(n1,n2,n3,n4)




for i in range(len(stream)):
    if i + wp + delta + ws > len(stream):
        break
    while len(open_list) > 0:
        cur_node = open_list.pop(0)
        children = systematic_expand(cur_node, wp, ws, delta, stream[i:i + wp], stream[i + wp + delta:i + wp + delta + ws])
        # print(children)
        children = remove_prunable(children)
        for child in children:
            open_list.append(child)
        for child in children:
            n = [{}, {}]
            score = f(child, stream, delta, wp, ws)
            if len(best) < k or f(child, stream, delta, wp, ws) > f(n, stream, delta, wp, ws):
                best.append(child)
            elif len(best) > k:
                best = sorted(best, reverse=True)
                if best[-1] < child:
                    best.pop()
                    best.append(child)

print(best)

for i in range(len(stream)):
    if i + wp + delta + ws >= len(stream):
        break
    p = stream[i:i + wp]
    s = stream[i + wp + delta:i + wp + delta + ws]
