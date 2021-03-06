from itertools import product
from pprint import pprint
from typing import Tuple, List, Dict, Set
from collections import defaultdict
from utils.pptree import print_tree


class Node:
    def __init__(self, token: Tuple, depth: int, count: int, parent=None, cost: float = 0, is_leaf: bool = False):
        self.children: Dict[Tuple, Node] = {}
        self.token: Tuple = token
        self.cost: float = cost
        self.depth: int = depth
        self.count: int = count
        self.parent: Node = parent

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def increase_count(self):
        self.count += 1

    def __repr__(self):
        return str({'Token': self.token})
        # return self.__str__()

    def __str__(self):
        return ('(T:' + str(self.token)+")")
        # return ('(T:' + str(self.token) + ",C:" + str(self.count) + ",D:" + str(self.depth) + ")")


class TrieStructure:
    unq_token_dict: Dict[Tuple, Node] = {}
    token_tree_pointer:Dict[Tuple,List[Node]] = {}
    WILDCARD = '*'

    @staticmethod
    def get_row_combination(data: List[str]) -> List[List[str]]:
        if len(data) == 1:
            return [[data[0]], [TrieStructure.WILDCARD]]
        curr_list = TrieStructure.get_row_combination(data[1:])
        result_list = []
        for index in range(len(curr_list)):
            a = [[data[0]] + curr_list[index]]
            b = [[TrieStructure.WILDCARD] + curr_list[index]]
            result_list.extend(a)
            result_list.extend(b)
        # result_tuple = tuple(tuple(x) for x in result_list)
        return result_list

    @staticmethod
    def convert_tuple(data: List[List[List[str]]]) -> Tuple[Tuple[Tuple[str]]]:
        return tuple(tuple(tuple(x) for x in inner_list) for inner_list in data)

    @staticmethod
    def get_all_combination(data: List[List[str]]) -> List[List[List[str]]]:
        keeper = []
        for row in data:
            res = TrieStructure.get_row_combination(row)
            keeper.append(res)
        if len(data) > 1:
            output = []
            for item in product(*keeper):
                output.append((item))
            return TrieStructure.convert_tuple(output)
        else:
            return TrieStructure.convert_tuple(keeper)

    @staticmethod
    def remove_pruneable(self):
        pass

    def systematic_expand(self, tokens):
        tokens = TrieStructure.get_all_combination(tokens)
        return tokens

    def add_token_to_node(self, node_tokens: Tuple, current_node: Node):
        for token in node_tokens:
            depth = current_node.depth
            if token not in current_node.children:
                newNode = Node(token, depth=(depth + 1), count=1, parent=current_node)
                current_node.children[token] = newNode
                if token not in TrieStructure.token_tree_pointer:
                    TrieStructure.token_tree_pointer[token] = []
                TrieStructure.token_tree_pointer[token].append(newNode)
                if token not in TrieStructure.unq_token_dict:
                    TrieStructure.unq_token_dict[token] = newNode
            else:
                current_node.children[token].increase_count()
            current_node = current_node.children[token]

    def add_chain(self, token: Tuple):
        assert (len(token) > 0)
        # if token[0] in TrieStructure.unq_token_dict:
        if False:
            current_token_node = TrieStructure.unq_token_dict[token[0]]
            self.add_token_to_node(token, current_token_node.parent)
        else:
            self.add_token_to_node(token, self.root)

    def print_depth_first(self, starting_token: Node = None):
        if starting_token is None:
            return
        cur: Node = starting_token
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

    def __init__(self, root_token, precursor_width, successor_width, delay):
        self.delay = delay
        self.successor_width = successor_width
        self.root = Node(root_token, depth=0, count=1)
        self.precursor_width = precursor_width


def get_prec_succ(token_stream, p_width=1, lag=0, s_width=1):
    result_stream = []
    for i in range(len(token_stream)):
        if i + p_width + lag + s_width > len(token_stream):
            break
        prec = token_stream[i:(i + p_width)]
        succ = token_stream[(i + p_width + lag):(i + p_width + lag + s_width)]
        prec.extend(succ)
        result_stream.append(prec)
    return result_stream


def test_get_prec_succ():
    test_data = [['D', 'X'], ['A', 'Y'], ['B', 'Z'], ['B', 'X'], ['A', 'Z'], ['B', 'Y']];
    sequence_data = get_prec_succ(test_data)
    pprint(sequence_data)
    for d in sequence_data:
        print(d)


def test_trie_simple_add():
    myTrie = TrieStructure('*', precursor_width=1, successor_width=1, delay=0)
    test_data = ((('D', 'X'), ('A', 'Y'), ('B', 'Z')), (('D', 'X'), ('A', 'Y'), ('*', 'Z')),
                 (('D', 'X'), ('A', 'Y'), ('B', '*')), (('D', 'X'), ('A', 'Y'), ('*', '*')),
                 (('D', 'X'), ('*', 'Y'), ('B', 'Z')))
    for d in test_data:
        myTrie.add(d)
    myTrie.print_depth_first(myTrie.root)
    pprint(myTrie.unq_token_dict)


def test_trie():
    # test_trie_simple_add()
    # test_data = ((('D', 'X'), ('A', 'Y'), ('B', 'Z')), (('D', 'X'), ('A', 'Y'), ('*', 'Z')),
    #              (('D', 'X'), ('A', 'Y'), ('B', '*')), (('D', 'X'), ('A', 'Y'), ('*', '*')),
    #              (('D', 'X'), ('*', 'Y'), ('B', 'Z')))
    myTrie = TrieStructure('*', precursor_width=1, successor_width=1, delay=0)
    # pprint(test_data)
    test_data = [['D', 'X'], ['A', 'Y'], ['B', 'Z'], ['B', 'X'], ['A', 'Z'], ['B', 'Y']]
    test_data = [['D', 'X'], ['A', 'Y'], ['D', 'X'], ['A', 'Y'], ['B', 'X'], ['C', 'X'], ['D', 'X'], ['A', 'Y']]
    # test_data = [['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X'],['D', 'X']]
    token_chains = get_prec_succ(test_data)
    all_token_chains = []
    for chain in token_chains:
        all_token_chains.append(TrieStructure.get_all_combination(chain))

    myTrie = TrieStructure('*', precursor_width=1, successor_width=1, delay=0)
    for chain in all_token_chains:
        for token in chain:
            myTrie.add_chain(token)
    print_tree(myTrie.root, childattr='children')
    pprint(TrieStructure.token_tree_pointer)
    # pprint(myTrie.unq_token_dict)
    # myTrie.print_depth_first(myTrie.root)
    print("This is the end")


if __name__ == '__main__':
    print("testing trie")
    test_trie()

    # def search(self, word, node=None):
    #     cur = node
    #     if not cur:
    #         cur = self.root
    #     for i, character in enumerate(word):
    #         if character == "*":
    #             if i == len(word) - 1:
    #                 for child in cur.children.values():
    #                     if child.is_terminal:
    #                         return True
    #                 return False
    #             for child in cur.children.values():
    #                 if self.search(word[i+1:], child) == True:
    #                     return True
    #             return False
    #         if character not in cur.children:
    #             return False
    #         cur = cur.children[character]
    #     return cur.is_leaf

'''
        check if a given node exist in the tree.
        return its position and enter the chains as it childs
        :param current_node: Parent node of the head of the token chain
        :param token_chain: token elements to inserted
        :return:
        '''

# def _add(self, node_tokens, current_node=None):
#     # current_node = None
#     if current_node is None:
#         current_node: Node = self.root
#     for token in node_tokens:
#         depth = current_node.depth
#         if token not in current_node.children:
#             newNode = Node(token, depth=depth + 1, count=1, parent=current_node)
#             current_node.children[token] = newNode
#             if token not in TrieStructure.unq_token_dict:
#                 TrieStructure.unq_token_dict[token] = newNode
#         else:
#             current_node.children[token].increase_count()
#         current_node = current_node.children[token]
#
