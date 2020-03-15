from pprint import pprint
from typing import Set, Dict, List, Tuple, Union


class MSDDToken:
    '''
    Token is hashable
    '''

    def __init__(self, value, stream_index, time_offset):
        self.time_offset = time_offset
        self.stream_index = stream_index
        self.value = value

    def __str__(self):
        # return str(self.value)
        # return str(dict(v=self.value))
        return str(dict(v=self.value, s=self.stream_index, t=self.time_offset))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not other:
            return False
        return self.value == other.value and self.stream_index == other.stream_index and self.time_offset == other.time_offset

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __ge__(self, other):
        if self.stream_index != other.stream_index:
            # raise Exception("Cannot compare between different streams")
            return True

        if other.value >= self.value:
            return False

    def __lt__(self, other):
        if self is None:
            return True
        if other is None:
            return False
        if self.stream_index != other.stream_index:
            # raise Exception("Cannot compare between different streams")
            return True

        if other.value < self.value:
            return False
        # if other.stream_index < self.stream_index:
        #     return False
        # if other.time_offset < self.time_offset:
        #     return False
        return True


if __name__ == '__main__':
    print("")
    t1 = MSDDToken('A', 0 , 0)
    t2 = MSDDToken('B', 0 , 0)
    t3 = MSDDToken('', 0, 0)
    print(t1 > t2)
    print(t1 < t2)

    # print(None < t1)
    a = [t1,t2, t3]
    print(sorted(a))




# class MSDDNodeElement:
#     def __init__(self, token, depth=0, cost=0, count=1, parent=None):
#         self.name = str(token)
#         if isinstance(token, (list, set)):
#             self.token = token
#         self.children = {}
#         self.cost: float = cost
#         self.depth: int = depth
#         self.count: int = count
#
#     def is_leaf(self) -> bool:
#         return len(self.children) == 0
#
#     def increase_count(self):
#         self.count += 1
#
#     def __str__(self):
#         return "(T:" + str(self.name) + ")"
#
#     def __repr__(self):
#         return self.__str__()

# class MSDDNode:
#
#     def __init__(self, token_chain: Union[str, List[List]], p_width: int = None, s_width: int = None, delay: int = None,
#                  depth: int = 0, count: int = 0,
#                  cost: float = 0, parent=None):
#         self.name = token_chain
#         if isinstance(token_chain, str):
#             self.pre_token: Set[MSDDToken] = set()
#             self.suc_token: Set[MSDDToken] = set()
#         else:
#             assert (isinstance(token_chain, (list, tuple))), "Token should be a list of List"
#             for e in token_chain:
#                 assert (isinstance(e, (list, tuple))), "Token should be a list of List"
#             assert (len(token_chain) >= (p_width + s_width + delay)), "Length of token is less than required"
#             self.pre_token: Set[MSDDToken] = self.get_multitoken(token_chain[:p_width], start_time=0)
#             self.suc_token: Set[MSDDToken] = self.get_multitoken(token_chain[p_width + delay:p_width + delay + s_width],
#                                                                  start_time=(p_width + delay))
#         self.children: Dict = {}
#         self.cost: float = cost
#         self.depth: int = depth
#         self.count: int = count
#         self.parent: MSDDNode = parent
#
#     def is_leaf(self) -> bool:
#         return len(self.children) == 0
#
#     def increase_count(self):
#         self.count += 1
#
#     def get_multitoken(self, token_chain, start_time) -> Set[MSDDToken]:
#         token_set: Set[MSDDToken] = set()
#         for time_stream_index, single_time_stream in enumerate(token_chain):
#             for elem_index, elem in enumerate(single_time_stream):
#                 # tok = MSDDToken(value=elem, stream_index=elem_index, time_offset=(start_time + time_stream_index))
#                 if elem != '*':
#                     token_set.add(
#                         MSDDToken(value=elem, stream_index=elem_index, time_offset=(start_time + time_stream_index)))
#                     # token_set.append(
#                     #     MSDDToken(value=elem, stream_index=elem_index, time_offset=(start_time + time_stream_index)))
#         return token_set
#
#     def __str__(self):
#         return "(P:" + str(self.pre_token) + ", S:" + str(self.suc_token) + ")"
#
#     def __repr__(self):
#         return self.__str__()
#
#
# def test_msdd_node():
#     print("test_msdd_node")
#     # test_stream = [['D', '*'], ['*', 'Y'], ['B', 'Z'], ['B', 'X'], ['A', 'Z'], ['B', 'Y']]
#     # node = MSDDNode(test_stream, p_width=2, s_width=2, delay=1)
#     # pprint(node)
#     # node = MSDDNode(test_stream, p_width=1, s_width=1, delay=0)
#     # pprint(node)
#     # d = []
#     # t = MSDDToken(1, 2, 3)
#     # d.append(t)
#     #
#     # dupt = MSDDToken(1, 2, 3)
#     # if dupt in d:
#     #     print("Found")
#     #
#     # d = {}
#     # # d[node] =2
#     # d[t] = 1
#     # if dupt in d:
#     #     print("Found in dict")
#     #
#     # n = MSDDNode("Hello")
#     # pprint(n)
#     #
#
#
# if __name__ == '__main__':
#     pprint("testing MSDD node.....")
#     test_msdd_node()
