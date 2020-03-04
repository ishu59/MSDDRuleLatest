from itertools import product
from pprint import pprint
from typing import Tuple, List, Dict, Set

class ConvertStream:
    def __init__(self, stream, prec_wid, succ_wid, delay, wildcard = '*'):
        self.wildcard = wildcard
        self.delay = delay
        self.succ_wid = succ_wid
        self.prec_wid = prec_wid
        self.stream = stream
        self.modified_stream = self._get_prec_succ()
        self.res = []


    def get_res(self):
        for x in self.modified_stream:
            self.res.append(self.get_all_combination(x))
        return self.res

    def _get_row_combination(self, data: List[str]) -> List[List[str]]:
        if len(data) == 1:
            return [[data[0]], [self.wildcard]]
        curr_list = self._get_row_combination(data[1:])
        result_list = []
        for index in range(len(curr_list)):
            a = [[data[0]] + curr_list[index]]
            b = [[self.wildcard] + curr_list[index]]
            result_list.extend(a)
            result_list.extend(b)
        # result_tuple = tuple(tuple(x) for x in result_list)
        return result_list

    def _convert_tuple(self, data: List[List[List[str]]]) -> Tuple[Tuple[Tuple[str]]]:
        return tuple(tuple(tuple(x) for x in inner_list) for inner_list in data)

    def get_all_combination(self, data: List[List[str]]) -> List[List[List[str]]]:
        keeper = []
        for row in data:
            res = self._get_row_combination(row)
            keeper.append(res)
        if len(data) > 1:
            output = []
            for item in product(*keeper):
                output.append((item))
            return self._convert_tuple(output)
        else:
            return self._convert_tuple(keeper)

    def _get_prec_succ(self):
        result_stream = []
        for i in range(len(self.stream)):
            if i + self.prec_wid + self.delay + self.succ_wid > len(self.stream):
                break
            prec = self.stream[i:(i + self.prec_wid)]
            succ = self.stream[(i + self.prec_wid + self.delay):(i + self.prec_wid + self.delay + self.succ_wid)]
            prec.extend(succ)
            result_stream.append(prec)
        return result_stream

def test_convert_stream():
    data = [['D', 'X'], ['A', 'Y'], ['B', 'Z'], ['B', 'X'], ['A', 'Z'], ['B', 'Y']]
    s = ConvertStream(data,1,1,0)
    pprint(s.get_res())

if __name__ == '__main__':
    test_convert_stream()