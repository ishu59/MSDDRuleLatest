from pprint import pprint

from MSDD import get_all_combination, get_row_combination

def test_get_all_precursors():
    data = [['D','X','2'],['A', 'Y', '1'],['B', 'Z', '3']]
    data = [['D', 'X'], ['A', 'Y'],['B', 'Z']]
    # data = [['D','X','2']]
    # data = [['D'], ['A'],['B']]
    res = get_all_combination(data)
    pprint(res)

def test_get_row_combination():
    data = ['D']
    data = ['D','X','2']
    data = ['D','X']
    res = get_row_combination(data)
    pprint(res)

if __name__ == '__main__':
    test_get_all_precursors()
    test_get_row_combination()