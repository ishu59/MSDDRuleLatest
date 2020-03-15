from itertools import combinations, product


def compute_comb(ordering_list):
    res = []
    for i in range(len(ordering_list)):
        for j in range(i + 1, len(ordering_list) + 1):
            if i - j == 0:
                continue
            for item in list(combinations(ordering_list[i:], j)):
                if len(item) > 0:
                    res.append(item)
    return res


def compute_comb_adv(ordering_list):
    res = []
    res.append((' ',))
    for i in range(len(ordering_list)):
        for j in range(i + 1, len(ordering_list) + 1):
            if i - j == 0:
                continue
            for item in list(combinations(ordering_list[i:], j)):
                if len(item) > 0:
                    res.append(item)
    return res


# def compute_comb(ordering_list):
#     res = []
#     for i in range(len(ordering_list)):
#         for j in range(i + 1, len(ordering_list) + 1):
#             if i - j == 0:
#                 continue
#             for item in list(combinations(ordering_list[i:], j)):
#                 if len(item) > 0:
#                     res.append(item)
#     return res

if __name__ == '__main__':
    print("")
    print(compute_comb(["A", 'B', 'C']))
    print(compute_comb_adv(["A", 'B', 'C']))
    a = compute_comb_adv(["A", 'B', 'C'])
    b = compute_comb_adv(["1", '2', '3'])
    print(list(product(a, b)))
