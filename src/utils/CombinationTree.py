from itertools import combinations


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