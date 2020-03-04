from itertools import combinations, permutations, product


c = ['A','B','C']
# for i in range(1,len(c)+1):
#     print(list(combinations(c,i)))

d = ['1','2']
e = ['X','Y']
z = [c,d,e]

final = []
for i in range(len(z)):

    for j in range(i+1,len(z)):
        temp = z[i:]
        final.extend(list(product(*temp)))
        final.append(z[i])
print(final)










# res = list(product(c,d, e))
# print(len(res))
# print(res)
#
# res = list(product(c,d))
# print(len(res))
# print(res)
#
# res = list(product(c,e))
# print(len(res))
# print(res)
#
# res = list(product(d,e))
# print(len(res))
# print(res)
#
#
# res = list(product(*z))
# print(len(res))
# print(res)
