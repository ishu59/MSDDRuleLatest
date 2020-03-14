a = [[x for x in "ABCD"], [x for x in 'EFG']]
b = [[x for x in 'WXY'], [x for x in 'PP']]
l = []
l.append(zip(a,b))
print(l)
print(l[0])
print(*list(zip(a,b)))