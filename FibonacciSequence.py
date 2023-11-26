res = []
for i in range (0, 20):
   
    if i <= 1:
        res.append(i)

    if i > 1:
        res.append(res[i-1] + res[i-2])

for i in res:
    print(i, end=" ")
