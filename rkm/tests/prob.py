

import numpy as np

cnt = 0
N = 1000
d = 6   # number of dimension
for i in range(N):
    vs = np.random.randint(0, d, 5)
    if len(set(vs)) != len(vs):
        cnt += 1
print(cnt, cnt/N)

p = 1
for i in range(1, 5):
    p *= (d-i)/d
print(1- p)

