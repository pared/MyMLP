import numpy as np

W = np.ones((3, 2))
i = np.ones((2, 1))
i[1][0] = 2
print(W)
print(i)

print(np.matmul(W, i))

W = np.random.rand(3, 2)
print(W)