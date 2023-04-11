import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [4,5,6,7,8]])

print(np.sum(arr[:, (1,2)], axis=1) - 3)

print(type(arr))