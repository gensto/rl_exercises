import numpy as np

Observation = [10,10,10,10]
q_table = np.random.uniform(low=0, high=1, size=(Observation + [2]))

discreteState = (7,8,5,3)
print(q_table[discreteState + (1,)])