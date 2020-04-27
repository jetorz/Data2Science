import numpy as np
from sklearn.preprocessing import normalize

x = np.array([[150, 5], [120, 4], [125, 4], [80, 3]])
mu = np.mean(x, axis=0)
sigma = np.std(x,axis=0,ddof=0)
z = (x - mu)/sigma
print(z)

normalize(x, norm='l1', axis=0)
normalize(x, norm='l2', axis=0)
