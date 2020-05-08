a = [1, 2, 3]
b = [4, 5, 6]
c = a + b
print(c)
d = [a, b]
print(d)

a = {'a': 1, 'b':2, 'c': 3}
b = {'a': 4, 'e':5, 'f': 6}
c = {**a, **b}
print(c)

a = set([1, 2, 3])
b = set([1, 5, 6])
c = a.union(b)
print(c)
d = {item for item in list(a) + list(b)}
print(d)

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
c = np.concatenate((a, b))
print(c)
d = np.concatenate((a, b), axis=1)
print(d)

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9]])
c = np.concatenate((a, b))
print(c)

d = np.array([[7, 8]]).transpose()
e = np.concatenate((a, d), axis=1)
print(e)

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([7, 8])
c = b[:, np.newaxis]
d = np.concatenate((a, c), axis=1)
print(d)

import pandas as pd

a = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
b = pd.DataFrame([[7, 8], [9, 10]], columns=['b', 'd'])
c = pd.concat([a, b], axis=0)
c
d = pd.concat([a, b], axis=1)
d