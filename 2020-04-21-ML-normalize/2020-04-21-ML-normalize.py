import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

x = np.array([[150, 5], [120, 4], [125, 4], [80, 3]])
mu = np.mean(x, axis=0)
sigma = np.std(x,axis=0,ddof=0)
z = (x - mu)/sigma
z
lone = normalize(x, norm='l1', axis=0)
ltwo = normalize(x, norm='l2', axis=0)

plt.style.use('ggplot')
plt.bar(x=lone.ravel(),height=lone.ravel(),width=0.001)
plt.bar(x=ltwo.ravel(),height=ltwo.ravel(),width=0.002)
