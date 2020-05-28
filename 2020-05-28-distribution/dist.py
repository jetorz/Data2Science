import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt', delimiter=',', encoding='utf8')

dists = {'norm': stats.norm, 'lognorm': stats.lognorm, 'expon': stats.expon}
for d in dists:
    paras = dists[d].fit(data)
    test = stats.kstest(data, dists[d].cdf, paras)
    print('{:1}\tpvalue:{:2}'.format(d, test[-1]))

x = np.arange(0, 20, 0.01)

paras = stats.norm.fit(data)
ynorm = stats.norm.pdf(x, paras[0], paras[1])

paras = stats.lognorm.fit(data)
ylognorm = stats.lognorm.pdf(x, paras[0], paras[1], paras[2])

paras = stats.expon.fit(data)
yexpon = stats.expon.pdf(x, paras[0], paras[1])

fig, ax = plt.subplots()

ax.plot(x, ynorm, label='norm')
ax.plot(x, ylognorm, label='lognorm')
ax.plot(x, yexpon, label='expon')

ax.legend()
