import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0].reshape(-1, 1); y = data[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y)
lr = LinearRegression().fit(X_train, y_train)

lr.coef_
lr.intercept_


print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

def regLine(coef, inter, xmin, xmax):
    ymin = inter + coef * xmin
    ymax = inter + coef * xmax
    return ymin, ymax

fig = plt.figure(figsize=(8, 8),dpi=100)
ax = fig.add_subplot()
ax.scatter(X, y)
xmin = min(X)
xmax = max(X)
ymin, ymax = regLine(lr.coef_, lr.intercept_, xmin, xmax)
ax.plot([xmin, xmax], [ymin, ymax])