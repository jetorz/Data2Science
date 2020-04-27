import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = np.loadtxt('assets\ex2data1.txt', delimiter=',')
X = data[:, 0:2]; y = data[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y)
lr = LogisticRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

data = np.loadtxt('assets\ex2data2.txt', delimiter=',')
X = data[:, 0:8]; y = data[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y)
lr = LogisticRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))