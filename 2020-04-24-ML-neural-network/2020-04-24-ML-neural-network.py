import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data = np.loadtxt('assets\ex3data.txt', delimiter=' ')
X = data[:, 0:400]; y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y)
mlp = MLPClassifier(solver='lbfgs', random_state=1).fit(X_train, y_train)

print("Training set score: {:.2f}".format(mlp.score(X_train, y_train)))
print("Test set score: {:.2f}".format(mlp.score(X_test, y_test)))