import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))