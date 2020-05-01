from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

tree = DecisionTreeClassifier()
tree = tree.fit(X_train, y_train)

print("Training set score: {:.2f}".format(tree.score(X_train, y_train)))
print("Test set score: {:.2f}".format(tree.score(X_test, y_test)))

sklearn.tree.export_text(tree)

tree = DecisionTreeClassifier(max_depth=3)
tree = tree.fit(X_train, y_train)

print("Training set score: {:.2f}".format(tree.score(X_train, y_train)))
print("Test set score: {:.2f}".format(tree.score(X_test, y_test)))

sklearn.tree.export_text(tree)
