from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

forest = RandomForestClassifier(n_estimators=100,max_features='auto')
forest = forest.fit(X_train, y_train)

print("Training set score: {:.2f}".format(forest.score(X_train, y_train)))
print("Test set score: {:.2f}".format(forest.score(X_test, y_test)))

forest.feature_importances_

name = ['mean - radius', 'mean - texture', 'mean - perimeter', 'mean - area', 'mean - smoothness', 'mean - compactness', 'mean - concavity', 'mean - concave points', 'mean - symmetry', 'mean - fractal dimension', 'SE - radius', 'SE - texture', 'SE - perimeter', 'SE - area', 'SE - smoothness', 'SE - compactness', 'SE - concavity', 'SE - concave points', 'SE - symmetry', 'SE - fractal dimension', 'worse - radius', 'worse - texture', 'worse - perimeter', 'worse - area', 'worse - smoothness', 'worse - compactness', 'worse - concavity', 'worse - concave points', 'worse - symmetry', 'worse - fractal dimension']

plt.bar(name, forest.feature_importances_)
plt.tight_layout()
plt.show()