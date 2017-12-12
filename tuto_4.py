# Use K-nearest Neighbors to find similar datapoints

from sklearn import datasets, metrics, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.scatter(X_test[:,0], X_test[:,1], c='m')
plt.show()

model = neighbors.KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.score(X_test, y_test))
print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))
