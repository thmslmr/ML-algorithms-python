# Impletement Decision Trees


from sklearn import datasets, metrics, tree
from sklearn.model_selection import train_test_split
import graphviz

breast_cancer = datasets.load_breast_cancer()

print(breast_cancer.keys())
print(breast_cancer.feature_names)
print(breast_cancer.target_names)

X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=33)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.score(X_test, y_test))
print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))

graph_data = tree.export_graphviz(model, out_file= None, feature_names=breast_cancer.feature_names, filled=True)
graph = graphviz.Source(graph_data)
graph.render("breast_cancer", view=True)
