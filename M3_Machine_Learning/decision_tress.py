import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.datasets import load_iris


# load data
df_iris = load_iris()
# df_iris.keys()

# create dataframe
df = pd.DataFrame(
    df_iris.data, columns=["sepal length", "sepal width", "petal length", "petal width"]
)
print(df.head())

y = df_iris.target
X = df

# Decision Trees - Classifier on Iris Data
# train test split - then we fit on x train - predict on x test and compare y test to y predict
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
clf = tree.DecisionTreeClassifier(random_state=1)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
# sum((y_test-pred) == 0) / len(pred)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, pred))


# #### Zoo data ####
# tree.plot_tree(clf)

# #load data
# df_zoo = pd.read_csv('/work/zoo.csv')

# # define X and y
# y = df_zoo.class_type
# X = df_zoo.drop(['class_type','animal_name'], axis=1)

# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# # Decision Tree Classifier
# clf = tree.DecisionTreeClassifier(random_state=1)

# # fiti model
# clf = clf.fit(X_train, y_train)

# # predictions
# pred = clf.predict(X_test)

# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, pred))

