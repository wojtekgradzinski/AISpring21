# import libraries
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


# make the dataset with make_blobs
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
fig = plt.fig
plt.scatter(X[:, 0], X[:, 1], s=35)
# plt.show()


# create KNN function

# define train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

x_new1 = X_train[0]
x_new2 = X_test[0]
y_new1 = y_train[0]
y_new2 = y_test[0]
# X_train
# X_test
# y_train
# y_test

v1 = x_new1
v2 = x_new2


def get_eucledian_distance(v1, v2):
    distance = np.linalg.norm(v1, v2)
    return distance


# dist = numpy.linalg.norm(a-b)

