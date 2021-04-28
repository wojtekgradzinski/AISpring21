import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt


############

iris_data = load_iris()
iris_data

############

# Create pd DF the same way you did the first day
# Create the class and target columns and remove de cm from the columns names


def create_df():
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df["target"] = iris_data.target
    df["class"] = iris_data.target_names[iris_data.target]
    df.columns = [col.replace("(cm)", "").strip() for col in df.columns]
    return df


df_iris = create_df()
df_iris.sample(n=10)

############

# From our data we will only get variable columns as np.array
x = 0
x = df_iris[["sepal length", "sepal width", "petal length", "petal width"]].to_numpy()
print(x.shape)

# or

# x_iloc = df_iris.iloc[:,:4].to_numpy()
# print(x_iloc)

############

# merge
mergings = 0
mergings = linkage(x, "complete")

############

# plot
dn = 0
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(mergings)
plt.show()

############

# KMEANS

############

from sklearn.cluster import KMeans

df = pd.read_csv("./datasets/ch1ex1.csv")
points = df.values

new_df = pd.read_csv("./datasets/ch1ex2.csv")
new_points = new_df.values


############

# model = 0
model = KMeans(n_clusters=3)

############

model.fit(points)

############

labels = model.predict(points)
labels.shape

############

# Make a function that returns 3 numpy arrays each one with the points associated for each class
# If the label is 0 they go into data_0
# If the label is 1 they go into data_1
# If the label is 2 they go into data_2


def separate_labels(labels, points):
    data_0 = []
    data_1 = []
    data_2 = []
    for i in range(labels.shape[0]):
        if labels[i] == 0:
            data_0.append(points[i])
        elif labels[i] == 1:
            data_1.append(points[i])
        else:
            data_2.append(points[i])
    # YOUR CODE HERE
    return np.array(data_0), np.array(data_1), np.array(data_2)


data_0, data_1, data_2 = separate_labels(labels, points)

print(data_0.shape)
print(data_1.shape)
print(data_2.shape)


##################

# plotting the data
plt.scatter(data_0[:, 0], data_0[:, 1], c="steelblue")
plt.scatter(data_1[:, 0], data_1[:, 1], c="orange")
plt.scatter(data_2[:, 0], data_2[:, 1], c="firebrick")
plt.title("K-MEANS")


##################

new_labels = model.predict(new_points)
new_labels

#################

# using enumerate


def separate_labels(labels, points):
    data_0 = []
    data_1 = []
    data_2 = []

    for index, i in enumerate(labels):
        if i == 0:
            data_0.append(points[index])
        elif i == 1:
            data_1.append(points[index])
        elif i == 2:
            data_2.append(points[index])

    return np.array(data_0), np.array(data_1), np.array(data_2)


data_0, data_1, data_2 = separate_labels(labels, points)

print(data_0.shape)
print(data_1.shape)
print(data_2.shape)


################

# separate the data by labels like we did before

new_0 = []
new_1 = []
new_2 = []

new_0, new_1, new_2 = separate_labels(new_labels, new_points)
abs(new_0.shape[0])
abs(new_1.shape[0])
abs(new_2.shape[0])


###################


# plot the new labels

plt.scatter(new_0[:, 0], new_0[:, 1], c="aqua")
plt.scatter(new_1[:, 0], new_1[:, 1], c="forestgreen")
plt.scatter(new_2[:, 0], new_2[:, 1], c="gold")
plt.title("K-MEANS - new")


################

# plot all the points together

plt.scatter(data_0[:, 0], data_0[:, 1], c="steelblue")
plt.scatter(data_1[:, 0], data_1[:, 1], c="orange")
plt.scatter(data_2[:, 0], data_2[:, 1], c="firebrick")

plt.scatter(new_0[:, 0], new_0[:, 1], c="aqua")
plt.scatter(new_1[:, 0], new_1[:, 1], c="forestgreen")
plt.scatter(new_2[:, 0], new_2[:, 1], c="gold")
plt.title("K-MEANS - ALL")


################


################


################


################

