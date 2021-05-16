# import the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
Linear regression is the most basic type of regression commonly used for
predictive analysis. The idea is pretty simple: we have a dataset and we have
features associated with it. Features should be chosen very cautiously
as they determine how much our model will be able to make future predictions.
We try to set the weight of these features, over many iterations, so that they best
fit our dataset. In this particular code, I had used a CSGO dataset (ADR vs
Rating). We try to best fit a line through dataset and estimate the parameters.
"""
#################

# load and inspect the data
df = pd.read_csv("reg_data.csv")

#################

# plot the data (scatter)
plt.figure(figsize=(12, 5))
plt.title("Scatter Plot")
plt.scatter(df.X, df.Y, c="Blue")

#################

# make the train test split tested with seed 0 and a test_size = 0.2
# X = df['X'].values
# Y = df['Y'].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#################

# implement a least squares function to find a, b
x_mean = 0
y_mean = 0
b_upper = 0
b_lower = 0
b = 0
a = 0
X1 = df.X
Y1 = df.Y
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)


for i in range(len(X1)):
    b_upper += (X1[i] - x_mean) * (Y1[i] - y_mean)
    b_lower += (X1[i] - x_mean) ** 2

b = b_upper / b_lower
a = y_mean - b * x_mean


print(b)
print(a)


#################

# Lets now plot a line
# rang = np.arange(0, 100)
# y = a + b * rang
# plt.figure(figsize=(12,6))
# plt.plot(y)
# plt.show()

line = 0
x = np.array([])
### BEGIN SOLUTION
x = np.arange(0, 100)
line = a + b * x
plt.plot(line)

#################

# plt.figure(figsize=(12,6))
# plt.title('Linear Regression')
# plt.scatter(X, Y, c = Y, alpha = .6)
# plt.plot(X, y_pred)
# plt.show()

plt.plot(line)
plt.scatter(x_train, y_train)


#################

# Classify your test data in to classes
# if the Y value for a certain X is lower than the line then the class is 0
# class_0 = [y_test[y_test > (a + b*x_test)]]
# class_1 = [y_test[y_test < (a + b*x_test)]]
class_0 = []
class_1 = []

### BEGIN SOLUTION
for i in range(x_test.shape[0]):

    # We check the line value vs the real Y value
    if y_test[i] < (a + x_test[i] * b):
        class_0.append((x_test[i], y_test[i]))
    else:
        class_1.append((x_test[i], y_test[i]))


class_0 = np.array(class_0)
class_1 = np.array(class_1)
print(class_0.shape)
print(class_1.shape)


#################

# we can plot the line with each class so we can clearly see the split
plt.scatter(class_0[:, 0], class_0[:, 1])
plt.scatter(class_1[:, 0], class_1[:, 1])
plt.plot(y)

#################

# get the total error for the classes
# MSE (predictor)
err = 0
for i in range(x_test.shape[0]):
    err += abs(y_test[i] - (a + x_test[i] * b))

print(err)


#################

