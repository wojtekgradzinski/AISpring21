import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression



np.random.seed(12)
num_observations = 5000


################


x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))


################

# plot
plt.figure(figsize=(12, 8))
plt.scatter(
    simulated_separableish_features[:, 0],
    simulated_separableish_features[:, 1],
    c=simulated_labels,
    alpha=0.4,
)
# plt.show()


################

# sigmoid function
def sigmoid(scores):
    # the sigmoid function is 1 / (1 + e^x)
    return 1.0 / (1 + np.exp(-scores))

# Calculating the Log-Likelihood (sum over all the training data)
def log_likelihood(features, target, weights):
    #model output
    scores = np.dot(features,weights)
    nll = np.sum(target*scores - np.log(1 + np.exp(scores)))
    return nll


################

# Building the Logistic Regression Function
def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        # scores -> feature * weights
        # preds -> sigmoid(scores)
        # error = target-preds
        # gradient -> transposed feature * error
        # weight -> weights + learning_rate * gradient
        
        # YOUR CODE HERE
        scores = np.dot(features, weights) 
        preds = sigmoid(scores)
        error = target - preds
        gradient = np.dot(features.T,error)
        weights += learning_rate * gradient
        #raise NotImplementedError()
        # Print log-likelihood every so often
        if step % 10000 == 0:
            print(log_likelihood(features, target, weights))
        
    return weights



################

# time to do the regression
weights = logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)



################

# Comparing to Sk-Learn's LogisticRegression


# implement sklearn logistic regresion and fit
# fit_intercept=True
# C = 1e15
clf = 0
X = simulated_separableish_features #look at the beginning (pre-definied)
y = simulated_labels
clf = LogisticRegression( random_state = 0, C = 1e15).fit(X, y)
# print(X)
# print(y)
# print(clf)



################

# accuracy

#final scores =  np.ones(data.shape[0]) -> stack with data -> stacked_data * w
#preds = round(sigmoid(inal scores ))
# accuracy is the percentages of correct guesses 0-1
final_scores = np.dot(np.hstack((np.ones((X.shape[0], 1)),X)), weights)
# print(X)
preds = np.round(sigmoid(final_scores ))                               
# accuracy = 0

accuracy = clf.score(simulated_separableish_features, simulated_labels)
# YOUR CODE HERE here 
final_scores = np.dot(np.hstack((np.ones((X.shape[0], 1)),X)), weights)
preds = np.round(sigmoid(final_scores )) 

print('Accuracy from scratch: {0}'.format((preds == simulated_labels).sum().astype(float) / len(preds)))
print('Accuracy from sk-learn: {0}'.format(clf.score(simulated_separableish_features, simulated_labels)))


################

plt.figure(figsize = (12, 8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = preds == simulated_labels - 1, alpha = .8, s = 50)