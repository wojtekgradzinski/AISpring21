# Import needed packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

# Set random seed
np.random.seed(0)


# Load the Iris dataset included with scikit-learn
iris = load_iris()

# pandas dataframe
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df_iris.head())

df_iris["target"] = iris.target
df_iris["class"] = iris.target_names[iris.target]
df_iris.columns = [col.replace("(cm)", "").strip() for col in df_iris.columns]


# start PCA

X = df_iris.iloc[:, 0:4]
scaler = StandardScaler()

pca = PCA()

# print(pd.DataFrame(pca.fit_transform(X)))

# Put data in a pandas DataFrame
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target and class to DataFrame
df_iris["target"] = iris.target

# Show 10 random samples
# print(df_iris.sample(n=10))


# A graph to visualize
sns.set(style="ticks")
sns.pairplot(data=df_iris.loc[:, :"target"], hue="target")
# plt.show()

# Run the PCA model
pca = PCA(n_components=2)
df_pca = df_iris.drop(["target"], axis=1)


# project = pd.Dataframe(project)
# print(df_pca.shape)
# print(project.shape)


# Transformed dataframe
project = pca.fit_transform(df_pca)

# check the variance in each component
plt.plot(np.cumsum(pca.explained_variance_ratio_))


# from sklearn.decomposition import PCA

# pca = PCA(…)

# Arguments in PCA:
# n_components = number of components

# svd_solver =  # 'randomized'… (eigenvalues and eigenvectors)

# whiten = True  # True or False → For pixels (0-255) and image processing
# pca.fit(data)


# Attributes:

# np.cumsum(pca.explained_variance_ratio_)
# pca.explained_variance_ratio_

# coefficients of the linear transformation of the original data
# pca.components_

# number of components
# pca.n_components_

# data coordinates using the principal components
# data_pca = pca.transform(data)
