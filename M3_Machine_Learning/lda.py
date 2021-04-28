import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

# Set random seed
np.random.seed(0)

# load data
iris = load_iris()

df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df_iris)

# print(df_iris.describe())

df_iris["target"] = iris.target
df_iris["class"] = iris.target_names[iris.target]

# print(df_iris.sample(n=10))

### Split X & y
X = df_iris.iloc[:, :4]
y = df_iris["target"]


# now... implement the LDA (bear in mind that you do pass it the y's and it is supervised!)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()


# same process as always, identify your X's and your y's
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


# Scale the X's
from sklearn.preprocessing import StandardScaler

#######
df_iris.StandardScaler(X_train, X_test)


# apply the lda transformation
lda = LinearDiscriminantAnalysis().fit(X_train, y_train)


# select 2 principal components
lda.explained_variance_ratio_

