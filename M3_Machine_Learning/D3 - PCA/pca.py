from sklearn.decomposition import PCA

pca = PCA(…)

# Arguments in PCA:
n_components = number of components

svd_solver =  # 'randomized'… (eigenvalues and eigenvectors)

whiten = True  # True or False → For pixels (0-255) and image processing
pca.fit(data)


# Attributes:

# np.cumsum(pca.explained_variance_ratio_)
pca.explained_variance_ratio_

# coefficients of the linear transformation of the original data
pca.components_

# number of components
pca.n_components_

# data coordinates using the principal components
data_pca = pca.transform(data)
