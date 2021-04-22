import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
from matplotlib import cm
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

import seaborn as sns
from LogisticRegresssion.logistic_regression import MultiClassRegressor


# Get X and y
X,y = load_digits(return_X_y=True)
print("shape of X:", X.shape)
print("shape of y:", y.shape)

# Normalize dataset
X = X/64

# K-Fold, K=3
kf = KFold(n_splits=3)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train,X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


X_train = X_train.T
X_test = X_test.T

# One Hot Encoding
enc = OneHotEncoder(sparse=False, categories='auto')
y_train = enc.fit_transform(y_train.reshape(len(y_train), -1)).T
y_test = enc.transform(y_test.reshape(len(y_test), -1)).T

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Multi-Class Regressor
mcr = MultiClassRegressor()
mcr.fit(X_train, y_train, verbose=True, plot_loss=True)

y_hat, y_test = mcr.predict(X_test, y_test)

# Plot Confusion Matrix
fig = plt.figure(figsize=(10,8))
cf_matrix = confusion_matrix(y_test, y_hat)
sns.heatmap(cf_matrix, annot=True)
plt.show()
fig.savefig('confusionmatrix.png')

# PCA
X,y = load_digits(return_X_y=True) 
pca = PCA(n_components=2)

principal_comp = pca.fit_transform(X)

pca_df = pd.DataFrame(data=principal_comp, columns=['pc1', 'pc2'])
df = pd.concat((pca_df, pd.DataFrame(y)), axis=1)
# print(df.columns)
# print(df.head(20))

fig = plt.figure(figsize=(10,8))
targets = range(0,10,1)
# targets = list(map(str,targets))
colors = ['#ff6600', '#ff9999', '#ffff99', '#ff99ff', '#66ffff', '#9900cc', '#66ff66', '#800000', '#cc6699', '#9999ff']

for t, c in zip(targets, colors):
    idx = df[0] == t
    plt.scatter(df.loc[idx, 'pc1'], df.loc[idx, 'pc2'], c=c, s=50)

plt.legend(targets)
plt.grid()
plt.title("PCA Plot")
plt.show()
fig.savefig('pcaplot.png')
