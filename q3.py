import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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



