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
kf = KFold(n_splits=4)
kf.get_n_splits(X)

best_acc = 0
for train_index, test_index in kf.split(X):
    temp_X_train, temp_X_test = X[train_index], X[test_index]
    temp_y_train, temp_y_test = y[train_index], y[test_index]
    temp_X_train = temp_X_train.T
    temp_X_test = temp_X_test.T

    # One Hot Encoding
    enc = OneHotEncoder(sparse=False, categories='auto')
    temp_y_train = enc.fit_transform(temp_y_train.reshape(len(temp_y_train), -1)).T
    temp_y_test = enc.transform(temp_y_test.reshape(len(temp_y_test), -1)).T

    # Multi-Class Regressor
    mcr = MultiClassRegressor()
    mcr.fit(temp_X_train, temp_y_train, verbose=False, plot_loss=False)

    temp_y_hat, temp_y_test, acc = mcr.predict(temp_X_test, temp_y_test)
    
    if acc > best_acc:
        best_acc = acc
        y_test = temp_y_test
        y_hat = temp_y_hat




# Plot Confusion Matrix
ax= plt.subplot()
cf_matrix = confusion_matrix(y_test, y_hat)
sns.heatmap(cf_matrix, annot=True, fmt='g', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix")
plt.show()
# plt.show()
# plt.title('Confusion Matrix')
# plt.xlabel("Predicted")
# plt.ylabel("True Labels")
# fig.savefig('confusionmatrix.png')

# PCA
X,y = load_digits(return_X_y=True) 
pca = PCA(n_components=2)

principal_comp = pca.fit_transform(X)

pca_df = pd.DataFrame(data=principal_comp, columns=['pc1', 'pc2'])
df = pd.concat((pca_df, pd.DataFrame(y)), axis=1)
# print(df.columns)
# print(df.head(20))

fig = plt.figure(figsize=(8,8))
targets = range(0,10,1)
# targets = list(map(str,targets))
colors = ['#ff6600', '#ff9999', '#ffff99', '#ff99ff', '#66ffff', '#9900cc', '#66ff66', '#800000', '#cc6699', '#9999ff']

for t, c in zip(targets, colors):
    idx = df[0] == t
    plt.scatter(df.loc[idx, 'pc1'], df.loc[idx, 'pc2'], c=c, s=50)

plt.legend(targets)
plt.grid()
plt.title("PCA Plot")
plt.xlabel("Principle Component 01")
plt.ylabel("Principle Component 02")
plt.show()
fig.savefig('pcaplot.png')
