import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from LogisticRegresssion.logistic_regression import LogisticRegressor

# Load Dataset
X,y = load_breast_cancer(return_X_y=True)
# X = X.astype(np.float128)

# Normalization
x_mean = X.mean()
x_std = X.std()
X = (X-x_mean)/x_std

# K-Fold, K=3
kf = KFold(n_splits=3)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train,X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# train model
lr = LogisticRegressor(X=X_train, y=y_train)
_, parameters = lr.train_model(num_iter=10000,learning_rate=0.00075, verbose=False, plot_loss=False)

# Predict
y_hat = lr.predict(parameters,X_test, y_test)





