import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from LogisticRegresssion.logistic_regression import LogisticRegressor
import matplotlib.pyplot as plt
import copy


# Load Dataset
X,y = load_breast_cancer(return_X_y=True)
# X = X.astype(np.float128)
temp_X = copy.deepcopy(X)
# print(temp_X)
# Normalization
x_mean = X.mean()
x_std = X.std()
X = (X-x_mean)/x_std
# print(temp_X)
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
print("Logistic Regression on Breast Cancer Model ->")

lr = LogisticRegressor(X=X_train, y=y_train)
_, parameters = lr.train_model(num_iter=10000,learning_rate=0.001, verbose=False, plot_loss=False)

# Predict
y_hat = lr.predict(parameters,X_test, y_test)


# Plotting decision boundary
print("\nPlotting Decision Boundary!")
plot_X = X[:,3:5]
data = load_breast_cancer()
feature_names = [data.feature_names[3], data.feature_names[4]]
# print(feature_names)

min1, max1 = plot_X[:, 0].min()-1, plot_X[:, 0].max()+1
min2, max2 = plot_X[:, 1].min()-1, plot_X[:, 1].max()+1

h = 0.2

x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)

xx, yy = np.meshgrid(x1grid, x2grid)

r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

grid = np.hstack((r1,r2))

lr = LogisticRegressor(X=plot_X, y=y)
_, parameters = lr.train_model(num_iter=10000,learning_rate=0.001, verbose=False, plot_loss=False)

yhat = lr.predict(parameters, X=grid)
# y_hat = np.array(y_hat)
# print(xx.shape)
zz = yhat.reshape(xx.shape)

plt.contourf(xx, yy, zz, cmap='Paired')

for class_val in range(2):
    row_ix = np.where(y == class_val)
    plt.scatter(plot_X[row_ix, 0], plot_X[row_ix, 1], edgecolors='k',cmap='Paired')
# print(xx)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Decision Boundary")
plt.show()
