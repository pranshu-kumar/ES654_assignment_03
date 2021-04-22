import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from NeuralNetwork.neural_net import NeuralNetwork

# Get X and y
X, y = load_boston(return_X_y=True)

# Normalize Dataset

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = y.reshape((1, len(y)))
# y = scaler.fit_transform(y)

# unscaled = scaler.inverse_transform(scaled)


# K-Fold, K=3
kf = KFold(n_splits=3)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train,X_test = X[train_index], X[test_index]
    y_train, y_test = y[0,train_index], y[0,test_index]

X_train = X_train.T
X_test = X_test.T
y_train = y_train.reshape((1, len(y_train)))
y_train = scaler.fit_transform(y_train)

y_test = y_test.reshape((1, len(y_test)))
y_test = scaler.fit_transform(y_test)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Neural Network Model
nn = NeuralNetwork(layer_info=[13, 5, 1], activations=['sigmoid', 'relu', 'sigmoid'])
nn.train_model(X_train, y_train, verbose=True, plot_loss=False)

y_hat = nn.predict(X_test)

y_hat = scaler.inverse_transform(y_hat)
y_test = scaler.inverse_transform(y_test)

# Calculate rmse
sq_sum = 0
for i in range(y_hat.shape[1]):
    sq_sum += (y_test[0,i] - y_hat[0,i])**2
rmse = sq_sum/y_hat.shape[1]

print("RMSE: ", rmse)


y_hat = y_hat.reshape((y_hat.shape[1],))
y_test = y_test.reshape((y_test.shape[1],))
# print(range(1, y_hat.shape[1]+1, 1))
# plt.hist2d()
plt.plot(range(1, len(y_hat)+1, 1), y_hat, '--', label="Prediction")
plt.plot(range(1, len(y_hat)+1, 1), y_test, alpha=0.7, label="Test Data")
plt.legend()
plt.grid()
plt.title("RMSE:{}".format(rmse))
plt.show()
