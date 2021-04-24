import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork.neural_net import NeuralNetwork

# Load data
data = load_digits()

# Visualize data
# plt.matshow(data.images[4], cmap='gray')
# plt.show()

# Get X and y
X,y = load_digits(return_X_y=True)


print("shape of X:", X.shape)
print("shape of y:", y.shape)

# Normalize dataset
X = X/64

# K-Fold, K=3
kf = KFold(n_splits=3)
kf.get_n_splits(X)

best_acc = 0
for train_index, test_index in kf.split(X):
    X_train,X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = X_train.T
    X_test = X_test.T

    # One Hot Encoding
    enc = OneHotEncoder(sparse=False, categories='auto')
    y_train = enc.fit_transform(y_train.reshape(len(y_train), -1)).T
    y_test = enc.transform(y_test.reshape(len(y_test), -1)).T
    
    # Defining Neural Network
    nn = NeuralNetwork(layer_info=[64, 20, 10], activations=['sigmoid', 'relu', 'softmax'])
    nn.train_model(X=X_train, y=y_train,verbose=False, plot_loss=False, num_iter=5000)

    # Predict
    y_hat, acc = nn.predict(X=X_test, y=y_test)

    if acc > best_acc:
        best_acc = acc
    


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


