import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from NeuralNetwork.neural_net import NeuralNetwork
import copy 

# Get X and y
X, y = load_boston(return_X_y=True)

# Normalize Dataset

scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
y = y.reshape((1, len(y)))
# y = scaler.fit_transform(y)

# unscaled = scaler.inverse_transform(scaled)


# K-Fold, K=3
kf = KFold(n_splits=3)
kf.get_n_splits(X)

best_rmse = np.inf
for train_index, test_index in kf.split(X):
    temp_X_train,temp_X_test = X[train_index], X[test_index]
    temp_y_train, temp_y_test = y[0,train_index], y[0,test_index]

    temp_X_train = scaler.fit_transform(temp_X_train)
    temp_X_train = temp_X_train.T

    temp_unscaled_X_test = copy.deepcopy(temp_X_test)
    temp_unscaled_X_test = temp_unscaled_X_test.T

    temp_X_test = scaler.fit_transform(temp_X_test)
    temp_X_test = temp_X_test.T

    temp_y_train = temp_y_train.reshape((len(temp_y_train),1))
    # y_train = scaler.fit_transform(y_train)
    temp_y_train = temp_y_train.T

    temp_y_test = temp_y_test.reshape((len(temp_y_test),1))
    # temp_y_test = scaler.fit_transform(temp_y_test)
    temp_y_test = temp_y_test.T
    # print(y_test)

    # Neural Network Model
    nn = NeuralNetwork(layer_info=[13,5,1], activations=['sigmoid', 'relu', 'linear'])
    nn.train_model(temp_X_train, temp_y_train, verbose=False, plot_loss=False, num_iter=5000, learning_rate=0.001)

    temp_y_hat, rmse = nn.predict(temp_X_test, temp_y_test)
    if rmse < best_rmse:
        X_train,X_test = X[train_index], X[test_index]
        y_train,y_test = y[0,train_index], y[0,test_index]
        y_hat = temp_y_hat
        # y_test = y_test.T
        unscaled_X_test = temp_unscaled_X_test
        best_rmse = rmse


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)



# y_hat = scaler.inverse_transform(y_hat)
# y_test = scaler.inverse_transform(y_test)




y_hat = y_hat.reshape((y_hat.shape[1],))
# y_test = y_test.reshape((y_test.shape[1],))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
fig.suptitle('Predictions - Boston Housing Prices Dataset')

ax1.scatter(unscaled_X_test[0,:], y_hat, color='g',label="Prediction")
ax1.scatter(unscaled_X_test[0,:], y_test, alpha=0.7, label="Test Data")
ax1.set_title("Predicted Price vs CRIM\nRMSE:{}".format(best_rmse))
ax1.set_xlabel("CRIM")
ax1.set_ylabel("Price")
ax1.grid()
ax1.legend()

ax2.plot(np.squeeze(y_hat), '--', label="Prediction")
ax2.plot(np.squeeze(y_test), label="Test Data")
ax2.set_ylabel("Price")
ax2.set_xlabel("Samples")
ax2.set_title("Predicted Price vs Samples\nRMSE:{}".format(best_rmse))
ax2.legend()
ax2.grid()

plt.show()

fig.savefig("bostonpredictions.png")

# fig2 = plt.figure(figsize=(8,8))
# 
# # # fig.savefig("pricevscrim.png")
# # fig.savefig('pricevssample.png')
# plt.show()