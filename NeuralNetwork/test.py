from neural_net import NeuralNetwork
import numpy as np
import random

X = np.random.rand(64,1198)
y = [0]*(1198-40) + [1]*40
random.shuffle(y)
y = np.array(y)
y = np.reshape(y,(1,1198))
# print(y)
# print("X:\n", X)
# print("y:\n", y)

# Test parameters init
nn = NeuralNetwork(layer_info=[64,20,1], activations=['sigmoid', 'sigmoid', 'sigmoid'])
nn.train_model(X=X, y=y, verbose=True)
# nn.initialize_parameters()
# print(nn.parameters['Bias1'].shape)
# print(np.array(params[0][0]).shape)