from neural_net import NeuralNetwork
import numpy as np
import random

X = np.random.rand(100,50)
y = [0]*10 + [1]*40
random.shuffle(y)
y = np.array(y)
y = np.reshape(y,(1,50))
# print(y)
print("X:\n", X)
print("y:\n", y)

# Test parameters init
nn = NeuralNetwork(X=X, y=y,layer_info=[100,50,1], activations=['sigmoid', 'sigmoid', 'sigmoid'])
nn.train_model(verbose=True)