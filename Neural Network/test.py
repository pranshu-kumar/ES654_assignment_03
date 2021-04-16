from neural_net import NeuralNetwork
import numpy as np

X = np.random.rand(3,4)
# print("X:\n", X)

# Test parameters init
nn = NeuralNetwork(X=X, layer_info=[3,2], activations=['sigmoid', 'sigmoid', 'sigmoid'])
# parameters = nn.initialize_parameters()
# print("W1 = " + str(parameters["Weight1"]))
# print("b1 = " + str(parameters["Bias1"]))
# print("W2 = " + str(parameters["Weight2"]))
# print("b2 = " + str(parameters["Bias2"]))

# # Test forward prop
# AL, _ = nn.forward_prop()
# print("AL:\n", AL)

# Test backward Prop
np.random.seed(3)
AL = np.random.randn(1, 2)
Y = np.array([[1, 0]])

A1 = np.random.randn(4,2)
W1 = np.random.randn(3,4)
b1 = np.random.randn(3,1)
Z1 = np.random.randn(3,2)
linear_cache_activation_1 = ((A1, W1, b1), Z1)

A2 = np.random.randn(3,2)
W2 = np.random.randn(1,3)
b2 = np.random.randn(1,1)
Z2 = np.random.randn(1,2)
linear_cache_activation_2 = ((A2, W2, b2), Z2)

caches = (linear_cache_activation_1, linear_cache_activation_2)
print(len(caches))
grads = nn.backward_prop(AL, Y, caches)
print(grads)