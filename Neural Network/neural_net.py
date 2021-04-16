# Importing libraries
import numpy as np
import jax.numpy as jnp
from jax import grad
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, X=None, y=None, layer_info=None, activations=None):
        '''
        > X: input X
        > y: output y
        > layer_info: list of format [n1,n2,..nh] where ni is the number of neurons in i^th hidden layer
        > activations: [g1, g2, ..., gh] where gi in {‘relu’, ‘identity’,‘sigmoid’} are the activations for i^th layer
        '''
        self.parameters = {}
        self.layer_info = layer_info
        self.activations = activations
        self.X = X
        self.y = y
        
        self.num_layers = len(layer_info)



    def initialize_parameters(self):
        '''
        Initialize layers with random weights
        '''
        for l in range(1, self.num_layers):
            self.parameters['Weight' + str(l)] = np.random.randn(self.layer_info[l], self.layer_info[l-1])*0.01
            self.parameters['Bias' + str(l)] = np.zeros((self.layer_info[l], 1))

        return self.parameters


    # Forward Propagation
    def calculate_Z(self, A, W, b):
        '''
        Function to calculate Z
        > A: Activations from the previous layer
        > W: weight matrix of the current layer
        > b: bias vector of the current layer
        '''
        print(A.shape, W.shape, b.shape)
        Z = W @ A + b
        # print(Z.shape)
        cache = (A,W,b)
        # assert Z shape below

        return Z, cache

    def sigmoid(self, Z):
        '''
        Function to calculate the sigmoid of Z
        '''
        g_z = 1/(1 + np.exp(-Z))
        # print("g(z) shape:", g_z.shape)
        return g_z, Z
    
    def relu(self,Z):
        '''
        Function to calculate the ReLU of Z
        '''
        zero = np.zeros((Z.shape))
        return np.maximum(zero,Z), Z

    def activate_Z(self, A, W, b, activation):
        '''
        Function to calculate g(Z) or activation of Z
        > A: activations from prev layer
        > W: weight matrix of current layer
        > b: bias vector of current layer
        > activation: activation name for current layer
        '''
        if activation == 'relu':
            Z, l_cache = self.calculate_Z(A, W, b)
            A_curr, a_cache = self.relu(Z)

        elif activation == 'sigmoid':
            Z, l_cache = self.calculate_Z(A, W, b)
            A_curr, a_cache = self.sigmoid(Z)

        # assert A_curr shape
        cache = (l_cache, a_cache)
        
        return A_curr, cache

    def forward_prop(self):
        '''
        Function to perform forward propagation 
        '''
        caches = []
        A = self.X
        
        for l in range(1, self.num_layers-1):
            print("Layer:", l)
            A_prev = A
            A, cache = self.activate_Z(A_prev, self.parameters['Weight' + str(l)], self.parameters['Bias' + str(l)], activation=self.activations[l])
            # print(A)
            caches.append(cache)
            break

        AL, cache = self.activate_Z(A, self.parameters['Weight' + str(self.num_layers-1)], self.parameters['Bias' + str(self.num_layers-1)], activation=self.activations[self.num_layers-1])
        # print(AL)
        caches.append(cache)
        # assert AL shape

        return AL, caches

    
    # Backward Propagation
    def backward_linear(self, dZ, cache):
        '''
        Function to implement ...
        > dZ:
        > cache:
        '''
        
        A_prev, W, b = cache
        n = A_prev.shape[1]

        dW = 1 / n * dZ @ A_prev.T
        db = 1 / n * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T @ dZ

        return dA_prev, dW, db



    def backward_relu(self, dA, cache):
        pass

    def backward_sigmoid(self, dA, cache):
        '''
        Function to implement ...
        '''

        Z = cache
        g_z, _ = self.sigmoid(Z)

        dZ = dA*g_z*(1-g_z)

        return dZ


    def backward_activation(self, dA, cache, activation):
        '''
        Function to implement ...
        > dA:
        > cache:
        > activation
        '''
        l_cache, a_cache = cache

        if activation == 'relu':
            dZ = self.backward_relu(dA, a_cache)
            dA_prev, dW, db = self.backward_linear(dZ, l_cache)

        elif activation == 'sigmoid':
            dZ = self.backward_sigmoid(dA, a_cache)
            dA_prev, dW, db = self.backward_linear(dZ, l_cache)

        return dA_prev, dW, db

    def backward_prop(self, AL, y, caches):
        '''
        Function to perform Backward Propagation
        > AL:
        > y: 
        > caches:
        '''

        gradients = {}
        
        y = y.reshape(AL.shape)

        dAL = -(np.divide(y,AL)-np.divide(1-y, 1-AL))

        curr_cache = caches[self.num_layers-1]

        gradients['dA' + str(self.num_layers-1)], gradients['dW'+str(self.num_layers-1)], gradients['db' + str(self.num_layers-1)] = self.backward_activation(dAL, curr_cache, self.activations[self.num_layers-1])

        for l in range(self.num_layers-2, 0, -1):
            curr_cache = caches[l]
            dA_prev, dW, db = self.backward_activation(gradients['dA' + str(l+1), curr_cache, self.activations[l]])
            gradients['dA' + str(l)] = dA_prev
            gradients['dW' + str(l+1)] = dW
            gradients['db' + str(l+1)] = db

        return gradients

    
    def parameters_update(self, gradients, learning_rate):
        '''
        Function to update parameters
        > gradients:
        > learning_rate: 
        '''

        for l in range(1, self.num_layers):
            self.parameters['Weight' + str(l)] = self.parameters['Weight' + str(l)] - learning_rate * gradients['dW' + str(l)]
            self.parameters['Bias' + str(l)] = self.parameters['Bias' + str(l)] - learning_rate * gradients['db' + str(l)]

        return self.parameters

    def cross_entropy_loss(self, y_hat, y):
        '''
        Function to implement cross entropy loss
        > y_hat:
        > y: 
        '''