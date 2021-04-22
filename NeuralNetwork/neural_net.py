# Importing libraries
# import numpy as np
import autograd.numpy as np
from autograd import grad
import pandas as pd
import matplotlib.pyplot as plt
from NeuralNetwork.utils import *
from sklearn.preprocessing import OneHotEncoder
import copy
from NeuralNetwork.optimizer import param_update




class NeuralNetwork():
    def __init__(self, layer_info=None, activations=None):
        '''
        > X: input X
        > y: output y
        > layer_info: list of format [n1,n2,..nh] where ni is the number of neurons in i^th hidden layer
        > activations: [g1, g2, ..., gh] where gi in {‘relu’, ‘identity’,‘sigmoid’} are the activations for i^th layer
        '''
        self.parameters = {}
        self.layer_info = layer_info
        self.activations = activations
        
        self.num_layers = len(layer_info)

        self.loss_list = []

    def initialize_parameters(self):
        '''
        Initialize layers with random weights
        '''
        scale = 0.01
        return [
            (scale * np.random.randn(m, n).T, 
            scale * np.random.randn(n)) 
            for m, n in zip(self.layer_info[:-1], self.layer_info[1:])]


    # Forward Propagation
    def sigmoid(self, Z):
        '''
        Function to calculate the sigmoid of Z
        '''
        g_z = 1/(1 + np.exp(-Z))
        # print("Z shape:", Z.shape)
        return g_z
    
    def relu(self,Z):
        '''
        Function to calculate the ReLU of Z
        '''
        return np.maximum(0,Z)

    def softmax(self, Z):
        return np.exp(Z)/np.sum(np.exp(Z), axis=1,keepdims=True)
        # expZ = np.exp(Z - np.max(Z))
        # return expZ / expZ.sum(axis=1, keepdims=True)

    def forward_prop(self, X, params):
        '''
        Function to perform forward propagation 
        '''
        layer_num = 0
        A = X
        for W,b in params:
            b = b.reshape((len(b),1))
            A_prev = A
            # print(W.shape, A_prev.shape)
            # print("Layer: ", layer_num)
            Z = np.dot(W, A_prev) + b
            # print(W.shape, A_prev.shape)
            # print(Z.shape)
            if self.activations[layer_num] == 'sigmoid':
                A = self.sigmoid(Z)
            elif self.activations[layer_num] == 'relu':
                A = self.relu(Z)
            layer_num += 1
        
        if self.activations[-1] == 'sigmoid':
            y_hat = self.sigmoid(Z)
        elif self.activations[-1] == 'relu':
            y_hat = self.relu(Z)
        elif self.activations[-1] == 'softmax':
            y_hat = self.softmax(Z)
        elif self.activations[-1] == 'linear':
            y_hat = Z

        return y_hat


    def loss_func(self, params, iter):
        '''
        Function to implement cross entropy loss
        > y_hat:
        > y: 
        '''

        n = self.y.shape[1]
        
        y_hat = self.forward_prop(self.X, params)
        # print(self.y.shape)
        if self.activations[-1] == 'softmax':
            # Softmax Loss
            loss = np.mean(-np.sum(np.log(y_hat) * self.y))
        
        elif self.activations[-1] == 'linear':
            # MSE loss
            loss = np.square(np.subtract(self.y,y_hat)).mean()
        
        else:
            # Cross Entropy Loss
            loss = -1/n*(np.sum(self.y*np.log(y_hat) + (1-self.y)*np.log(1-y_hat)))
        
        
        if self.verbose:
            if iter % 100 == 0:
                print("> loss after iteration {}: {}".format(iter, loss._value))
                # print(loss._value)
        
        self.loss_list.append(loss._value)
        return loss

    def train_model(self, X, y, learning_rate = 0.001, num_iter=2500, verbose=False, plot_loss=True):
        '''
        Function to train model
        > learning_rate: learning rate for gradient descent
        > num_iter: number of iterations
        > verbose: Print loss or not
        '''
        X_temp = copy.deepcopy(X)
        y_temp = copy.deepcopy(y)
        
        self.X = X
        self.y = y

        self.verbose = verbose
        
        loss_list = []
        params = self.initialize_parameters()


        training_gradient_func = grad(self.loss_func)

        # update parameters
        self.final_params = param_update(training_gradient_func, params, learning_rate=learning_rate, num_iter=num_iter)
        
        if plot_loss:
            self.plot_loss(self.loss_list, learning_rate)


    def predict(self, X, y=None):
        '''
        Function to predict 
        '''
        AL = self.forward_prop(X, self.final_params)
        # print(AL[:,2],AL[:,4])
        if self.activations[-1] == 'softmax':
            
            # print(AL.shape-)
            y_hat = AL.argmax(axis=0)
            y = np.argmax(y, axis=0)

            acc = (y_hat == y).mean()
            # print(y_hat.shape, y.shape)
            print(y_hat)
            # y_hat = y_hat.reshape((1, len(y_hat)))
            # # print(y)
        
        else:
            # Classification Problem
            # print(np.unique(y))
            if len(np.unique(y)) == self.layer_info[-1] or len(np.unique(y)) == 2:
                y_hat = np.zeros((1, AL.shape[1]))
                for i in range(AL.shape[1]):
                    if AL[0,i] > 0.5:
                        y_hat[0,i] = 1
                    else:
                        y_hat[0,i] = 0
                # print(AL.shape)
                acc = (abs(y_hat - y)).mean()

                return y_hat
            
            # Regression Problem
            else:
                y_hat = AL
                return y_hat

        if y is None:
            return y_hat
        
       

        print("Test Accuracy:", acc)

        return y_hat

    def plot_loss(self,loss_list, learning_rate):
        '''
        Function to plot loss
        > loss_list: list containing losses at every 10 iterations
        > learning_rate: the learning rate for GD
        '''

        plt.plot(np.squeeze(loss_list))
        plt.grid()
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.title("Loss vs Number of iterations\nLearning Rate: {}".format(learning_rate))
        plt.show()

    def _get_one_hot_encoding(self, y):
        k = self.layer_info[-1]
        
        y_enc = np.zeros((self.X.shape[1], k))
        k = np.arange(0, k, 1)

        for cls in k.astype(int):
            y_enc[np.where(y[:] == cls), cls] = 1


        return y_enc.T
        
