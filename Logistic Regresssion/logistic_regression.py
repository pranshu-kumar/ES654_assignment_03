# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from jax import grad
# import jax.np as jnp

from utils import *


class LogisticRegressor():
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y


    def initialize_parameters(self, num_samples):
        '''
        Function to initialize W, b matrix
        > num_samples: No of rows in X or number of samples
        '''
        W = np.zeros((num_samples,1))
        b = 0

        return W,b

    # Forward and Backward Propagation
    def forward_backward_prop(self, W, b):
        '''
        Function to implement Forward Propagation
        > W: Weight matrix
        > b: bias matrix
        '''

        n = self.X.shape[1]

        A = sigmoid(np.dot(W.T,self.X)+b)

        loss = -1/n*np.sum(self.y*np.log(A) + (1-self.y)*np.log(1-A))

        dw = 1/n * np.dot(self.X, (A-self.y).T)
        db = 1/n * np.sum(A-self.y)

        loss = np.squeeze(loss)

        gradients = {
            'dW':dw,
            'db':db
        }

        return gradients, loss

    # Train Model
    def train_model(self, num_iter=2000, learning_rate=0.001, verbose=False, plot_loss=True):
        '''
        Function to fit the model
        '''
        loss_list = []

        W, b = self.initialize_parameters(self.X.shape[0])

        for i in range(num_iter):
            gradients, loss = self.forward_backward_prop(W,b)

            # parameters update
            W = W - learning_rate*gradients['dW']
            b = b - learning_rate*gradients['db']

            if i % 10 == 0:
                loss_list.append(loss)

            if verbose:
                if i % 100 == 0:
                    print("> Loss after iteration {}: {}".format(i, loss))

            dw = gradients['dW']
            db = gradients['db']

            parameters = {
                'w':W,
                'b':b
            }

            gradients = {
                'dW':dw,
                'db':db

            }

        if plot_loss:
            self.plot_loss(loss_list)

        return parameters, gradients


    # Plot loss
    def plot_loss(self,loss_list):
        '''
        Function to plot loss
        '''
        plt.plot(np.squeeze(loss_list))
        plt.grid()
        plt.show()



        