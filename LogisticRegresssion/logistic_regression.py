# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from jax import grad
# import jax.np as jnp
from LogisticRegresssion.utils import *


class LogisticRegressor():
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

        # self.X = self.X.astype(np.float128)

        self.y = np.reshape(y,(self.X.shape[0],1))


    def initialize_parameters(self, num_features):
        '''
        Function to initialize W, b matrix
        > num_features: No of columns in X or number of samples
        '''
        # print(num_features)
        W = np.zeros((1,num_features))
        b = 0

        return W,b

    # Forward and Backward Propagation
    def forward_backward_prop(self, W, b):
        '''
        Function to implement Forward Propagation
        > W: Weight matrix
        > b: bias matrix
        '''

        n = self.X.shape[0]
        A = sigmoid(np.dot(W,self.X.T)+b)
        
        # Unregularized
        loss = -(1/n)*np.sum(self.y.T*np.log(A)) + (1-self.y.T)*np.log(1-A)

        dw = (1/n) * (np.dot(self.X.T, (A-self.y.T).T))
        db = (1/n) * (np.sum(A-self.y.T))

        # loss = np.squeeze(loss)
        # print(np.squeeze(loss))

        # assert(loss.shape == ())

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

        W, b = self.initialize_parameters(self.X.shape[1])

        for i in range(num_iter):
            gradients, loss = self.forward_backward_prop(W,b)

            # parameters update
            W = W - learning_rate*gradients['dW'].T
            b = b - learning_rate*gradients['db']

            if i % 10 == 0:
                loss_list.append(loss)

            if verbose:
                if i % 100 == 0:
                    print("> Loss after iteration {}: {}".format(i, loss))

            dw = gradients['dW']
            db = gradients['db']

            parameters = {
                'W':W,
                'b':b
            }

            gradients = {
                'dW':dw,
                'db':db

            }
        

        if plot_loss:
            self.plot_loss(loss_list, learning_rate)

        return gradients, parameters


    def predict(self, parameters, X, y):
        '''
        Function to make predictions using the trained weights

        > Returns accuracy
        '''
        W = parameters['W']
        b = parameters['b']
        A = sigmoid(np.dot(W,X.T) + b)

        y_hat = np.zeros((X.shape[0],))
        # print(A[0][4])
        for i in range(A.shape[1]):
            # print(i)
            if A[0][i] > 0.5:
                y_hat[i] = 1
                # print(y[i], y_hat[i])

        # print(y_hat)
        
        y_hat = np.array(y_hat)

        acc = accuracy(y_hat, y)

        print("Accuracy:", acc)

        return y_hat


    # Plot loss
    def plot_loss(self,loss_list, learning_rate):
        '''
        Function to plot loss
        '''
        plt.plot(np.squeeze(loss_list))
        plt.grid()
        plt.xlabel("Number of iterations (10 steps interval)")
        plt.ylabel("Loss")
        plt.title("Loss vs Number of iterations\nLearning Rate: {}".format(learning_rate))
        plt.show()



        