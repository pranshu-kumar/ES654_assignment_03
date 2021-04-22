# Importing Libraries
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
# import jax.np as jnp
from LogisticRegresssion.utils import *
from LogisticRegresssion.optimizer import param_update


class LogisticRegressor():
    def __init__(self, X=None, y=None, regularization='unreg', lambd=None):
        self.X = X
        self.y = y

        self.regularization = regularization
        if self.regularization == 'l1' or self.regularization == 'l2' :
            self.lambd = lambd
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
    def cost_function(self, A, n, W):
        loss = -(1/n)*np.sum(self.y.T*np.log(A) + (1-self.y.T)*np.log(1-A))
        
        # Unregularized
        if self.regularization == 'unreg':
            return loss
        
        # L2 Regularization
        elif self.regularization == 'l2':
            reg = (self.lambd/(2*n)) * (W.T @ W) 

        # L1 Regularization    
        elif self.regularization == 'l1':
            reg = (self.lambd/(2*n)) * abs(W)
        
        loss += reg
        return loss

    def compute_grad(self, A, W, n):

        dw = (1/n) * (np.dot(self.X.T, (A-self.y.T).T))
        db = (1/n) * (np.sum(A-self.y.T))
        
        # Unregularized
        if self.regularization == 'unreg':
            return dw, db

        # L2 Regularization
        elif self.regularization == 'l2':
            dw += (self.lambd/n) * W

        # L1 Regularization
        elif self.regularization == 'l1':
            dw += (self.lambd/n) * W

        return dw, db

            
    def forward_backward_prop(self, W, b):
        '''
        Function to implement Forward Propagation
        > W: Weight matrix
        > b: bias matrix
        '''

        n = self.X.shape[0]
        A = sigmoid(np.dot(W,self.X.T)+b)
        
        # Unregularized
        loss = self.cost_function(A, n, W)
        # loss = -(1/n)*np.sum(self.y.T*np.log(A) + (1-self.y.T)*np.log(1-A))

        dw, db = self.compute_grad(A, W, n)

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


    def predict(self, parameters, X, y=None):
        '''
        Function to make predictions using the trained weights

        > Returns accuracy
        '''
        # print(y)
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

        if y is None:
            return y_hat

        acc = accuracy(y_hat, y)

        print("Test Accuracy:", acc)

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


class MultiClassRegressor():
    def __init__(self):
        # self.theta = None
        self.loss_list = []

    def initialize_theta(self):
        '''
        Function to initialize weight matrix
        '''
        theta = np.zeros((self.y.shape[0], self.X.shape[0]))
        return theta
    
    def softmax(self,Z):
        '''
        Function to implement softmax(Z)
        '''
        return np.exp(Z)/np.sum(np.exp(Z), axis=1,keepdims=True)

    def get_y_hat(self, X, theta):
        Z = np.dot(X,theta)
        y_hat = self.softmax(Z)

        return y_hat

    def loss_func(self, theta, iter, y_hat):
        '''
        Function to compute loss
        '''
        loss = np.mean(-np.sum(np.log(y_hat) * self.y))
        
        if self.verbose:
            if iter%10 == 0:
                print("> Loss after iteration {}: {}".format(iter, loss))
                self.loss_list.append(loss)

        return loss

    def fit(self, X, y, num_iter=1000, learning_rate=0.001, verbose=False, plot_loss=True):
        '''
        Function to fit training data to the MultiClassRegressor Model
        '''
        self.X = X
        
        self.y = y
        self.verbose = verbose 
        self.X = np.vstack([np.ones((1,X.shape[1])), self.X])
        # print(self.X)
        theta = self.initialize_theta()
        
        print(X.shape, theta.shape)
        # print(theta.shape)

        # training_grad_func = grad(self.loss_func)

        # theta = param_update(training_grad_func, theta)
        # for i in range(num_iter):
        #     der = training_grad_func(theta, i)
        #     theta -= theta * learning_rate

        for i in range(num_iter):
            y_hat = self.softmax((np.dot(theta, self.X)))
            
            loss = self.loss_func(theta, i, y_hat)
            
            der = (1/self.X.shape[1]) * np.dot((y_hat - self.y), self.X.T)
            
            theta = theta - learning_rate * der

        print(y_hat.shape)
        # print(self.loss_func(theta))
        self.trained_theta = theta
        # print(theta)

        if plot_loss:
            self.plot_loss(learning_rate)


    def predict(self, X=None, y=None):
        '''
        Function to make predictions
        '''
        X = np.vstack([np.ones((1,X.shape[1])), X])
        AL = self.softmax((np.dot(self.trained_theta, X)))
        
        y_hat = AL.argmax(axis=0)
        y = np.argmax(y, axis=0)

        acc = (y_hat == y).mean()
        print("Accuracy:", acc)
        # print(y_hat.shape, y.shape)
        # print(y_hat)
        
    
    def plot_loss(self, learning_rate):
        plt.plot(np.squeeze(self.loss_list))
        plt.grid()
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.title("Loss vs Number of iterations\nLearning Rate: {}".format(learning_rate))
        plt.show()






        