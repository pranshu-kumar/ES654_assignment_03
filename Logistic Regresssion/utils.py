import numpy as np

def sigmoid(x):
    '''
    Function to compute sigmoid(x)
    '''
    return 1/(1+np.exp(-x))