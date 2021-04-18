import numpy as np

# Helper Functions
def sigmoid(x):
    '''
    Function to compute sigmoid(x)
    '''
    # print(x)
    return 1/(1+np.exp(-x))

def accuracy(y_hat, y):
    '''
    Function to print accuracy
    '''
    assert(y_hat.shape == y.shape)

    count = 0
    for i in range(y.shape[0]):
        if y_hat[i] == y[i]:
            count += 1

    # print(count)

    return count/y.shape[0]