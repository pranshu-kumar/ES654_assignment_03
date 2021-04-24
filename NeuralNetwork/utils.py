def accuracy(y_hat, y):
    '''
    Function to print accuracy
    '''
    # print(y.shape, y_hat.shape)
    assert(y_hat.shape == y.shape)
    print(y_hat)
    count = 0
    for i in range(y.shape[1]):
        if y_hat[0,i] == y[0,i]:
            count += 1

    # print(count)

    return count/y.shape[1]