import numpy as np
import random
from logistic_regression import LogisticRegressor

X = np.random.rand(380,30)
y = [0]*300 + [1]*80
random.shuffle(y)
y = np.array(y)
# y = np.reshape(y,(380,1))
# print(y)
print("X:\n", X)
print("y:\n", y)

lr = LogisticRegressor(X, y)

lr.train_model(verbose=True)

