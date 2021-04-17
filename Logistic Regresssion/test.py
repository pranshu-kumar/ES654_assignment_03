import numpy as np
import random
from logistic_regression import LogisticRegressor

X = np.random.rand(100,50)
y = [0]*10 + [1]*40
random.shuffle(y)
y = np.array(y)
y = np.reshape(y,(1,50))
# print(y)
print("X:\n", X)
print("y:\n", y)

lr = LogisticRegressor(X, y)

lr.train_model(verbose=True)

