import numpy as np
import copy, math

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def pred_func(x, w, b):
    p = np.dot(x,w) + b
    return p

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    fwb_i = np.zeros(m)
    for i in range(m):
        fwb_i[i] = pred_func(X[i], w, b)
    err = fwb_i - y
    err_square = err**2
    cost = sum(err_square) / (2 * m)
    return cost

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
cost = compute_cost(X_train, y_train, w_init, b_init)

print(f'Cost at optimal w : {cost}')