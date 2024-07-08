import numpy as np
import math
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], 
                    [1416, 3, 2, 40], 
                    [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def pred_func(x, w, b):
    p = np.dot(x,w) + b
    return p

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    fwb = np.zeros(m)
    for i in range(m):
        fwb[i] = pred_func(X[i], w, b)
    err = fwb - y
    err_square = err**2
    cost = sum(err_square) / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        err = pred_func(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw , dj_db

def gradient_descent(X, y, w_in, b_in, learning_rate, iters):
    w = w_in
    b = b_in
    cost = 0
    cost_history = []

    for i in range(iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - (learning_rate * dj_dw)
        b = b - (learning_rate * dj_db)

        cost = compute_cost(X, y, w, b)
        if i < 1000:
            cost_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost:8.2f}   ")

    return w, b, cost_history


def checking_gradient_descent(cost_history):
    n = len(cost_history)
    x = np.arange(1, n+1)
    y = np.array(cost_history)

    plt.plot(x, y)

    plt.title('checking gradient descent')
    plt.xlabel('iteration')
    plt.ylabel('cost')

    plt.ylim(680, 765)

    plt.show()


b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
cost = compute_cost(X_train, y_train, w_init, b_init)

print(f'Cost at optimal w : {cost}')

tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)

print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

# initialize parameters
initial_w = np.zeros_like(w_init)
print(initial_w)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent
w_final, b_final, cost_history = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

checking_gradient_descent(cost_history)