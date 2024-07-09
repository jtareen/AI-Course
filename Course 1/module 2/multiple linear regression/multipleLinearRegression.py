import numpy as np
import math, os
import matplotlib.pyplot as plt
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Student_Performance.csv')

data = pd.read_csv(csv_path)

data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

independent_vars = ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']

X_train = data[independent_vars].to_numpy()
y_train = np.array(data['Performance Index'])


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

def checking_gradient_descent(cost_history, pred_val, trgt_val):
    n = len(cost_history)
    x_1 = np.arange(1, n+1)
    y = np.array(cost_history)

    x_2 = np.arange(1, len(pred_val)+1)

    plt.subplot(1, 2, 1)
    plt.plot(x_1, y)

    plt.title('checking gradient descent')
    plt.xlabel('iteration')
    plt.ylabel('cost')

    plt.ylim(1, 50)

    plt.subplot(1, 2, 2)
    plt.scatter(x_2, pred_val, color='orange', label='predicted value')
    plt.scatter(x_2, trgt_val, color='blue', label='actual values')

    plt.title('actual values vs predicted values')
    plt.xlabel('student No')
    plt.ylabel('performance index')

    plt.show()

total_features = X_train.shape[1]
# initialize parameters
initial_w = np.zeros(total_features)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 9.0e-5
# run gradient descent
w_final, b_final, cost_history = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
predicted_values = []
actual_values = []
for i in range(0, m, 500):
    predicted_values.append(np.dot(X_train[i], w_final) + b_final)
    actual_values.append(y_train[i])
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

checking_gradient_descent(cost_history, predicted_values, actual_values)