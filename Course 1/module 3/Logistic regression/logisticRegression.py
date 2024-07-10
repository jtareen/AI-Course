# Still not completed | contain bugs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'car_data.csv')

data = pd.read_csv(csv_path)

data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

independet_variables = ['User ID', 'Gender', 'Age', 'AnnualSalary']

x_train = data[independet_variables].to_numpy()
y_train = np.array(data['Purchased'])

def sigmoid(z):
    # Clip values to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def pred_model(x, w, b):
    return sigmoid(np.dot(x, w) + b)

def loss_func(x, y, w, b):
    pred = pred_model(x, w, b)
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    res = -1 * ((y * np.log(pred)) + ((1 - y) * (np.log(1 - pred))))
    return res

def compute_cost(X, y, w, b):
    m = X.shape[0]

    cost = 0
    for i in range(m):
        cost += loss_func(X[i], y[i], w, b)
    cost = cost / m
    return cost

def compute_gradient(X, y, w, b):
    m,n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0.0
    for i in range(m):
        err = pred_model(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i][j] 
        dj_db += err

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, iters):

    w = w_in
    b = b_in
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        if i% 10 == 0:
            print(f"Iteration {i:4d}: Cost {cost:8.2f}   ")

    return w, b

total_features = x_train.shape[1]
print(total_features)

w_init = np.zeros(total_features)
b_init = 0.1
alpha = 1.0e-4
iterations = 60

w_model , b_model = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)

print(pred_model(x_train[0], w_model, b_model))