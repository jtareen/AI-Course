import numpy as np
import matplotlib.pyplot as plt

x_values = np.array([120, 120, 140, 150, 170, 190, 190, 200, 220])
y_values = np.array([120000, 110000, 130000, 150000, 200000, 220000, 250000, 280000, 350000])


def der_wrt_w(x, y, w, b):
    m = len(x)
    sum = 0
    for i in range(0,m):
        sum += ((w * x[i] + b) - y[i]) * x[i]

    return sum / m

def der_wrt_b(x, y, w, b):
    m = len(x)
    sum = 0
    for i in range(0,m):
        sum += (w * x[i] + b) - y[i]
    return sum / m

def linear_Regression_model(x, y):
    w = 1
    b = 0
    alpha = 0.01
    iterations = 10
    for i in range(0,iterations):
        temp_w = w - (alpha * der_wrt_w(x, y, w, b))
        temp_b = b - (alpha * der_wrt_b(x, y, w, b))
        w = temp_w
        b = temp_b

    def model(x):
        return w * x + b

    return model

model1 = linear_Regression_model(x_values, y_values)