import matplotlib.pyplot as plt
import numpy as np
import math

# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

def pred_func(x, w, b):
    return w * x + b

def cost_func(x, y, w, b):
    m = x.shape[0]
    err = (pred_func(x, w, b) - y)
    err_square = err**2 
    return sum(err_square) / (2 * m)

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    err = (pred_func(x, w, b) - y)
    dj_dw = sum(err * x) / m
    dj_db = sum(err) / m
    return dj_dw, dj_db
    
def gradient_descent(x, y, w_in, b_in, leraning_rate, iters):
    w = w_in
    b = b_in
    cost = cost_func(x, y, w, b)
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - (leraning_rate * dj_dw)
        b = b - (leraning_rate * dj_db)
        cost = cost_func(x, y, w, b)

        if i% math.ceil(iters/10) == 0:
            print(f"Iteration {i:4}: Cost {cost:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b

def plot(x, y, w, b):
    y_pred = pred_func(x, w, b)

    plt.scatter(x, y, color='red', label='data points')
    plt.plot(x, y_pred, label='prediction model')

    plt.title('Linear Regression')
    plt.xlabel('input')
    plt.ylabel('output')

    plt.legend()

    plt.show()

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2
w_final, b_final = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
plot(x_train, y_train, w_final, b_final)