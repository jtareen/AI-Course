import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("E:/Courses/AI deeplearning.AI/Course 1/module 1/linear regression/advertising.csv")
data.head()


def cost_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error += (sales[i] - (weight*radio[i] + bias))**2
    return total_error / companies

def update_wb(x, y, w, b, lr):
    m = len(x)
    der_w = 0
    der_b = 0
    for i in range(m):
        der_w += ((w * x[i] + b) - y[i]) * x[i]
        der_b += ((w * x[i] + b) - y[i])


    w -= lr * (der_w / m)
    b -= lr * (der_b / m)

    return w, b

def train(x, y, lr, iters):
    w = 0
    b = 0
    for i in range(iters):
        w,b = update_wb(x, y, w, b, lr)

    def model(x):
        print('w : ',w)
        print('b : ',b)
        return w * x + b

    print('cost of the function :',cost_function(x, y, w, b))
    return model

def visualize(x, y, model_func):
    y_val = model_func(x)

    plt.title('My First Model')
    plt.xlabel('Car Speed km/h')
    plt.ylabel('price in rupees')

    plt.scatter(x, y, color = 'green', label = 'training set')
    plt.plot(x, y_val, color = 'blue', label = 'linear regression model')

    #plt.ylim(-5,30)
    #plt.xlim(-10,60)

    plt.legend()
    plt.show()


# Extracting the Radio and Sales columns
radio = data['Radio'].values
sales = data['Sales'].values
iter = 10000
lr = 0.0000001

model1 = train(radio, sales, 0.01, 100)

visualize(radio, sales, model1)