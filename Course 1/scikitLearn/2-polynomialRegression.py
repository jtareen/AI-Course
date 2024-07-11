from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data/advertising.csv')

data = pd.read_csv(csv_path)

# data set
x_train = np.array(data['TV']).reshape(-1, 1)
y_train = np.array(data['Sales'])

def train_polynomial_model(x_train, y_train, deg):
    # make a polynomial with degree 3
    model_polynomial_features = PolynomialFeatures(degree=deg)

    # make a polynomial model with a pipeline
    model = make_pipeline(model_polynomial_features, LinearRegression())

    # fit data to the model
    model.fit(x_train, y_train)

    return model

def visualize_model(x_train, y_train, x_pred, y_pred):
    plt.title('Linear regression visualization')

    plt.plot(x_pred, y_pred, color = 'red', label = 'prediction model')
    plt.scatter(x_train, y_train, color = 'blue', label = 'data points')

    plt.xlabel('TV')
    plt.ylabel('Price')

    plt.legend()
    plt.show()

# train model
model = train_polynomial_model(x_train, y_train, 5)

x_pred = np.linspace(x_train.min(), x_train.max(), 100)[:, np.newaxis]
y_pred = model.predict(x_pred)

print(r2_score(y_train, model.predict(x_train)))

visualize_model(x_train, y_train, x_pred, y_pred)