from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# get csv path
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'advertising.csv')

# get data from csv through pandas
data = pd.read_csv(csv_path)

data.head()

# data set
x = np.array(data['TV']).reshape(-1,1)
y = np.array(data['Sales'])

def visualize_model(x_train, y_train, x_pred, y_pred):
    plt.title('Linear regression visualization')

    plt.plot(x_pred, y_pred, color = 'red', label = 'prediction model')
    plt.scatter(x_train, y_train, color = 'blue', label = 'data points')

    plt.xlabel('TV')
    plt.ylabel('Price')

    plt.legend()
    plt.show()

# make model
model = LinearRegression()

# feed data to the model
model.fit(x,y)

# predict values of y by model
y_pred = model.predict(x)

# check the score of predicted values of y againts actual values of y
print(r2_score(y, y_pred))

visualize_model(x, y, x, y_pred)