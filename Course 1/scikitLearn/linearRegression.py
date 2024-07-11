import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv('E:/Courses/AI deeplearning.AI/Course 1/module 1/linear regression/advertising.csv')

# data set
x = np.array(data['TV']).reshape(-1,1)
y = np.array(data['Sales']).reshape(-1,1)

model = LinearRegression()

model.fit(x,y)

d = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])

print('w : ',model.coef_)
print('b : ',model.intercept_)

y_pred = model.coef_ * x + model.intercept_

plt.title('Linear regression visualization')

plt.plot(x,y_pred, color = 'red', label = 'prediction model')
plt.scatter(x,y, color = 'blue', label = 'data points')

plt.xlabel('TV')
plt.ylabel('Price')

plt.legend()
plt.show()