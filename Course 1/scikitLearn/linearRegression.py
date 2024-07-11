import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import math, os
import pandas as pd

data = pd.read_csv('E:/Courses/AI deeplearning.AI/Course 1/module 1/linear regression/advertising.csv')

# data set
x = np.array(data['TV']).reshape(-1,1)
y = np.array(data['Sales']).reshape(-1,1)

model = LinearRegression()

model.fit(x,y)

print('w : ',model.intercept_)
print('b : ',model.coef_)


plt.scatter(x,y)

plt.show()