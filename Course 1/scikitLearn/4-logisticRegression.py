import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("e:/Courses/AI deeplearning.AI/Course 1/scikitLearn/data/car_data.csv")

data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

independet_variables = ['User ID', 'Gender', 'Age', 'AnnualSalary']

x_train = data[independet_variables].to_numpy()
y_train = np.array(data['Purchased'])

scaler = StandardScaler().fit(x_train)

x_scaled = scaler.transform(x_train)

logr = LogisticRegression()
logr.fit(x_scaled,y_train)

p = x_train[3]

predicted = logr.predict(p.reshape(-1,4))
print(predicted)