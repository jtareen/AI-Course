import numpy as np
import math, os
import matplotlib.pyplot as plt
import pandas as pd

data_relative_path = 'data.csv'
data_abs_path = os.path.abspath(data_relative_path)

data = pd.read_csv('E:\\Courses\\AI deeplearning.AI\\Course 1\\data.csv')

data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

x_train = np.array([data['Hours Studied'], data['Previous Scores'], data['Extracurricular Activities'], data['Sleep Hours'], data['Sample Question Papers Practiced']])
y_train = np.array(data['Performance Index'])

print(x_train)