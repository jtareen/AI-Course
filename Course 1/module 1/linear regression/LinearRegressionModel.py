import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('E:/Courses/AI deeplearning.AI/Course 1/module 1/linear regression/deeplearning.mplstyle')

data = pd.read_csv("E:/Courses/AI deeplearning.AI/Course 1/module 1/linear regression/advertising.csv")
data.head()

# Extracting the Radio and Sales columns
radio = data['Radio'].values
sales = data['Sales'].values