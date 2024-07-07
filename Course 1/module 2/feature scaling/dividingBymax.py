import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("E:/Courses/AI deeplearning.AI/Course 1/module 1/linear regression/advertising.csv")
data.head()

# Extracting the Radio and Sales columns
radio = data['Radio'].values
TV = data['TV'].values

radio_max = radio.max()
TV_max = TV.max()

radio_scaled = radio / radio_max
Tv_scaled = TV / TV_max

plt.subplot(1, 2, 1)
plt.scatter(radio, TV)
plt.title('Without Scaling')
plt.xlabel('radio')
plt.ylabel('TV')
plt.xlim(0,300)
plt.ylim(0,300)

plt.subplot(1, 2, 2)
plt.scatter(radio_scaled, Tv_scaled)
plt.title('With scaling (dividing by max)')
plt.xlabel('radio scaled')
plt.ylabel('TV scaled')
plt.xlim(0,1)
plt.ylim(0,1)

plt.show()