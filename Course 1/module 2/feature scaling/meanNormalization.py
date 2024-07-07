import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("E:/Courses/AI deeplearning.AI/Course 1/module 1/linear regression/advertising.csv")
data.head()

# Extracting the Radio and Sales columns
radio = data['Radio'].values
TV = data['TV'].values

radio_mean = radio.mean()
radio_max = radio.max()
radio_min = radio.min()

TV_mean = TV.mean()
TV_max = TV.max()
TV_min = TV.min()


radio_scaled = (radio - radio_mean) / (radio_max - radio_min)
Tv_scaled =  (TV - TV_mean) / (TV_max - TV_min)

plt.subplot(1, 2, 1)
plt.scatter(radio, TV)
plt.title('Without Scaling')
plt.xlabel('radio')
plt.ylabel('TV')
plt.xlim(0,300)
plt.ylim(0,300)

plt.subplot(1, 2, 2)
plt.scatter(radio_scaled, Tv_scaled)
plt.title('With scaling (Mean Normalization)')
plt.xlabel('radio scaled')
plt.ylabel('TV scaled')
plt.xlim(-1,1)
plt.ylim(-1,1)

plt.show()