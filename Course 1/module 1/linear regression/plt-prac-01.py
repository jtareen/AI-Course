import matplotlib.pyplot as plt
import numpy as np


x_axis = np.array([3, 9])
y_axis = np.array([25, 92, 45, 56])

plt.plot( y_axis, marker = 'o')

plt.title("Practicing")
plt.xlabel("X values")
plt.ylabel("y values")

plt.grid(color = 'green', ls = '--')

plt.show()