import matplotlib.pyplot as plt
import numpy as np
#plot 1:
x = np.array([0, 1, 1, 3])
y = np.array([3, 8, 1, 10])

scatter_values_x = np.array([3, 4, 5, 6, 2, 23, 78,5 ,5 ,65 ,54 , 94, 65, 25, 32, 45])
scatter_values_y = np.random.normal(70, 25, 16)

plt.subplot(1, 3, 1)
plt.plot(x,y)

plt.subplot(1, 3, 2)
plt.plot(x, y, color = 'green')

plt.subplot(1, 3, 3)
plt.scatter(scatter_values_x, scatter_values_y)

plt.show()