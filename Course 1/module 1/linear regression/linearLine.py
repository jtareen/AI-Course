import matplotlib.pyplot as plt
import numpy as np

x = np.array([120, 120, 140, 150, 170, 190, 190, 200, 220])
y = np.array([120000, 110000, 130000, 150000, 200000, 220000, 250000, 280000, 350000])

x_line = np.array([120, 220])
y_line = np.array([110000, 350000])

plt.title('linear regression rough example')
plt.xlabel('speed km/h')
plt.ylabel('price in RS')

plt.scatter(x, y, label = 'scatter points')
plt.plot(x_line, y_line, label = 'line')

plt.legend()

plt.show()