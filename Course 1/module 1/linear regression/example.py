import matplotlib.pyplot as plt
import numpy as np

# Sample data for scatter points
x = np.linspace(0, 10, 30)
y = np.sin(x)

# Sample data for the line
x_line = np.linspace(0, 10, 100)
y_line = np.sin(x_line)

# Scatter points
plt.scatter(x, y, color='blue', label='Scatter Points')

# Line
plt.plot(x_line, y_line, color='red', label='Line')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Points and Line on One Graph')

plt.legend()

# Show the plot
plt.show()
