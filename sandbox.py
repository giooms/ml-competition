import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Sample data points
x = np.linspace(0, 10, 10)
y = np.sin(x)

# Linear interpolation
linear_interp = np.interp

# Spline interpolation
spline_interp = UnivariateSpline(x, y, s=0)

# Points to evaluate
x_eval = np.linspace(0, 10, 100)

# Evaluate interpolations
y_linear = linear_interp(x_eval, x, y)
y_spline = spline_interp(x_eval)

# Plotting
plt.plot(x, y, 'ro', label='Data points')
plt.plot(x_eval, y_linear, 'b-', label='Linear Interpolation')
plt.plot(x_eval, y_spline, 'g-', label='Spline Interpolation')
plt.legend()
plt.show()