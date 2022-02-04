# Import libraries
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Accuracy and hyperparam vals
r_1_vals = np.array([2, 5, 6])
N_1_vals = np.array([30, 14, 17])
accuracies = np.array([0.28, 0.27, 0.22])


# Reshape arrays
X, Y = np.meshgrid(r_1_vals, N_1_vals)
Z = accuracies.reshape(X.shape)
#Z = accuracies

# Plot graph
fig,ax = plt.subplots(1, 1, figsize=(25, 5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, vmin=0, vmax=1, cmap=cm.coolwarm)
ax.set_title("placeholder")
#plt.legend()

# Save fig

plt.show()