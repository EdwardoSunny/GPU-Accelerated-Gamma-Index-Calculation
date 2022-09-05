from fast_interp import interp2d
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
z = np.array([1, 2, 3, 4, 5])


interpolater = interp2d([0, 0], x, y, z)
fe = interpolater([0, 0], [5, 5], [1, 1], )
