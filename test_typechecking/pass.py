import einx
import numpy as np

x = np.zeros((2, 3))
y = 3.0

z: np.ndarray = einx.add("a b,", x, y)
