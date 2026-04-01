import einx
import numpy as np

x = np.zeros((2, 3))
y = 3.0

z1: np.ndarray = einx.add(None, x, y)
z2: np.ndarray = einx.add("a b,", x, y, a=None)
z3: np.ndarray = einx.add("a b,", x, y, backend=42)
