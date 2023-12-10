import einx
import numpy as np

def test_graph():
    x = np.zeros((10, 10))
    graph = einx.sum("a [b]", x, graph=True)
    str(graph)
