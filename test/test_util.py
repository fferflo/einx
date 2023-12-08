import einx
import numpy as np

def test_graph():
    x = np.zeros((10, 10))
    graph = einx.sum("a [b]", x, graph=True)
    str(graph)

def test_solve():
    x = np.ones((2, 3, 4))
    assert einx.matches("a b c", x)
    assert not einx.matches("a b", x)

    x = np.ones((6, 4))
    assert einx.matches("(a b) c", x)

    x = np.ones((2, 3, 4))
    assert einx.matches("a b...", x)

    x = np.ones((5, 4))
    assert einx.matches("(a + b) c", x)
    assert einx.matches("(a + b) c", x, a=2)
    assert not einx.matches("(a + b) c", x, a=10)