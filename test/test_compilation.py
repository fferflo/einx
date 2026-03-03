import einx
import numpy as np


def get_line_of(code, search):
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if search in line:
            return idx
    assert False


def test_compilation():
    # Ensure no-op is compiled to simple identity function
    x = np.zeros((2, 3, 4))
    code = einx.id("a b c", x, graph=True)
    assert code.count("reshape") == 0
    assert get_line_of(code, "def") + 1 == get_line_of(code, "return")

    # Ensure only a single reshape operation is used for multiple flattened axes
    x = np.zeros((2 * 3 * 4 * 5,))
    code = einx.id("(a (b (c d))) -> a b c d", x, graph=True, a=2, b=3, c=4, d=5)
    assert code.count("reshape") == 1
    code = einx.id("(a (b (c d))) -> (a b) (c d)", x, graph=True, a=2, b=3, c=4, d=5)
    assert code.count("reshape") == 1

    x = np.zeros((2, 3, 4, 5))
    code = einx.id("a b c d -> (a (b (c d)))", x, graph=True, a=2, b=3, c=4, d=5)
    assert code.count("reshape") == 1

    x = np.zeros((2 * 3, 4 * 5))
    code = einx.id("(a b) (c d) -> (a (b (c d)))", x, graph=True, a=2, b=3, c=4, d=5)
    assert code.count("reshape") == 1
    code = einx.id("(a b) (c d) -> a b c d", x, graph=True, a=2, b=3, c=4, d=5)
    assert code.count("reshape") == 1
