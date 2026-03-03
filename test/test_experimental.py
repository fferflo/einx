import conftest
import numpy as np
import importlib
import einx
import pytest


if importlib.util.find_spec("functorch"):
    try:
        import functorch
        import torch
        import functorch.dim as ftdim

        x = torch.zeros(2, 3)
        a = ftdim.Dim("a", 2)
        b = ftdim.Dim("b", 3)
        x = x[a, b]
        available = True
    except:
        available = False
    if available:

        def test_functorchdim():
            @einx.experimental.functorchdim.adapt
            def einfunc(x, y, *, axes):
                return (x + y).sum(axes[1])

            x = np.zeros((2, 3, 4))
            y = np.zeros((2, 6))
            assert einfunc("a [b c], a ([b] d) -> d [c] a", x, y).shape == (2, 4, 2)

            with pytest.raises(einx.errors.EinxError):
                einfunc("(a + 1) [b c], a ([b] d) -> d [c] a", x, y)
